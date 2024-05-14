import os
from pathlib import Path
import logging
import json
import gc

import wandb
import modin.pandas as modin_pd
import ray
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from torch.utils.data import DataLoader
from accelerate import (
    dispatch_model,
)

from .statistic_profiler import register_scale_hooks
from .datasets import get_data_module
from .evaluate import (
    evaluate_perplexity,
    evaluate_harness_downstream,
    harness_make_table,
)
from .approximate import get_model_approximator_cls
from .models import quantize_model
from .utils import (
    parse_args,
    save_config,
    set_package_logging_verbosity,
    create_device_map,
)

logger = logging.getLogger(__name__)


def _load_tensor_dict(path: str | list[str]) -> dict[str, torch.Tensor]:
    """
    Load a pickled tensor dict from path, or merge multiple pickled tensor dicts from a list of paths.
    """
    if isinstance(path, str) or (isinstance(path, Path) and path.is_file()):
        return torch.load(path)
    elif isinstance(path, (list, tuple)):
        ret = {}
        for p in path:
            ret.update(torch.load(p))
        return ret
    else:
        raise TypeError(f"Unsupported type: {type(path)}")


def run_profiler(config: dict, project_path) -> dict:
    """
    Append register hooks to linear layers, record `max([mean(abs(X), keepdim=-1) for X in CalibrationSet])`

    ---
    Performance:
    - No significant overhead as model layers + hooks are distributed across GPUs in model parallel mode.
    """
    profile_config = config["profile"]
    dtype = getattr(torch, profile_config.get("dtype", "float32"))

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"], torch_dtype=dtype, _attn_implementation="eager"
    )
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    device_map = create_device_map(
        model=model,
        max_memory=profile_config.get("max_memory", None),
        device_map=profile_config.get("device_map", None),
    )
    logger.info(f"dtype: {dtype}, device_map: {device_map}")
    model = dispatch_model(model, device_map=device_map)

    scaling_mode = profile_config.get("scaling_mode", "mean(abs())")
    logger.info(f"scaling_mode: {scaling_mode}")
    profiler_factory = register_scale_hooks(model, scaling_mode)

    dataset_dict = get_data_module(
        name=profile_config["dataset"],
        tokenizer=tokenizer,
        padding=profile_config.get("padding", "max_length"),
        max_length=profile_config.get("max_length", 2048),
        num_raw_samples=profile_config.get("num_raw_samples", None),
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_dataloader = DataLoader(
        dataset_dict["train"],
        batch_size=profile_config.get("batch_size", 4),
        shuffle=False,
        collate_fn=data_collator,
        num_workers=profile_config.get("num_workers", 4),
    )

    results = evaluate_perplexity(
        model=model,
        eval_dataloader=train_dataloader,
        num_samples=profile_config["num_samples"],
        progress_bar=profile_config.get("progress_bar", True),
        description=f"Profiling on {profile_config['dataset']}...",
    )

    logger.info(f"results: \n{json.dumps(results, indent=4)}")

    scale_dict = profiler_factory.get_scale_dict()
    scale_dict = {k: v.cpu() for k, v in scale_dict.items()}

    torch.save(
        scale_dict,
        project_path / f"scale_dict.pt",
    )
    config["profile"]["scale_dict"] = project_path.joinpath("scale_dict.pt").as_posix()

    return config


def run_approximator(config: dict, project_path: Path) -> dict:
    ray.init(logging_level=logging.ERROR)
    dtype = getattr(torch, config["profile"].get("dtype", "float32"))
    model: torch.nn.Module = AutoModelForCausalLM.from_pretrained(
        config["model_name"], torch_dtype=dtype, _attn_implementation="eager"
    )
    state_dict = model.state_dict()

    approximator_cls = get_model_approximator_cls(config["approximate"]["name"])
    approximator = approximator_cls(state_dict, config["approximate"])

    logger.info(f"dtype: {dtype}")
    # load weight scale dict if SVD-scaled
    if approximator.requires_scale_dict:
        if not Path(config["profile"]["scale_dict"]).exists():
            txt = f"scale_dict does not exist: {config['profile']['scale_dict']}, but required by {config['approximate']['name']}."
            raise FileNotFoundError(txt)
        state_dict = _load_tensor_dict(config["profile"]["scale_dict"])
        approximator.load_scale_dict(state_dict)

    approximator.to(dtype)

    ret_vals = approximator.compute(delete_after_compute=True)
    result_df = ret_vals["df"]
    low_rank_dict = ret_vals["low_rank_dict"]
    error_T_dict = ret_vals["error_T_dict"]

    if config["enable_wandb"]:
        table = wandb.Table(data=result_df)
        wandb.log({"1/n * ||AB - Q_error^T||_1": table})

    # save low_rank_dict, error_T_dict, and result_df
    low_rank_dict_path = project_path / "low_rank_dict.pt"
    error_T_dict_path = project_path / "error_T_dict.pt"
    result_df_path = project_path / "results.pkl"

    torch.save(low_rank_dict, low_rank_dict_path)
    config["evaluate"]["low_rank_dict"] = low_rank_dict_path.as_posix()
    torch.save(error_T_dict, error_T_dict_path)
    if "visualize" not in config:
        config["visualize"] = {}
    config["visualize"]["error_T_dict"] = error_T_dict_path.as_posix()
    result_df.to_pickle(result_df_path)
    if "scale_dict" in ret_vals:
        scale_dict_path = project_path / "scale_dict.pt"
        torch.save(ret_vals["scale_dict"], scale_dict_path)
        config["approximate"]["sgd_svd"]["scale_dict"] = scale_dict_path.as_posix()

    del error_T_dict
    del approximator
    gc.collect()
    logger.info("results: \n{}".format(result_df.to_markdown(tablefmt="github")))

    result_df = modin_pd.DataFrame(result_df)
    df_describe = result_df.describe()
    logger.info(
        "result summary: \n{}".format(
            df_describe.to_markdown(tablefmt="github", floatfmt=".6f"),
        )
    )
    df_describe.to_csv(project_path / "results_summary.csv")

    if config["enable_wandb"]:
        wandb.run.summary["avg_abs_error"] = result_df.loc[
            :, "l1_norm(AB-Q_error_T)/n"
        ].mean()

    ray.shutdown()
    return config


def run_evaluate_perplexity(config: dict, project_path: Path) -> dict:
    """
    Evaluate perplexity on test set.

    - Note that max_token_len=2048, which is memory-consuming.
    """
    eval_config = config["evaluate"]
    eval_ppl_config = eval_config["perplexity"]
    dtype = getattr(torch, eval_config.get("dtype", "float16"))

    disable_lqer = eval_config["disable_lqer"]

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"], torch_dtype=dtype, _attn_implementation="eager"
    )
    # create FP32 model, WxAy model (no Ak Bk), or LQER model
    quantize_model(
        model,
        q_config=config["q_config"],
        l_config=config.get("l_config", None),
    )
    if disable_lqer:
        logger.info("🔉 LQER disabled. Evaluating WxAy without Ak Bk")
    else:
        AB_dict = _load_tensor_dict(eval_config["low_rank_dict"])
        AB_dict = {k: v.to(dtype) for k, v in AB_dict.items()}
        model.load_state_dict(AB_dict, strict=False)
        logger.info("🔉 Evaluating LQER model")

    if hasattr(model, "tie_weights"):
        model.tie_weights()
    device_map = create_device_map(
        model,
        eval_ppl_config.get("max_memory", None),
        eval_ppl_config.get("device_map", None),
    )
    logger.info(f"dtype: {dtype}, device_map: {device_map}")

    model = dispatch_model(model, device_map=device_map)

    data_module = get_data_module(
        name=eval_ppl_config["dataset"],
        tokenizer=tokenizer,
        padding=eval_ppl_config.get("padding", "max_length"),
        max_length=eval_ppl_config.get("max_length", 2048),
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    test_dataloader = DataLoader(
        data_module["test"],
        batch_size=eval_ppl_config.get("batch_size", 4),
        shuffle=False,
        collate_fn=data_collator,
        num_workers=eval_ppl_config.get("num_workers", 8),
    )

    results = evaluate_perplexity(
        model=model,
        eval_dataloader=test_dataloader,
        num_samples=eval_ppl_config.get("num_samples", None),
        progress_bar=eval_ppl_config.get("progress_bar", True),
        description=f"Evaluating perplexity on {eval_ppl_config['dataset']}...",
    )

    del model
    gc.collect()

    logger.info(f"results: \n{json.dumps(results, indent=4)}")

    save_file = project_path.joinpath(
        eval_ppl_config["dataset"].replace("/", "_") + ".json"
    )
    with open(save_file, "w") as f:
        json.dump(results, f, indent=4)

    if config["enable_wandb"]:
        table = wandb.Table(columns=["entry", "value"])
        for k, v in results.items():
            table.add_data(k, v)
        wandb.log({f"{eval_ppl_config['dataset']}_results": table})
        wandb.run.summary[f"{eval_ppl_config['dataset']}_ppl"] = results["perplexity"]

    return config


def run_evaluate_harness_downstream(config: dict, project_path: Path) -> dict:
    eval_config = config["evaluate"]
    eval_hd_config = eval_config["harness_downstream"]
    disable_lqer = eval_config["disable_lqer"]

    dtype = getattr(torch, eval_config.get("dtype", "float16"))
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"], torch_dtype=dtype, _attn_implementation="eager"
    )
    quantize_model(
        model,
        q_config=config["q_config"],
        l_config=config.get("l_config", None),
    )
    if disable_lqer:
        logger.info("🔉 LQER disabled. Evaluating baseline WxAy without Ak Bk")
    else:
        AB_dict = _load_tensor_dict(eval_config["low_rank_dict"])
        AB_dict = {k: v.to(dtype) for k, v in AB_dict.items()}
        model.load_state_dict(AB_dict, strict=False)
        logger.info("🔉 Evaluating LQER model")

    device_map = create_device_map(
        model,
        eval_hd_config.get("max_memory", None),
        eval_hd_config.get("device_map", None),
    )
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    model = dispatch_model(model, device_map=device_map)

    results = evaluate_harness_downstream(
        model,
        tasks=eval_hd_config["datasets"],
        num_fewshot=eval_hd_config.get("num_fewshot", 0),
        no_cache=eval_hd_config.get("no_cache", True),
        batch_size=eval_hd_config.get("batch_size", None),
    )

    dumped = json.dumps(results, indent=4)
    save_path = project_path.joinpath("harness_results.json")
    if save_path.exists():
        save_path = save_path.parent.joinpath(
            f"harness_results_{len(list(project_path.glob('harness_results_*.json')))}.json"
        )
    with open(save_path, "w") as f:
        f.write(dumped)
    logger.info(f"results saved to {save_path}")
    logger.info("\n" + harness_make_table(results))

    if config["enable_wandb"]:
        table = wandb.Table(columns=["dataset", "accuracy"])
        task_cnt = 0
        accu_sum = 0
        for task in eval_hd_config["datasets"]:
            task_acc = results["results"][task]["acc"]
            accu_sum += task_acc
            task_cnt += 1
            table.add_data(task, task_acc)
            wandb.run.summary[f"{task}_acc"] = task_acc
        wandb.run.summary["avg_harness_acc"] = accu_sum / task_cnt
        wandb.log({"harness_downstream_results": table})

    return config


def run_pipeline() -> None:
    """
    Profile -> Approximate -> Evaluate
    """
    set_package_logging_verbosity("error")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    config, pipeline_prj_pth = parse_args(action="pipeline")

    # fmt: off
    profile_pth = pipeline_prj_pth.parent.joinpath("profile")
    approx_pth = pipeline_prj_pth.parent.joinpath("approximate")
    eval_ppl_pth = pipeline_prj_pth.parent.joinpath("evaluate_perplexity")
    eval_harness_downstream_pth = pipeline_prj_pth.parent.joinpath("evaluate_harness_downstream")
    # fmt: on

    if config["enable_wandb"]:
        tags = config["wandb"].get("tags", [])
        run = wandb.init(
            project=config["wandb"]["project"],
            entity=config["wandb"].get("entity", None),
            job_type=config["wandb"].get("job_type", None),
            tags=list(set(tags + ["pipeline"] + config.get("tags", []))),
        )
    if config.get("enable_profiling", False):
        logger.info("🚀 Profiling...")
        profile_pth.mkdir(parents=True, exist_ok=True)
        config = run_profiler(config, profile_pth)
        config["enable_profiling"] = False
        save_config(config, pipeline_prj_pth / "config_after_profiling.toml")
    if config.get("enable_approximation", False):
        logger.info("🚀 Approximating...")
        approx_pth.mkdir(parents=True, exist_ok=True)
        config = run_approximator(config, approx_pth)
        config["enable_approximation"] = False
        save_config(config, pipeline_prj_pth / "config_after_approximation.toml")
    if config.get("enable_perplexity_evaluation", False):
        logger.info("🚀 Evaluating perplexity...")
        eval_ppl_pth.mkdir(parents=True, exist_ok=True)
        config = run_evaluate_perplexity(config, eval_ppl_pth)
        config["enable_perplexity_evaluation"] = False
        save_config(
            config, pipeline_prj_pth / "config_after_perplexity_evaluation.toml"
        )
    if config.get("enable_harness_downstream_evaluation", False):
        logger.info("🚀 Evaluating harness downstream...")
        eval_harness_downstream_pth.mkdir(parents=True, exist_ok=True)
        config = run_evaluate_harness_downstream(config, eval_harness_downstream_pth)
        config["enable_harness_downstream_evaluation"] = False
        save_config(
            config, pipeline_prj_pth / "config_after_harness_downstream_evaluation.toml"
        )

    save_config(config, pipeline_prj_pth / "config.toml")

    if config["enable_wandb"]:
        wandb.finish()

    logger.info("✅ Done.")
