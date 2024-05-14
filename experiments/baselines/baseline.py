"""
Run HuggingFace AWQ/LLM.int8()/GPTQ models on downstream tasks and evaluate perplexity

"""

import os
import json
from pathlib import Path
import logging
import sys
import ast

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GPTQConfig,
    BitsAndBytesConfig,
)

try:
    from auto_gptq import exllama_set_max_input_length
except ImportError:
    auto_gptq_is_available = False
else:
    auto_gptq_is_available = True

import wandb
from torch.utils.data import DataLoader

sys.path.append(Path(__file__).resolve().parents[2].joinpath("src").as_posix())
from lqer.utils import (
    parse_args,
    set_package_logging_verbosity,
)
from lqer.evaluate import (
    evaluate_harness_downstream,
    evaluate_perplexity,
    harness_make_table,
)
from lqer.datasets import get_data_module
from lqer.utils import save_config

logger = logging.getLogger("lqer.EVALUATE_BASELINES")

MAX_SEQ_LENGTH = 42688


def get_fp32_model(model_name, device_map):
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)
    return model


def get_fp16_model(model_name, device_map):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map=device_map, torch_dtype=torch.float16
    )
    return model


def get_awq_model(model_name, device_map):
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)
    msg = f"get model {model_name} with quantization config: {model.config.quantization_config['quant_method']}"
    assert model.config.quantization_config["quant_method"] == "awq", msg

    return model


def get_gptq_model(model_name, device_map):
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)
    msg = f"get model {model_name} with quantization config: {model.config.quantization_config}"
    assert isinstance(model.config.quantization_config, GPTQConfig), msg

    return model


def get_llm_int8_model(model_name, device_map):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map=device_map, load_in_8bit=True
    )

    return model


def get_llm_int4_model(model_name, device_map):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map=device_map, load_in_4bit=True, torch_dtype=torch.float16
    )
    return model


def get_baseline_model(q_method, name, device_map):
    match q_method:
        case "fp16":
            return get_fp16_model(name, device_map)
        case "fp32":
            return get_fp32_model(name, device_map)
        case "awq":
            return get_awq_model(name, device_map)
        case "gptq":
            assert auto_gptq_is_available, "auto_gptq is not available"
            return get_gptq_model(name, device_map)
        case "llm_int8":
            return get_llm_int8_model(name, device_map)
        case "llm_int4":
            return get_llm_int4_model(name, device_map)
        case _:
            raise ValueError(f"method {q_method} is not supported")


def main():
    set_package_logging_verbosity("error")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    config, pipeline_prj_dir = parse_args(action="pipeline")
    eval_config = config["evaluate"]

    eval_ppl_pth = pipeline_prj_dir.parent.joinpath("eval_ppl")
    eval_harness_downstream_pth = pipeline_prj_dir.parent.joinpath(
        "eval_harness_downstream"
    )

    # model
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.model_max_length > MAX_SEQ_LENGTH:
        logger.warning(
            f"tokenizer.model_max_length is set to {MAX_SEQ_LENGTH}, original value: {tokenizer.model_max_length}"
        )
        tokenizer.model_max_length = MAX_SEQ_LENGTH

    device_map = eval_config.get("device_map", "auto")
    if device_map != "auto":
        device_map = ast.literal_eval(eval_config["device_map"].removeprefix(":ast:"))

    logger.info(
        f"Creating model ({eval_config['hf_quant_method']}): {config['model_name']}"
    )
    logger.info(f"device_map: {device_map}, q_method: {eval_config['hf_quant_method']}")
    model = get_baseline_model(
        eval_config["hf_quant_method"],
        config["model_name"],
        device_map=device_map,
    )
    if eval_config["hf_quant_method"] == "gptq":
        try:
            model = exllama_set_max_input_length(
                model, max_input_length=tokenizer.model_max_length
            )
        except ValueError as e:
            logger.warning(f"exllama_set_max_input_length failed: {e}, skip this step")

    enable_wandb = config["enable_wandb"]

    if enable_wandb:
        tags = config["wandb"].get("tags", [])
        run = wandb.init(
            project=config["wandb"]["project"],
            entity=config["wandb"].get("entity", None),
            job_type=config["wandb"].get("job_type", None),
            tags=list(set(tags + ["pipeline"] + config.get("tags", []))),
        )

    # evaluate perplexity
    if config.get("enable_perplexity_evaluation", False):
        eval_ppl_config = eval_config["perplexity"]
        data_module = get_data_module(
            name=eval_ppl_config["dataset"],
            tokenizer=tokenizer,
            padding=eval_ppl_config.get("padding", "max_length"),
            max_length=eval_ppl_config.get("max_length", 2048),
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        test_dataloader = DataLoader(
            dataset=data_module["test"],
            batch_size=eval_ppl_config.get("batch_size", 2),
            shuffle=False,
            collate_fn=data_collator,
            num_workers=eval_ppl_config.get("num_workers", 8),
        )
        results = evaluate_perplexity(
            model=model,
            eval_dataloader=test_dataloader,
            num_samples=eval_ppl_config.get("num_samples", None),
            progress_bar=eval_ppl_config.get("progress_bar", True),
            input_device="cuda:0",
        )

        dumped = json.dumps(results, indent=4)
        logger.info(f"results: \n{dumped}")

        eval_ppl_pth.mkdir(parents=True, exist_ok=True)
        result_save_path = eval_ppl_pth.joinpath(
            eval_ppl_config["dataset"].replace("/", "_") + ".json"
        )
        with open(result_save_path, "w") as f:
            f.write(dumped)

        if enable_wandb:
            table = wandb.Table(columns=["entry", "value"])
            for k, v in results.items():
                table.add_data(k, v)
            wandb.log({f"{eval_ppl_config['dataset']}_results": table})
            wandb.run.summary[f"{eval_ppl_config['dataset']}_ppl"] = results[
                "perplexity"
            ]

    # evaluate downstream tasks
    if config.get("enable_harness_downstream_evaluation", False):
        eval_hd_config = eval_config["harness_downstream"]

        results = evaluate_harness_downstream(
            model=model,
            tasks=eval_hd_config["datasets"],
            num_fewshot=eval_hd_config.get("num_fewshot", 0),
            no_cache=eval_hd_config.get("no_cache", True),
            batch_size=eval_hd_config.get("batch_size", None),
        )

        dumped = json.dumps(results, indent=4)
        logger.info("\n" + harness_make_table(results))
        eval_harness_downstream_pth.mkdir(parents=True, exist_ok=True)
        save_path = eval_harness_downstream_pth / "harness_results.json"
        if save_path.exists():
            save_path = save_path.parent.joinpath(
                f"harness_results_{len(list(eval_harness_downstream_pth.glob('harness_results_*.json')))}.json"
            )
        with open(save_path, "w") as f:
            f.write(dumped)
        logger.info(f"results saved to {save_path}")

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

        accu_sum = 0
        task_cnt = 0
        for task in eval_hd_config["datasets"]:
            if "acc" not in results["results"][task]:
                continue
            accu_sum += results["results"][task]["acc"]
            task_cnt += 1
        logger.info(f"avg_harness_acc: {accu_sum / task_cnt}")

    # save config to pipeline dir
    save_config(config, pipeline_prj_dir / "config.toml")


if __name__ == "__main__":
    main()
