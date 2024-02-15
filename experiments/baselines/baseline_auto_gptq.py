"""
Run AutoGPTQ models uploaded to HuggingFace on downstream tasks and evaluate perplexity
"""

import os
import json
from pathlib import Path
import logging
import sys
import ast

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GPTQConfig,
    BitsAndBytesConfig,
)
from accelerate import dispatch_model, infer_auto_device_map
from auto_gptq import exllama_set_max_input_length, AutoGPTQForCausalLM
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

logger = logging.getLogger("lqer.EVALUATE_BASELINES")

MAX_SEQ_LENGTH = 42688


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def get_baseline_model(model_name, device_map, tokenizer):
    model = AutoGPTQForCausalLM.from_quantized(
        model_name, device="cuda:0", model_basename="model"
    )
    return model


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
    else:
        raise ValueError("device_map must be specified")

    logger.info(
        f"Creating model ({eval_config['hf_quant_method']}): {config['model_name']}"
    )
    logger.info(f"device_map: {device_map}")
    model = get_baseline_model(
        config["model_name"],
        device_map=device_map,
        tokenizer=tokenizer,
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


if __name__ == "__main__":
    main()
