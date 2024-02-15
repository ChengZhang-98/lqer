from pathlib import Path
import logging
import sys
from argparse import ArgumentParser
import json

from torch.utils.data import DataLoader
import ray
import modin.pandas as modin_pd
import pandas as pd

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)
from accelerate import dispatch_model

sys.path.append(Path(__file__).resolve().parents[2].joinpath("src").as_posix())

from lqer.statistic_profiler import register_threshold_hooks
from lqer.datasets import get_data_module
from lqer.evaluate import evaluate_perplexity
from lqer.utils import set_package_logging_verbosity

logger = logging.getLogger("lqer.Profile-LLM-INT8")

# fmt: off
DEFAULT = {
    "default-llama-7b": {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 0, 'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 0, 'model.layers.7': 0, 'model.layers.8': 0, 'model.layers.9': 0, 'model.layers.10': 0, 'model.layers.11': 1, 'model.layers.12': 1, 'model.layers.13': 1, 'model.layers.14': 1, 'model.layers.15': 1, 'model.layers.16': 1, 'model.layers.17': 1, 'model.layers.18': 1, 'model.layers.19': 1, 'model.layers.20': 1, 'model.layers.21': 1, 'model.layers.22': 2, 'model.layers.23': 2, 'model.layers.24': 2, 'model.layers.25': 2, 'model.layers.26': 2, 'model.layers.27': 2, 'model.layers.28': 2, 'model.layers.29': 2, 'model.layers.30': 2, 'model.layers.31': 2, 'model.norm': 2, 'lm_head': 2},
    "default-llama-13b": {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 0, 'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 0, 'model.layers.7': 0, 'model.layers.8': 0, 'model.layers.9': 0, 'model.layers.10': 0, 'model.layers.11': 0, 'model.layers.12': 0, 'model.layers.13': 0, 'model.layers.14': 1, 'model.layers.15': 1, 'model.layers.16': 1, 'model.layers.17': 1, 'model.layers.18': 1, 'model.layers.19': 1, 'model.layers.20': 1, 'model.layers.21': 1, 'model.layers.22': 1, 'model.layers.23': 1, 'model.layers.24': 1, 'model.layers.25': 1, 'model.layers.26': 1, 'model.layers.27': 1, 'model.layers.28': 1, 'model.layers.29': 2, 'model.layers.30': 2, 'model.layers.31': 2, 'model.layers.32': 2, 'model.layers.33': 2, 'model.layers.34': 2, 'model.layers.35': 2, 'model.layers.36': 2, 'model.layers.37': 2, 'model.layers.38': 2, 'model.layers.39': 2, 'model.norm': 2, 'lm_head': 2},
}
# fmt: on


def str_to_device_map(txt: str) -> dict:
    if txt == "auto":
        return "auto"
    elif txt.startswith("default"):
        error_msg = f"Invalid device map name {txt}"
        assert txt in DEFAULT, error_msg
        return DEFAULT[txt]
    else:
        return json.loads(txt)


if __name__ == "__main__":
    set_package_logging_verbosity("error")
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/opt-1.3b")
    parser.add_argument(
        "--dataset",
        type=str,
        default="slim_pajama_6b",
        choices=["slim_pajama_6b", "wikitext2"],
    )
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=6)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device_map", type=str_to_device_map, default="auto")
    parser.add_argument("--save_dir", type=str, default=None)

    args = parser.parse_args()

    logger.info(f"Arguments:\n {json.dumps(vars(args), indent=4)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, device_map=args.device_map
    )

    hook_factory = register_threshold_hooks(
        model=model,
        threshold=args.threshold,
        seq_len=args.seq_len,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    datamodule = get_data_module(
        name=args.dataset,
        tokenizer=tokenizer,
        padding="max_length",
        max_length=args.seq_len,
        num_raw_samples=args.num_samples * 4,
    )

    train_dataloader = DataLoader(
        datamodule["train"],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=8,
    )

    results = evaluate_perplexity(
        model=model,
        eval_dataloader=train_dataloader,
        num_samples=args.num_samples,
        progress_bar=True,
        input_device="cuda:0",
    )

    logger.info(f"Perplexity:\n {json.dumps(results['perplexity'], indent=4)}")

    threshold_dict = hook_factory.get_threshold_dict()

    keys = [
        "name",
        "weight_shape",
        "num_activation_columns_in_high_precision",
        "high_precision_weight_shape",
        "low_precision_weight_shape",
        "high_precision_activation_shape",
        "low_precision_activation_shape",
        "threshold",
        "seq_len",
    ]

    df = pd.DataFrame(data=None, columns=keys)
    for k, v in threshold_dict.items():
        df.loc[len(df)] = [k] + [v[key] for key in keys[1:]]

    ray.init()
    df = modin_pd.DataFrame(df)

    picked = df.loc[
        :,
        [
            "name",
            "weight_shape",
            "num_activation_columns_in_high_precision",
            "threshold",
            "seq_len",
        ],
    ]
    summary = picked.describe()

    logger.info(f"Threshold summary:\n{summary.to_markdown()}")

    if args.save_dir is not None:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        picked.to_csv(save_dir.joinpath("thresholds.csv"))
        summary.to_csv(save_dir.joinpath("thresholds_summary.csv"))
        df.to_pickle(save_dir.joinpath("complete_thresholds_df.pkl"))
        logger.info(f"Saved threshold results to {save_dir}")
