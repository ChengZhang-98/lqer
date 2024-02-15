import os
import math
from pathlib import Path
import logging
import json
import gc
from argparse import ArgumentParser

import pandas as pd
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
    setattr_recursive,
    load_config,
)

logger = logging.getLogger(__name__)


def run_approximator_chunk() -> None:
    set_package_logging_verbosity("error")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    config, pipeline_prj_pth = parse_args(action="approximate")
    project_path = pipeline_prj_pth.parent.joinpath("approximate")
    logger.info("🚀 Approximating...")

    model: torch.nn.Module = AutoModelForCausalLM.from_pretrained(config["model_name"])
    state_dict = model.state_dict()

    approximator_cls = get_model_approximator_cls(config["approximate"]["name"])
    approximator = approximator_cls(state_dict, config["approximate"])

    # load weight scale dict if SVD-scaled
    if approximator.requires_scale_dict:
        if not Path(config["profile"]["scale_dict"]).exists():
            txt = f"scale_dict does not exist: {config['profile']['scale_dict']}, but required by {config['approximate']['name']}."
            raise FileNotFoundError(txt)
        approximator.load_scale_dict(torch.load(config["profile"]["scale_dict"]))

    total_num_weights = len(approximator.approximators)
    chunk_size = config["approximate"]["chunk_size"]
    chunk_idx = config["approximate"]["chunk_idx"]
    total_groups = math.ceil(total_num_weights / chunk_size)

    assert chunk_size > 0, "chunk_size must be positive"
    assert chunk_size <= total_num_weights, "chunk_size must be smaller than total_num_weights"
    assert chunk_idx >= 0, "chunk_idx must be non-negative"
    assert chunk_idx <= total_num_weights // chunk_size, "chunk_idx must be smaller than total_num_weights // chunk_size"

    approximator.approximators = dict(
        list(approximator.approximators.items())[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]
    )

    ret_vals = approximator.compute(delete_after_compute=True)
    result_df = ret_vals["df"]
    low_rank_dict = ret_vals["low_rank_dict"]
    error_T_dict = ret_vals["error_T_dict"]

    num_bits = math.ceil(math.log10(total_groups + 1))
    chunk_label = f"chunk_{chunk_idx:0{num_bits}}_of_{total_groups-1}"

    low_rank_dict_dir = project_path / "low_rank_dict"
    error_T_dict_dir = project_path / "error_T_dict"
    result_df_dir = project_path / "results"
    scale_dict_dir = project_path / "scale_dict"
    config_dir = project_path / "config"

    low_rank_dict_path = low_rank_dict_dir / f"{chunk_label}.pt"
    error_T_dict_path = error_T_dict_dir / f"{chunk_label}.pt"
    result_df_path = result_df_dir / f"{chunk_label}.pkl"
    scale_dict_path = scale_dict_dir / f"{chunk_label}.pt"
    config_path = config_dir / f"{chunk_label}.toml"

    low_rank_dict_dir.mkdir(parents=True, exist_ok=True)
    error_T_dict_dir.mkdir(parents=True, exist_ok=True)
    result_df_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    torch.save(low_rank_dict, low_rank_dict_path)
    torch.save(error_T_dict, error_T_dict_path)
    result_df.to_pickle(result_df_path)

    config["evaluate"]["low_rank_dict"] = low_rank_dict_path.as_posix()
    if "visualize" not in config:
        config["visualize"] = {}
    config["visualize"]["error_T_dict"] = error_T_dict_path.as_posix()

    if "scale_dict" in ret_vals:
        scale_dict_dir.mkdir(parents=True, exist_ok=True)
        torch.save(ret_vals["scale_dict"], scale_dict_path)
        config["approximate"]["sgd_svd"]["scale_dict"] = scale_dict_path.as_posix()

    save_config(config, config_path)

    logger.info("✅ Done")


def merge_chunks() -> None:
    parser = ArgumentParser(prog="merge_chunks")
    parser.add_argument("project_dir", type=str)
    args = parser.parse_args()

    prj_dir = Path(args.project_dir)
    approx_dir = prj_dir / "approximate"
    assert approx_dir.exists(), f"directory of approximated low-rank weights does not exist: {approx_dir}"

    logger.info("🚀 Merging approximation chunks...")

    # merge config
    config_dir = approx_dir / "config"
    configs = list(config_dir.glob("chunk_*_of_*.toml"))
    total_chunks = int(configs[0].stem.split("_")[-1].removesuffix(".toml")) + 1
    assert (
        len(configs) == total_chunks
    ), f"number of config files does not match total chunks: {len(configs)} != {total_chunks}"
    configs = sorted(configs, key=lambda x: int(x.stem.removeprefix("chunk_").split("_")[0]))
    merged_cfg = load_config(configs[0])
    low_rank_dict = []
    error_T_dict = []
    scale_dict = []

    for config_pth_i in configs:
        config = load_config(config_pth_i)
        low_rank_dict.append(config["evaluate"]["low_rank_dict"])
        error_T_dict.append(config["visualize"]["error_T_dict"])
        scale_dict.append(config["approximate"]["sgd_svd"]["scale_dict"])

    merged_cfg["evaluate"]["low_rank_dict"] = low_rank_dict
    merged_cfg["visualize"]["error_T_dict"] = error_T_dict
    merged_cfg["approximate"]["sgd_svd"]["scale_dict"] = scale_dict
    save_config(merged_cfg, prj_dir / "pipeline" / "config.toml")

    # merge result df
    result_df_dir = approx_dir / "results"
    result_dfs = list(result_df_dir.glob("chunk_*_of_*.pkl"))
    assert (
        len(result_dfs) == total_chunks
    ), f"number of result df files does not match total chunks: {len(result_dfs)} != {total_chunks}"
    result_dfs = sorted(result_dfs, key=lambda x: int(x.stem.removeprefix("chunk_").split("_")[0]))
    merged_df = None
    for result_df in result_dfs:
        df = pd.read_pickle(result_df)
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.concat([merged_df, df], ignore_index=True)

    merged_df.to_pickle(approx_dir / "results.pkl")
    ray.init(logging_level=logging.ERROR)
    result_df = modin_pd.DataFrame(merged_df)
    df_describe = result_df.describe()
    logger.info(
        "result summary: \n{}".format(
            df_describe.to_markdown(tablefmt="github", floatfmt=".6f"),
        )
    )
    df_describe.to_csv(approx_dir / "results_summary.csv")
    logger.info("✅ Done")
