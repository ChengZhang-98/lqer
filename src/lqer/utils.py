import sys
import logging
import tomllib
from copy import deepcopy
import re
from pathlib import Path
import toml
from pprint import pformat
import ast

import transformers
import datasets as hf_datasets
from argparse import ArgumentParser
from accelerate import infer_auto_device_map

from .registry import LQER_PATH

logger = logging.getLogger(__name__)


def flatten_dict(d: dict, new_d: dict, join: str = ":", name: str = "root") -> dict:
    """
    Flatten a nested dict to a flat dict with keys joined by `join`.
    """
    for k, v in d.items():
        if isinstance(v, dict):
            flatten_dict(v, new_d, join, f"{name}{join}{k}")
        else:
            new_d[f"{name}{join}{k}"] = v


def expand_dict(d: dict, new_d: dict, join: str = ":", name: str = "root"):
    """
    Expand a flat dict to a nested dict with keys joined by `join`.
    """

    def create_nested_dict(d: dict, key_list: list[str], value):
        if len(key_list) == 1:
            if key_list[0] not in d:
                d[key_list[0]] = value
            elif isinstance(d[key_list[0]], dict):
                d[key_list[0]].update(value)
            else:
                raise ValueError(
                    f"Cannot create nested dict at {key_list} with value {value}"
                )
        else:
            if key_list[0] not in d:
                d[key_list[0]] = {}
            create_nested_dict(d[key_list[0]], key_list[1:], value)

    for k, v in d.items():
        k: str
        key_list = k.removeprefix(f"{name}{join}").split(join)
        create_nested_dict(new_d, key_list, v)


def convert_str_na_to_none(d) -> dict | None:
    """
    Since toml does not support None, we use "NA" to represent None.
    """
    if isinstance(d, dict):
        for k, v in d.items():
            d[k] = convert_str_na_to_none(v)
    elif isinstance(d, list):
        d = [convert_str_na_to_none(v) for v in d]
    elif isinstance(d, tuple):
        d = tuple(convert_str_na_to_none(v) for v in d)
    else:
        if d == "NA":
            return None
        else:
            return d
    return d


def convert_none_to_str_na(d):
    """
    Since toml does not support None, we use "NA" to represent None.
    Otherwise the none-value key will be missing in the toml file.
    """
    if isinstance(d, dict):
        for k, v in d.items():
            d[k] = convert_none_to_str_na(v)
    elif isinstance(d, list):
        d = [convert_none_to_str_na(v) for v in d]
    elif isinstance(d, tuple):
        d = tuple(convert_none_to_str_na(v) for v in d)
    else:
        if d is None:
            return "NA"
        else:
            return d
    return d


def load_config(config_path) -> dict:
    """Load from a toml config file and convert "NA" to None."""
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    config = convert_str_na_to_none(config)
    return config


def save_config(config, config_path):
    """Convert None to "NA" and save to a toml config file."""
    config = convert_none_to_str_na(deepcopy(config))
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        toml.dump(config, f)


def find_matched_pattern(query: str, patterns: list[str]) -> str | None:
    patterns: list[re.Pattern] = [re.compile(pattern) for pattern in patterns]

    matched_patterns = []

    for pattern in patterns:
        if pattern.fullmatch(query):
            matched_patterns.append(pattern)

    if len(matched_patterns) > 1:
        raise ValueError(f"Multiple patterns matched: {matched_patterns}")

    return matched_patterns[0].pattern if len(matched_patterns) == 1 else None


def find_all_matched_patterns(query: str, patterns: list[str]) -> list[str] | None:
    patterns: list[re.Pattern] = [re.compile(pattern) for pattern in patterns]

    matched_patterns = []

    for pattern in patterns:
        if pattern.fullmatch(query):
            matched_patterns.append(pattern.pattern)

    return matched_patterns if len(matched_patterns) > 0 else None


def set_dict_value(config: dict, keys: list[str], value):
    if len(keys) == 1:
        config[keys[0]] = value
    else:
        if keys[0] not in config:
            config[keys[0]] = {}
        set_dict_value(config[keys[0]], keys[1:], value)


def get_dict_value(config: dict, keys: list[str]):
    if len(keys) == 1:
        return config[keys[0]]
    else:
        txt = f"Unknown key {keys[0]}."
        assert keys[0] in config, txt
        return get_dict_value(config[keys[0]], keys[1:])


def override_args(config: dict, unknown_args: list[str]) -> tuple[dict, dict]:
    override_args = {}

    for flag in unknown_args:
        assert flag.startswith("-") or flag.startswith("--"), f"Unknown flag {flag}."
        assert "=" in flag, f"Unknown flag {flag}."

        keys, value = flag.removeprefix("-").removeprefix("-").split("=")
        key_list = keys.split(":")
        if value.startswith(":ast:"):
            try:
                value = ast.literal_eval(value.removeprefix(":ast:"))
            except SyntaxError as e:
                logger.error(f"Error when parsing args. keys: {keys}, value: {value}")
                raise SyntaxError(f"Error when parsing {value}") from e
        else:
            value = type(get_dict_value(config, key_list))(value)

        set_dict_value(override_args, key_list, value)
        set_dict_value(config, key_list, value)

    # config.update(override_args)

    return config, override_args


def get_project_path(config: dict, tags: list[str], action: str) -> Path:
    if "checkpoint_path" not in config:
        tag = "_".join(tags).replace("/", "-")
        project_path = LQER_PATH.parents[1].joinpath(
            "checkpoints", config["project"].replace("/", "-"), tag, action
        )
    else:
        project_path = Path(config["checkpoint_path"]).resolve().joinpath(action)

    if project_path.exists() and any(project_path.iterdir()):
        if not config.get("overwrite_checkpoint", False):
            msg = f"Project path {project_path} exists but is not empty."
            raise RuntimeError(msg)
        else:
            logger.warning(
                f"Project path {project_path} exists but is not empty. Overwriting..."
            )

    if not project_path.exists():
        project_path.mkdir(parents=True, exist_ok=True)

    return project_path


def enable_exception_hook(debugger="ipdb"):
    from .logging import set_logging_verbosity

    if debugger == "pudb":

        def excepthook(etype, evalue, etb):
            from IPython.core import ultratb
            import pudb

            ultratb.FormattedTB()(etype, evalue, etb)
            for exc in [KeyboardInterrupt, FileNotFoundError]:
                if issubclass(etype, exc):
                    sys.exit(-1)
            pudb.post_mortem(etb)

    elif debugger == "ipdb":

        def excepthook(etype, evalue, etb):
            from IPython.core import ultratb
            import ipdb

            ultratb.FormattedTB()(etype, evalue, etb)
            for exc in [KeyboardInterrupt, FileNotFoundError]:
                if issubclass(etype, exc):
                    sys.exit(-1)
            ipdb.post_mortem(etb)

    else:
        raise ValueError(f"Unknown debugger: {debugger}")

    sys.excepthook = excepthook
    set_logging_verbosity("debug")
    logger.info("Enabled exception hook.")


def parse_args(action: str):
    parser = ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("tags", type=str, nargs="*")
    parser.add_argument("--debug", choices=["ipdb", "pudb"], default=None)
    args, unknown_args = parser.parse_known_args()
    if args.debug:
        enable_exception_hook(args.debug)

    config = load_config(args.config)
    config, overridden_args = override_args(config, unknown_args)
    if len(overridden_args) > 0:
        f_d = {}
        flatten_dict(overridden_args, f_d, join=":", name="")
        logger.info(f"overridden_args: \n{pformat(f_d, indent=2)}")

    tags = args.tags + config.get("tags", []) + config.get("wandb", {}).get("tags", [])
    project_path = get_project_path(config, tags=tags, action=action)
    config["wandb"]["tags"] += args.tags

    return config, project_path


def set_package_logging_verbosity(level: str = "info"):
    level = level.lower()
    match level:
        case "debug":
            transformers.logging.set_verbosity_debug()
            hf_datasets.logging.set_verbosity_debug()
        case "info":
            transformers.logging.set_verbosity_info()
            hf_datasets.logging.set_verbosity_info()
        case "warning":
            transformers.logging.set_verbosity_warning()
            hf_datasets.logging.set_verbosity_warning()
        case "error":
            transformers.logging.set_verbosity_error()
            hf_datasets.logging.set_verbosity_error()
        case _:
            raise ValueError(
                f"Unknown logging level: {level}, should be one of: debug, info, warning, error, critical"
            )
    logger.info(f"Set HuggingFace logging level to {level}")


def create_device_map(model, max_memory, device_map) -> dict[str, int]:
    if max_memory is not None:
        max_memory = ast.literal_eval(max_memory.removeprefix(":ast:"))

    if device_map is not None:
        if device_map == "auto":
            device_map = infer_auto_device_map(
                model,
                no_split_module_classes=model._no_split_modules,
                max_memory=max_memory,
            )
        elif isinstance(device_map, str):
            device_map = ast.literal_eval(device_map.removeprefix(":ast:"))
        else:
            assert isinstance(device_map, dict)
    elif max_memory is not None:
        device_map = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=model._no_split_modules,
        )
    else:
        device_map = infer_auto_device_map(
            model, no_split_module_classes=model._no_split_modules
        )
    return device_map


def setattr_recursive(obj, attr, value):
    if "." not in attr:
        setattr(obj, attr, value)
    else:
        layer = attr.split(".")
        setattr_recursive(getattr(obj, layer[0]), ".".join(layer[1:]), value)
