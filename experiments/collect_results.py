from pathlib import Path
import json
import collections.abc
import pandas as pd
from argparse import ArgumentParser

PREFIX_TO_REMOVE = Path("./").resolve().parents[1].as_posix()


def flatten_dict(d: dict, new_d: dict, join: str = ":", name: str = "root") -> dict:
    """
    Flatten a nested dict to a flat dict with keys joined by `join`.
    """
    for k, v in d.items():
        if isinstance(v, dict):
            flatten_dict(v, new_d, join, f"{name}{join}{k}")
        else:
            new_d[f"{name}{join}{k}"] = v


def collect_perplexity(path, json_name: str = "wikitext2.json") -> pd.DataFrame:
    result_dir = Path(path)
    json_files = result_dir.rglob(json_name)
    results = {}

    for json_file in json_files:
        with open(json_file, "r") as f:
            result = json.load(f)
            results[json_file.parent.as_posix().removeprefix(PREFIX_TO_REMOVE)] = result

    df = pd.DataFrame.from_dict(results, orient="index")
    df = df.sort_index()
    print(f"Collected results\n {df.to_markdown()}")
    return df


def collect_harness_results(
    path, json_name: str = "harness_results.json"
) -> pd.DataFrame:
    result_dir = Path(path)
    json_files = result_dir.rglob(json_name)

    df = None
    for json_file in json_files:
        with open(json_file, "r") as f:
            result = json.load(f)["results"]
        index_name = json_file.parent.as_posix().removeprefix(PREFIX_TO_REMOVE)
        flatten_result = {}
        flatten_dict(result, flatten_result, join=":", name="")
        flatten_result = {"run_name": index_name, **flatten_result}
        if df is None:
            df = pd.DataFrame(flatten_result, index=[0])
        else:
            df = pd.concat(
                [df, pd.DataFrame(flatten_result, index=[0])],
                axis=0,
                ignore_index=True,
            )

    df.columns = [c.removeprefix(":") for c in df.columns]
    df = df.set_index("run_name")

    print(f"Collected results\n {df.to_markdown()}")
    return df


def main():
    parser = ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument(
        "--file-name",
        type=str,
        # choices=["wikitext2.json", "harness_results.json"],
        default="wikitext2.json",
    )
    parser.add_argument("--save-path", type=str, default=None)
    args = parser.parse_args()

    if args.save_path is None:
        file_name = Path(args.file_name).stem + ".csv"
        args.save_path = file_name
        print(f"No save path is given. Using {file_name} as save path.")

    if args.file_name == "wikitext2.json":
        df = collect_perplexity(args.path, args.file_name)
    elif "harness_results" in args.file_name and args.file_name.endswith(".json"):
        df = collect_harness_results(args.path, args.file_name)
    else:
        raise ValueError(f"Unknown file name: {args.file_name}")

    df.to_csv(args.save_path)
    print(f"Saved to {args.save_path}")


if __name__ == "__main__":
    main()
