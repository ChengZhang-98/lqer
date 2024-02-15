from pathlib import Path
from argparse import ArgumentParser


def convert_bytes_to_smart_size(num_bytes):
    if num_bytes < 1e3:
        return f"{num_bytes} B"
    elif num_bytes < 1e6:
        return f"{num_bytes / 1e3:.1f} KB"
    elif num_bytes < 1e9:
        return f"{num_bytes / 1e6:.1f} MB"
    elif num_bytes < 1e12:
        return f"{num_bytes / 1e9:.1f} GB"
    else:
        return f"{num_bytes / 1e12:.1f} TB"


def main():
    parser = ArgumentParser()
    parser.add_argument("path", type=str)

    args = parser.parse_args()
    pt_path = Path(args.path)

    assert pt_path.exists()

    bytes_removed = 0
    for p in pt_path.rglob("error_T_dict.pt"):
        print(f"Removing {p}")
        bytes_removed += p.stat().st_size
        p.unlink()

    print(f"Removed {convert_bytes_to_smart_size(bytes_removed)}")


if __name__ == "__main__":
    main()
