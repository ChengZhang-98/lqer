from pathlib import Path
import sys

sys.path.append(Path(__file__).resolve().parents[2].joinpath("src").as_posix())

from lqer.runners import run_pipeline
from lqer.logging import set_logging_verbosity

if __name__ == "__main__":
    set_logging_verbosity("info")
    run_pipeline()
