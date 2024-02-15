from .logging import root_logger
from pathlib import Path
import sys

# add lm-evaluation-harness to sys.path
sys.path.append(
    Path(__file__).parents[2].joinpath("submodules", "lm-evaluation-harness").as_posix()
)
