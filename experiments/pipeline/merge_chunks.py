from pathlib import Path
import sys

sys.path.append(Path(__file__).resolve().parents[2].joinpath("src").as_posix())

from lqer.chunked_runners import merge_chunks
from lqer.logging import set_logging_verbosity

if __name__ == "__main__":
    set_logging_verbosity("info")
    merge_chunks()
