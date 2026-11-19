"""Import the upstream TVL checkout."""

from pathlib import Path
import sys


def add_tvl_encoder_path(tvl_root: Path) -> None:
    sys.path.insert(0, str(tvl_root.resolve()))


def add_tvl_generation_paths(tvl_root: Path) -> None:
    root = tvl_root.resolve()
    sys.path[:0] = [str(root), str(root / "tvl_llama")]
