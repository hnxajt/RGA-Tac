"""Locations of the upstream data and model files."""

from dataclasses import dataclass
import json
from pathlib import Path

from rga_tac.integrations.tvl import BACKBONES, DEFAULT_BACKBONE
from rga_tac.paths import default_repo_root


@dataclass
class AssetPaths:
    dataset_root: Path
    tvl_root: Path
    llama_dir: Path
    tvl_encoder_checkpoint: Path
    tvl_llama_checkpoint: Path


def load_asset_paths(
    config_path: Path | None = None,
    *,
    repo_root: Path | None = None,
    backbone: str = DEFAULT_BACKBONE,
) -> AssetPaths:
    root = repo_root if repo_root is not None else default_repo_root()
    default_config = root / "configs" / "assets.local.json"
    if config_path is None and default_config.exists():
        config_path = default_config
    options = json.loads(config_path.read_text()) if config_path is not None else {}
    suffix = BACKBONES[backbone].checkpoint_suffix
    checkpoints = options.get("checkpoints", {}).get(backbone, {})
    if backbone == DEFAULT_BACKBONE:
        # Existing flat checkpoint entries configure the original Small workflow.
        checkpoints = options | checkpoints
    paths = {
        "dataset_root": options.get("dataset_root", "data/TVL-revise/tvl_dataset"),
        "tvl_root": options.get("tvl_root", "vendor/tvl"),
        "llama_dir": options.get("llama_dir", "weights/llama-2"),
        "tvl_encoder_checkpoint": checkpoints.get("tvl_encoder_checkpoint", f"weights/tvl_enc_{suffix}.pth"),
        "tvl_llama_checkpoint": checkpoints.get("tvl_llama_checkpoint", f"weights/tvl_llama_{suffix}.pth"),
    }
    return AssetPaths(**{
        name: (root / Path(value).expanduser()).resolve()
        for name, value in paths.items()
    })
