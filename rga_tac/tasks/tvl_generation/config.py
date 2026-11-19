from dataclasses import dataclass
from pathlib import Path

from rga_tac.integrations.tvl import TACTILE_MODEL


@dataclass
class TVLGenerationConfig:
    model_path: Path
    llama_dir: Path
    encoder_checkpoint_path: Path
    tvl_root: Path
    tactile_model: str = TACTILE_MODEL
    max_gen_len: int = 256
    generation_seed: int = 0
    device: str = "cuda"
