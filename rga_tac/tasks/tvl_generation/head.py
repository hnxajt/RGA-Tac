"""Generate tactile descriptions with the fine-tuned TVL-LLaMA model."""

import hashlib
import random
from types import SimpleNamespace

import numpy as np
import torch

from rga_tac.integrations.tvl import add_tvl_generation_paths
from .config import TVLGenerationConfig


class TVLGenerationHead:
    def __init__(self, config: TVLGenerationConfig):
        self.config = config
        add_tvl_generation_paths(config.tvl_root)
        import llama
        from tvl_enc import tacvis

        self.tacvis = tacvis
        self.format_prompt = llama.format_prompt
        args = SimpleNamespace(
            active_modality_names=["tactile", "vision"],
            lora_modality_names=["tactile", "vision"], has_lora=True, lora_rank=16,
            checkpoint_path=str(config.encoder_checkpoint_path), tactile_model=config.tactile_model,
        )
        self.model = llama.load(
            str(config.model_path), str(config.llama_dir), device=config.device,
            llama_type="llama-2-7b", args=args,
        )
        self.model.eval()

    def generate(self, record: dict, *, prompt: str) -> str:
        digest = hashlib.sha256(f"{self.config.generation_seed}:{record['evidence_id']}".encode()).digest()
        seed = int.from_bytes(digest[:8], "big") % (2**32)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        inputs = {
            "vision": [self.tacvis.load_vision_data(record["image_path"], dataset_version="v1", device=self.config.device).unsqueeze(0), 1],
            "tactile": [self.tacvis.load_tactile_data(record["tactile_path"], device=self.config.device).unsqueeze(0), 1],
        }
        return self.model.generate(inputs, [self.format_prompt(prompt)], max_gen_len=self.config.max_gen_len)[0].strip()
