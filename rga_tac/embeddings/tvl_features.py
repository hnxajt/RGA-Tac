"""Encode the SSVTP visual and tactile images with the frozen TVL model."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from rga_tac.data.manifest.io import read_jsonl
from rga_tac.integrations.tvl import TACTILE_MODEL, add_tvl_encoder_path


@dataclass
class ExtractionConfig:
    manifest_path: Path
    output_dir: Path
    tvl_root: Path
    checkpoint_path: Path
    tactile_model: str = TACTILE_MODEL
    batch_size: int = 32
    device: str = "cuda"


class ManifestEmbeddingDataset(Dataset):
    def __init__(self, records, tacvis):
        self.records = records
        self.tacvis = tacvis

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        return {
            "visual": self.tacvis.load_vision_data(record["image_path"]),
            "tactile": self.tacvis.load_tactile_data(record["tactile_path"], transform_tac=self.tacvis.TAC_PREPROCESS),
        }


def extract_embedding_jobs(configs) -> None:
    configs = tuple(configs)
    first = configs[0]
    add_tvl_encoder_path(first.tvl_root)
    from tvl_enc import tacvis, tvl

    device = torch.device(first.device)
    model = tvl.TVL(active_modalities=[tvl.ModalityType.VISION, tvl.ModalityType.TACTILE], tactile_model=first.tactile_model)
    checkpoint = torch.load(first.checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)
    del checkpoint
    model.to(device).eval()
    for config in configs:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        records = read_jsonl(config.manifest_path)
        loader = DataLoader(ManifestEmbeddingDataset(records, tacvis), batch_size=config.batch_size, shuffle=False, pin_memory=True)
        collected = {"visual": [], "tactile": []}
        with torch.no_grad():
            for batch in loader:
                with torch.cuda.amp.autocast():
                    output = model({"vision": batch["visual"].to(device), "tactile": batch["tactile"].to(device)})
                collected["visual"].append(output["vision"].float().cpu().numpy())
                collected["tactile"].append(output["tactile"].float().cpu().numpy())
        arrays = {view: np.concatenate(values).astype(np.float16) for view, values in collected.items()}
        fused = np.concatenate([arrays["visual"].astype(np.float32), arrays["tactile"].astype(np.float32)], axis=1)
        arrays["fused"] = (fused / np.maximum(np.linalg.norm(fused, axis=1, keepdims=True), 1e-12)).astype(np.float16)
        for view, values in arrays.items():
            np.save(config.output_dir / f"{view}.float16.npy", values)
