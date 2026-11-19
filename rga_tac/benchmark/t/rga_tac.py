"""SSVTP evidence-conditioned fine-tuning."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, replace
import hashlib
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Sequence

import numpy as np

from rga_tac.assets import load_asset_paths
from rga_tac.data.manifest.builder import prepare_manifests
from rga_tac.data.manifest.io import read_jsonl, write_jsonl
from rga_tac.integrations.tvl import BACKBONES, DEFAULT_BACKBONE, add_backbone_argument
from rga_tac.paths import default_repo_root, ssvtp_output_root
from rga_tac.retrieval.candidate_pool import CandidatePoolConfig, build_candidate_pools
from rga_tac.retrieval.s_graph import select_s_graph
from rga_tac.tasks.tvl_generation.prompt_constants import DEFAULT_BASE_PROMPT


@dataclass
class RGATacConfig:
    output_root: Path
    source_train_manifest: Path | None = None
    ssvtp_source_json: Path | None = None
    asset_config: Path | None = None
    dataset_root: Path | None = None
    tvl_root: Path | None = None
    llama_dir: Path | None = None
    encoder_checkpoint_path: Path | None = None
    tvl_llama_checkpoint: Path | None = None
    training_config: Path = Path("configs/t/training/rga_tac_small_train.json")
    exclusion_config: Path = Path("configs/t/data/rga_tac_ssvtp_quarantine.json")
    device: str = "cuda"
    backbone: str = DEFAULT_BACKBONE

    @property
    def tactile_model(self) -> str:
        return BACKBONES[self.backbone].tactile_model

    @property
    def sft_bank_jsonl(self) -> Path:
        return self.output_root / "sft_retrieval" / "bank" / "evidence_train.jsonl"

    @property
    def sft_query_jsonl(self) -> Path:
        return self.output_root / "sft_retrieval" / "bank" / "queries.jsonl"

    @property
    def backbones_root(self) -> Path:
        return self.output_root / "sft_retrieval" / "backbones"

    @property
    def bank_embeddings(self) -> Path:
        return self.backbones_root / self.backbone / "embeddings" / "bank"

    @property
    def query_embeddings(self) -> Path:
        return self.bank_embeddings.parent / "queries"

    @property
    def sft_root(self) -> Path:
        # The upstream TVL dataset selects its tactile branch from this directory name.
        return self.output_root / "training" / "ssvtp"

    @property
    def checkpoint_root(self) -> Path:
        return self.output_root / "training" / "checkpoints"

    @property
    def selected_checkpoint(self) -> Path:
        return self.checkpoint_root / "checkpoint-best.pth"


def _normalize_config(config: RGATacConfig, *, repo_root: Path) -> RGATacConfig:
    values = {
        field.name: value.resolve() if value.is_absolute() else (repo_root / value).resolve()
        for field in fields(config)
        if isinstance(value := getattr(config, field.name), Path)
    }
    assets = load_asset_paths(values.get("asset_config"), repo_root=repo_root, backbone=config.backbone)
    for name, asset_name in (
        ("dataset_root", "dataset_root"),
        ("tvl_root", "tvl_root"),
        ("llama_dir", "llama_dir"),
        ("encoder_checkpoint_path", "tvl_encoder_checkpoint"),
        ("tvl_llama_checkpoint", "tvl_llama_checkpoint"),
    ):
        values[name] = values.get(name, getattr(assets, asset_name))
    if config.source_train_manifest is None:
        values["source_train_manifest"] = values["output_root"] / "manifests" / "evidence_train.jsonl"
    if config.ssvtp_source_json is None:
        values["ssvtp_source_json"] = values["dataset_root"] / "ssvtp" / "finetune.json"
    normalized = replace(config, **values)
    for backbone in BACKBONES:
        if backbone != config.backbone and (normalized.backbones_root / backbone).is_dir():
            raise ValueError(
                f"{normalized.output_root} contains {backbone} training artifacts. "
                f"Use a separate directory for {config.backbone}."
            )
    return normalized


def render_rga_tac_prompt(terms: Sequence[str]) -> str:
    normalized = [str(term).strip().lower() for term in terms if str(term).strip()][:5]
    term_line = "Candidate tactile terms: " + (", ".join(normalized) if normalized else "none") + "."
    return "\n".join((
        DEFAULT_BASE_PROMPT,
        "Evidence cues:",
        term_line,
        "Use these train-memory cues only when they match the current visual-touch input.",
        "Do not copy evidence ids, scores, metadata, or numeric diagnostics.",
        "Answer with at most 5 tactile adjectives, comma-separated.",
    ))


def assign_rga_tac_conditions(source_ids: Sequence[str], *, split: str, seed: int, native_ratio: float) -> dict[str, str]:
    ranked = sorted(source_ids, key=lambda source_id: hashlib.sha256(
        f"rga-tac-assignment\0{seed}\0{split}\0{source_id}".encode("utf-8")
    ).digest())
    count = math.floor(len(source_ids) * native_ratio + 0.5)
    evidence_ids = set(ranked[:count])
    return {source_id: "native_evidence" if source_id in evidence_ids else "original_raw" for source_id in source_ids}


def _load_embedding_arrays(directory: Path) -> dict[str, np.ndarray]:
    return {view: np.load(directory / f"{view}.float16.npy", mmap_mode="r") for view in ("visual", "tactile", "fused")}


def prepare_training_data(config: RGATacConfig, options: dict) -> None:
    from rga_tac.embeddings.tvl_features import ExtractionConfig, extract_embedding_jobs

    source = json.loads(config.ssvtp_source_json.read_text())
    excluded = {row["source_id"] for row in json.loads(config.exclusion_config.read_text())["rows"]}
    eligible_ids = [f"ssvtp_train_{int(row['id']):06d}" for row in source]
    eligible_ids = [source_id for source_id in eligible_ids if source_id not in excluded]
    indices = list(range(len(eligible_ids)))
    random.Random(1).shuffle(indices)
    dev_indices = set(indices[:round(len(indices) * 0.1)])
    manifest = {row["evidence_id"]: row for row in read_jsonl(config.source_train_manifest)}
    train_rows = [manifest[source_id] for index, source_id in enumerate(eligible_ids) if index not in dev_indices]
    dev_rows = [manifest[source_id] for index, source_id in enumerate(eligible_ids) if index in dev_indices]
    queries = [dict(row, sft_split=split) for split, rows in (("train", train_rows), ("dev", dev_rows)) for row in rows]
    config.sft_bank_jsonl.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(config.sft_bank_jsonl, train_rows)
    write_jsonl(config.sft_query_jsonl, queries)
    extract_embedding_jobs(tuple(
        ExtractionConfig(
            manifest_path=manifest_path, output_dir=output_dir,
            tvl_root=config.tvl_root, checkpoint_path=config.encoder_checkpoint_path,
            tactile_model=config.tactile_model, device=config.device,
        )
        for manifest_path, output_dir in (
            (config.sft_bank_jsonl, config.bank_embeddings),
            (config.sft_query_jsonl, config.query_embeddings),
        )
    ))
    pools = build_candidate_pools(
        train_rows, queries,
        _load_embedding_arrays(config.bank_embeddings), _load_embedding_arrays(config.query_embeddings),
        CandidatePoolConfig(top_k=10, apply_pseudo_query_mask=True),
    )
    evidence = {pool["query_id"]: select_s_graph(pool) for pool in pools}
    original = {f"ssvtp_train_{int(row['id']):06d}": row["conversations"] for row in source}
    config.sft_root.mkdir(parents=True, exist_ok=True)
    for split, rows in (("train", train_rows), ("dev", dev_rows)):
        assignment = assign_rga_tac_conditions(
            [row["evidence_id"] for row in rows], split=split,
            seed=0, native_ratio=options["native_evidence_ratio"],
        )
        examples = []
        for row in rows:
            source_id = row["evidence_id"]
            condition = assignment[source_id]
            conversations = original[source_id]
            if condition == "native_evidence":
                conversations = [
                    {"from": "human", "value": "<image>\n" + render_rga_tac_prompt(evidence[source_id])},
                    {"from": "gpt", "value": row["caption_raw"]},
                ]
            examples.append({
                "id": f"{source_id}:{condition}", "image": row["image_path"],
                "tactile": row["tactile_path"], "conversations": conversations,
            })
        path = config.sft_root / f"{split}.json"
        path.write_text(json.dumps(examples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (config.sft_root / f"{split}.yaml").write_text(f"META:\n  - {json.dumps(path.as_posix())}\n")


def train(config: RGATacConfig) -> None:
    from rga_tac.tasks.finetune import run_finetune

    options = json.loads(config.training_config.read_text())
    prepare_training_data(config, options)
    args = SimpleNamespace(
        **{key: value for key, value in options.items() if key != "native_evidence_ratio"},
        data_config=config.sft_root / "train.yaml", val_data_config=config.sft_root / "dev.yaml",
        tvl_root=config.tvl_root, llama_path=config.llama_dir, llama_type="llama-2-7b",
        pretrained_path=config.tvl_llama_checkpoint, checkpoint_path=str(config.encoder_checkpoint_path),
        tactile_model=config.tactile_model, active_modality_names=["tactile", "vision"],
        lora_modality_names=["tactile", "vision"], device=config.device,
        output_dir=config.checkpoint_root,
    )
    run_finetune(args)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    add_backbone_argument(parser)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--source-train-manifest", type=Path)
    parser.add_argument("--ssvtp-source-json", type=Path)
    parser.add_argument("--asset-config", type=Path)
    parser.add_argument("--tvl-root", type=Path)
    parser.add_argument("--llama-dir", type=Path)
    parser.add_argument("--encoder-checkpoint-path", type=Path)
    parser.add_argument("--tvl-llama-checkpoint", type=Path)
    parser.add_argument("--training-config", type=Path, default=RGATacConfig.training_config)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    args.output_root = args.output_root or ssvtp_output_root(args.backbone)
    config = _normalize_config(RGATacConfig(**vars(args)), repo_root=default_repo_root())
    if args.source_train_manifest is None:
        prepare_manifests(config.dataset_root, config.source_train_manifest.parent)
    train(config)
    print(config.selected_checkpoint)


if __name__ == "__main__":
    main()
