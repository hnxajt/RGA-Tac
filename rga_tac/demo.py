"""SSVTP retrieval-guided generation from the original visual-tactile pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rga_tac.benchmark.t.rga_tac import (
    RGATacConfig,
    _load_embedding_arrays,
    _normalize_config,
    render_rga_tac_prompt,
)
from rga_tac.data.manifest.io import read_jsonl, write_jsonl
from rga_tac.data.manifest.sources import build_ssvtp_records
from rga_tac.integrations.tvl import add_backbone_argument
from rga_tac.paths import default_repo_root, ssvtp_output_root
from rga_tac.retrieval.candidate_pool import CandidatePoolConfig, build_candidate_pools
from rga_tac.retrieval.s_graph import SGraphConfig, select_s_graph


def run_demo(
    config: RGATacConfig,
    *,
    sample_id: int | None = None,
    model_path: Path | None = None,
    output_json: Path | None = None,
) -> dict[str, Any]:
    from rga_tac.embeddings.tvl_features import ExtractionConfig, extract_embedding_jobs
    from rga_tac.tasks.tvl_generation.config import TVLGenerationConfig
    from rga_tac.tasks.tvl_generation.head import TVLGenerationHead

    model_path = model_path if model_path is not None else config.selected_checkpoint
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing {config.backbone} RGA-Tac checkpoint: {model_path}")
    bank_records = read_jsonl(config.sft_bank_jsonl)
    bank_embeddings = _load_embedding_arrays(config.bank_embeddings)
    samples = build_ssvtp_records(config.dataset_root, "test")
    sample = samples[0] if sample_id is None else next(
        sample for sample in samples if sample["evidence_id"] == f"ssvtp_test_{sample_id:06d}"
    )
    query = {
        key: sample[key]
        for key in (
            "evidence_id", "sample_id",
            "image_path", "tactile_path", "image_rel_path", "tactile_rel_path",
        )
    }
    query_root = config.output_root / "inference" / sample["evidence_id"]
    query_root.mkdir(parents=True, exist_ok=True)
    query_manifest = query_root / "query.jsonl"
    write_jsonl(query_manifest, [query])
    query_embeddings = query_root / "embeddings"
    extract_embedding_jobs(
        (
            ExtractionConfig(
                manifest_path=query_manifest,
                output_dir=query_embeddings,
                tvl_root=config.tvl_root,
                checkpoint_path=config.encoder_checkpoint_path,
                tactile_model=config.tactile_model,
                device=config.device,
            ),
        )
    )
    pools = build_candidate_pools(
        bank_records,
        [query],
        bank_embeddings,
        _load_embedding_arrays(query_embeddings),
        CandidatePoolConfig(top_k=10),
    )
    terms = select_s_graph(pools[0], config=SGraphConfig(support_floor=2, top_m=5))
    prompt = render_rga_tac_prompt(terms)
    head = TVLGenerationHead(
        TVLGenerationConfig(
            model_path=model_path,
            llama_dir=config.llama_dir,
            encoder_checkpoint_path=config.encoder_checkpoint_path,
            tvl_root=config.tvl_root,
            tactile_model=config.tactile_model,
            max_gen_len=256,
            generation_seed=0,
            device=config.device,
        )
    )
    result = {"sample_id": sample["sample_id"], "generated_text": head.generate(query, prompt=prompt)}
    destination = output_json if output_json is not None else config.output_root / "generated.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def demo_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate a tactile description for an SSVTP test sample.")
    parser.add_argument("--training-root", type=Path)
    add_backbone_argument(parser)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--asset-config", type=Path)
    parser.add_argument("--tvl-root", type=Path)
    parser.add_argument("--llama-dir", type=Path)
    parser.add_argument("--encoder-checkpoint", "--encoder-checkpoint-path", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--sample-id", type=int, help="Upstream SSVTP test sample number; defaults to the first test sample.")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    config = _normalize_config(
        RGATacConfig(
            output_root=args.training_root or ssvtp_output_root(args.backbone),
            backbone=args.backbone,
            dataset_root=args.dataset_root,
            asset_config=args.asset_config,
            tvl_root=args.tvl_root,
            llama_dir=args.llama_dir,
            encoder_checkpoint_path=args.encoder_checkpoint,
            device=args.device,
        ),
        repo_root=default_repo_root(),
    )
    result = run_demo(config, sample_id=args.sample_id, model_path=args.model_path, output_json=args.output_json)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    demo_main()
