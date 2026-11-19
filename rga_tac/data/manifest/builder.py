"""Prepare SSVTP training and test manifests."""

import argparse
from pathlib import Path

from rga_tac.integrations.tvl import add_backbone_argument
from rga_tac.paths import ssvtp_output_root

from .io import write_jsonl
from .sources import build_ssvtp_records


def prepare_manifests(dataset_root: Path, output_dir: Path) -> None:
    for split, name in (("train", "evidence_train"), ("test", "eval_queries")):
        write_jsonl(output_dir / f"{name}.jsonl", build_ssvtp_records(dataset_root, split))


def prepare_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    add_backbone_argument(parser)
    args = parser.parse_args(argv)
    output_dir = args.output_dir or ssvtp_output_root(args.backbone) / "manifests"
    prepare_manifests(args.dataset_root, output_dir)


if __name__ == "__main__":
    prepare_main()
