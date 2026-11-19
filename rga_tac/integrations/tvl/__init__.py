from argparse import ArgumentParser
from dataclasses import dataclass

from .paths import add_tvl_encoder_path, add_tvl_generation_paths


@dataclass(frozen=True)
class TVLBackbone:
    tactile_model: str
    checkpoint_suffix: str


BACKBONES = {
    "tiny": TVLBackbone("vit_tiny_patch16_224", "vittiny"),
    "small": TVLBackbone("vit_small_patch16_224", "vits"),
    "base": TVLBackbone("vit_base_patch16_224", "vitb"),
}
DEFAULT_BACKBONE = "small"
TACTILE_MODEL = BACKBONES[DEFAULT_BACKBONE].tactile_model


def add_backbone_argument(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--backbone", choices=BACKBONES, default=DEFAULT_BACKBONE,
        help="TVL tactile encoder size (default: %(default)s).",
    )
