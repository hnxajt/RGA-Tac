"""Read the original SSVTP visual-tactile pairs and captions."""

from pathlib import Path
import re

from .caption import normalize_ssvtp_caption
from .io import read_csv_rows


def build_ssvtp_records(dataset_root: Path, split: str) -> list[dict]:
    ssvtp_root = dataset_root.resolve() / "ssvtp"
    records = []
    for row in read_csv_rows(ssvtp_root / f"{split}.csv"):
        image = (row.get("url") or row.get("image")).strip()
        tactile = row.get("tactile", "").strip() or image.replace("images_rgb", "images_tac").replace("_rgb.", "_tac.")
        sample_id = re.search(r"image_(\d+)_rgb\.", Path(image).name, re.I).group(1)
        caption = normalize_ssvtp_caption(row["caption"])
        records.append({
            "evidence_id": f"ssvtp_{split}_{int(sample_id):06d}",
            "sample_id": sample_id,
            "image_path": str(ssvtp_root / image), "tactile_path": str(ssvtp_root / tactile),
            "image_rel_path": f"ssvtp/{image}", "tactile_rel_path": f"ssvtp/{tactile}",
            "caption_raw": caption.caption_raw, "caption_terms_norm": caption.caption_terms_norm,
        })
    return records
