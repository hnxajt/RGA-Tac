"""Fixed SSVTP caption normalization."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .normalize import clean_caption, normalize_terms, split_caption_terms


_INTERNAL_LINEBREAK_RE = re.compile(r"(?<=[A-Za-z])\s*[\r\n]+\s*(?=[A-Za-z])")
_LINEBREAK_RE = re.compile(r"[\r\n]+")
_DELIMITER_SPACING_RE = re.compile(r"\s*([,;/])\s*")
_TERM_CORRECTIONS: dict[str, tuple[str, ...]] = {
    "deformab le": ("deformable",),
    "deformalbe": ("deformable",),
    "facrib": ("fabric",),
    "fabricgranular": ("fabric", "granular"),
    "flat smooth": ("flat", "smooth"),
    "frabric": ("fabric",),
    "g rainy": ("grainy",),
    "gariny": ("grainy",),
    "graint": ("grainy",),
    "grainy fabric": ("grainy", "fabric"),
    "grainyfabric": ("grainy", "fabric"),
    "griany": ("grainy",),
    "line": ("lined",),
    "lined.fabric": ("lined", "fabric"),
    "lined.deformable": ("lined", "deformable"),
    "linedsewn": ("lined", "sewn"),
    "patterend": ("patterned",),
    "patterened": ("patterned",),
    "rigit": ("rigid",),
    "smmoth": ("smooth",),
    "smoothfabric": ("smooth", "fabric"),
    "smoth": ("smooth",),
    "smotth": ("smooth",),
    "spolotchy": ("splotchy",),
    "stripped": ("striped",),
    "vauge": ("vague",),
    "wirey": ("wiry",),
}


@dataclass(frozen=True)
class NormalizedCaption:
    caption_raw: str | None
    caption_terms_norm: list[str]


def normalize_ssvtp_caption(caption: str | None) -> NormalizedCaption:
    original = "" if caption is None else str(caption)
    text = _INTERNAL_LINEBREAK_RE.sub("", original)
    text = _LINEBREAK_RE.sub(" ", text)
    text = _DELIMITER_SPACING_RE.sub(r"\1 ", text).strip()
    terms = split_caption_terms(clean_caption(text))
    normalized_terms = [replacement for term in terms for replacement in _TERM_CORRECTIONS.get(_key(term), (term,))]
    caption_raw = clean_caption(", ".join(normalized_terms))
    return NormalizedCaption(
        caption_raw=caption_raw,
        caption_terms_norm=normalize_terms(normalized_terms),
    )


def _key(term: str) -> str:
    return re.sub(r"\s+", " ", term.strip().lower().strip(" .\t\n\r\"'"))
