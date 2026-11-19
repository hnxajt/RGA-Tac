"""Caption normalization."""

from __future__ import annotations

import re
from typing import Iterable


def clean_caption(caption: str | None) -> str | None:
    if caption is None:
        return None
    cleaned = str(caption).strip().strip("\"'")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.rstrip(".")


def split_caption_terms(caption: str | None) -> list[str]:
    cleaned = clean_caption(caption)
    if not cleaned:
        return []
    return [token for token in (part.strip(" .\t\n\r\"'") for part in re.split(r"[,;/]", cleaned)) if token]


def normalize_term(term: str) -> str:
    return re.sub(r"\s+", " ", term.strip().lower().strip(" .\t\n\r\"'"))


def normalize_terms(terms: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    normalized_terms: list[str] = []
    for term in terms:
        normalized = normalize_term(term)
        if normalized and normalized not in seen:
            normalized_terms.append(normalized)
            seen.add(normalized)
    return normalized_terms
