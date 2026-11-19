"""Cosine retrieval across visual, tactile, and fused views."""

from dataclasses import dataclass

import numpy as np


VIEWS = ("visual", "tactile", "fused")


@dataclass
class CandidatePoolConfig:
    top_k: int = 10
    apply_pseudo_query_mask: bool = False


def build_candidate_pools(bank_records, query_records, bank_embeddings, query_embeddings, config: CandidatePoolConfig) -> list[dict]:
    bank_ids = np.asarray([row["evidence_id"] for row in bank_records])
    bank_media = [(row["image_rel_path"], row["tactile_rel_path"]) for row in bank_records]
    scores = {view: _cosine_scores(query_embeddings[view], bank_embeddings[view]) for view in VIEWS}
    pools = []
    for index, query in enumerate(query_records):
        excluded = []
        if config.apply_pseudo_query_mask:
            media = (query["image_rel_path"], query["tactile_rel_path"])
            excluded = [i for i, pair in enumerate(bank_media) if pair == media]
        ranks = {}
        for view in VIEWS:
            values = scores[view][index].copy()
            values[excluded] = -np.inf
            top_rows = np.lexsort((bank_ids, -values))[:config.top_k]
            ranks[view] = {int(row): rank for rank, row in enumerate(top_rows, 1)}
        union = sorted(set().union(*(set(view) for view in ranks.values())), key=lambda row: bank_ids[row])
        pools.append({
            "query_id": query["evidence_id"],
            "candidate_pool": [
                {
                    "evidence_id": bank_records[row]["evidence_id"],
                    "Y_e": bank_records[row]["caption_terms_norm"],
                    "r_v": ranks["visual"].get(row),
                    "r_t": ranks["tactile"].get(row),
                    "r_vt": ranks["fused"].get(row),
                }
                for row in union
            ],
        })
    return pools


def _cosine_scores(queries: np.ndarray, bank: np.ndarray) -> np.ndarray:
    queries, bank = np.asarray(queries, dtype=np.float32), np.asarray(bank, dtype=np.float32)
    numerator = queries @ bank.T
    denominator = np.linalg.norm(queries, axis=1)[:, None] * np.linalg.norm(bank, axis=1)[None, :]
    return np.divide(numerator, denominator, out=np.full_like(numerator, -np.inf), where=denominator > 0)
