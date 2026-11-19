"""Select tactile terms by support from distinct retrieved cases."""

from dataclasses import dataclass


@dataclass
class SGraphConfig:
    support_floor: int = 2
    top_m: int = 5


def select_s_graph(pool_record: dict, config: SGraphConfig = SGraphConfig()) -> list[str]:
    support = {}
    best_rank = {}
    for case in pool_record["candidate_pool"]:
        rank = min(case[field] for field in ("r_v", "r_t", "r_vt") if case[field] is not None)
        for term in set(case["Y_e"]):
            support.setdefault(term, set()).add(case["evidence_id"])
            best_rank[term] = min(best_rank.get(term, rank), rank)
    terms = sorted(support, key=lambda term: (-len(support[term]), best_rank[term], term))
    return [term for term in terms if len(support[term]) >= config.support_floor][:config.top_m]
