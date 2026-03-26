from __future__ import annotations
from typing import Dict, List
import math
import numpy as np

def recall_at_k(y_true: List[int], y_score: List[float], k: int) -> float:
    if not y_true:
        return 0.0
    order = np.argsort(y_score)[::-1][:k]
    hits = sum(y_true[idx] for idx in order)
    total_relevant = sum(y_true)
    return float(hits / total_relevant) if total_relevant > 0 else 0.0

def ndcg_at_k(y_true: List[int], y_score: List[float], k: int) -> float:
    order = np.argsort(y_score)[::-1][:k]
    dcg = 0.0
    for rank, idx in enumerate(order, start=1):
        rel = y_true[idx]
        dcg += (2 ** rel - 1) / math.log2(rank + 1)
    best = sorted(y_true, reverse=True)
    ideal = 0.0
    for rank, rel in enumerate(best[:k], start=1):
        ideal += (2 ** rel - 1) / math.log2(rank + 1)
    return float(dcg / ideal) if ideal > 0 else 0.0

def reciprocal_rank(y_true: List[int], y_score: List[float]) -> float:
    order = np.argsort(y_score)[::-1]
    for rank, idx in enumerate(order, start=1):
        if y_true[idx] == 1:
            return 1.0 / rank
    return 0.0

def mutual_top_k_rate(reco_dict: Dict[int, List[int]], k: int) -> float:
    pairs = 0
    mutual = 0
    for u, recs in reco_dict.items():
        for v in recs[:k]:
            pairs += 1
            if v in reco_dict and u in reco_dict[v][:k]:
                mutual += 1
    return float(mutual / pairs) if pairs > 0 else 0.0
