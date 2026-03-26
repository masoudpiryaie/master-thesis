from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .features import orientation_match, lifestyle_similarity, same_location, numeric_diff, age_similarity, text_sim_from_tfidf

@dataclass
class LabelConfig:
    age_threshold: int = 8
    text_threshold: float = 0.15
    negative_ratio: int = 2
    random_seed: int = 42

def build_pair_dataset(df: pd.DataFrame, tfidf_matrix, lifestyle_cols, cfg: LabelConfig, id_col="profile_id", age_col="age", gender_col="sex", orientation_col="orientation") -> pd.DataFrame:
    rng = np.random.default_rng(cfg.random_seed)
    rows = []
    for i in range(len(df)):
        row_i = df.iloc[i]
        feasible = []
        for j in range(len(df)):
            if i == j:
                continue
            row_j = df.iloc[j]
            if orientation_match(row_i, row_j, gender_col, orientation_col) == 0:
                continue
            text_sim = text_sim_from_tfidf(tfidf_matrix, i, j)
            age_diff = abs(float(row_i[age_col]) - float(row_j[age_col]))
            label = int(age_diff < cfg.age_threshold and text_sim > cfg.text_threshold)
            feasible.append((j, text_sim, age_diff, label))
        positives = [x for x in feasible if x[3] == 1]
        negatives = [x for x in feasible if x[3] == 0]
        if not positives:
            continue
        max_negs = min(len(negatives), len(positives) * cfg.negative_ratio)
        sampled_negatives = []
        if max_negs > 0:
            idxs = rng.choice(len(negatives), size=max_negs, replace=False)
            sampled_negatives = [negatives[k] for k in idxs]
        for j, text_sim, age_diff, label in positives + sampled_negatives:
            row_j = df.iloc[j]
            rows.append({
                "user_i": row_i[id_col],
                "user_j": row_j[id_col],
                "age_diff": float(age_diff),
                "age_sim": age_similarity(row_i[age_col], row_j[age_col]),
                "orientation_match": orientation_match(row_i, row_j, gender_col, orientation_col),
                "text_sim": float(text_sim),
                "lifestyle_sim": lifestyle_similarity(row_i, row_j, lifestyle_cols),
                "same_location": same_location(row_i, row_j),
                "height_diff": numeric_diff(row_i, row_j, "height"),
                "income_diff": numeric_diff(row_i, row_j, "income"),
                "label": int(label),
            })
    pairs = pd.DataFrame(rows)
    if pairs.empty:
        raise ValueError("No pair dataset built. Lower text_threshold or increase max_users.")
    return pairs
