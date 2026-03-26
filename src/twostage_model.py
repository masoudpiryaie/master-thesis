from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from .features import orientation_match, lifestyle_similarity, same_location, numeric_diff, age_similarity

@dataclass
class TwoStageConfig:
    candidate_top_n: int = 50
    alpha: float = 0.25
    beta: float = 0.45
    gamma: float = 0.15
    delta: float = 0.15

def reciprocal_score(p_ij: float, p_ji: float) -> float:
    return min(float(p_ij), float(p_ji))

def stage1_candidates(user_id, df: pd.DataFrame, user_index: Dict[int, int], tfidf_matrix, candidate_top_n: int = 50, gender_col="sex", orientation_col="orientation", id_col="profile_id") -> List[int]:
    i = user_index[user_id]
    row_i = df.iloc[i]
    sims = cosine_similarity(tfidf_matrix[i], tfidf_matrix).flatten()
    candidates = []
    for j, sim in enumerate(sims):
        if i == j:
            continue
        row_j = df.iloc[j]
        if orientation_match(row_i, row_j, gender_col, orientation_col) == 0:
            continue
        candidates.append((row_j[id_col], float(sim)))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [uid for uid, _ in candidates[:candidate_top_n]]

def stage2_rank(user_id, candidates: List[int], df: pd.DataFrame, user_index: Dict[int, int], tfidf_matrix, scaler, log_reg_model, cfg: TwoStageConfig, lifestyle_cols, age_col="age", gender_col="sex", orientation_col="orientation"):
    i = user_index[user_id]
    row_i = df.iloc[i]
    rows = []
    for uid_j in candidates:
        j = user_index[uid_j]
        row_j = df.iloc[j]
        text_sim = float(cosine_similarity(tfidf_matrix[i], tfidf_matrix[j]).item())
        age_sim = age_similarity(row_i[age_col], row_j[age_col])
        orient_ij = orientation_match(row_i, row_j, gender_col, orientation_col)
        orient_ji = orientation_match(row_j, row_i, gender_col, orientation_col)
        lifestyle = lifestyle_similarity(row_i, row_j, lifestyle_cols)
        loc = same_location(row_i, row_j)
        height_diff = numeric_diff(row_i, row_j, "height")
        income_diff = numeric_diff(row_i, row_j, "income")
        x_ij = np.array([[age_sim, orient_ij, text_sim, lifestyle, loc, height_diff, income_diff]])
        x_ji = np.array([[age_sim, orient_ji, text_sim, lifestyle, loc, height_diff, income_diff]])
        x_ij = scaler.transform(x_ij)
        x_ji = scaler.transform(x_ji)
        p_ij = log_reg_model.predict_proba(x_ij)[0, 1]
        p_ji = log_reg_model.predict_proba(x_ji)[0, 1]
        hybrid_ij = cfg.alpha * age_sim + cfg.beta * text_sim + cfg.gamma * lifestyle + cfg.delta * loc
        hybrid_ji = cfg.alpha * age_sim + cfg.beta * text_sim + cfg.gamma * lifestyle + cfg.delta * loc
        final_ij = 0.5 * p_ij + 0.5 * hybrid_ij
        final_ji = 0.5 * p_ji + 0.5 * hybrid_ji
        rec = reciprocal_score(final_ij, final_ji)
        rows.append({"user_i": user_id, "user_j": uid_j, "p_ij": float(p_ij), "p_ji": float(p_ji), "hybrid_ij": float(hybrid_ij), "hybrid_ji": float(hybrid_ji), "reciprocal_score": float(rec)})
    return pd.DataFrame(rows).sort_values("reciprocal_score", ascending=False).reset_index(drop=True)
