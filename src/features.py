from __future__ import annotations
import math
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def orientation_match(row_i: pd.Series, row_j: pd.Series, gender_col="sex", orientation_col="orientation") -> int:
    sex_i = str(row_i[gender_col]).lower()
    sex_j = str(row_j[gender_col]).lower()
    ori_i = str(row_i[orientation_col]).lower()
    if ori_i == "straight":
        return int(sex_i != sex_j)
    if ori_i == "gay":
        return int(sex_i == sex_j)
    if ori_i == "bisexual":
        return 1
    return 0

def same_location(row_i: pd.Series, row_j: pd.Series, location_col="location") -> float:
    li = str(row_i.get(location_col, "")).strip().lower()
    lj = str(row_j.get(location_col, "")).strip().lower()
    if not li or not lj or li == "nan" or lj == "nan":
        return 0.0
    return float(li == lj)

def numeric_diff(row_i: pd.Series, row_j: pd.Series, col: str) -> float:
    vi = row_i.get(col, np.nan)
    vj = row_j.get(col, np.nan)
    if pd.isna(vi) or pd.isna(vj):
        return 0.0
    return float(abs(float(vi) - float(vj)))

def lifestyle_similarity(row_i: pd.Series, row_j: pd.Series, cols) -> float:
    matches, total = 0.0, 0.0
    for col in cols:
        if col not in row_i.index or col not in row_j.index:
            continue
        vi = str(row_i[col]).strip().lower()
        vj = str(row_j[col]).strip().lower()
        if vi == "unknown" or vj == "unknown":
            continue
        total += 1.0
        matches += float(vi == vj)
    return matches / total if total else 0.0

def age_similarity(age_i: float, age_j: float, decay: float = 0.1) -> float:
    return math.exp(-decay * abs(float(age_i) - float(age_j)))

def text_sim_from_tfidf(tfidf_matrix, i: int, j: int) -> float:
    return float(cosine_similarity(tfidf_matrix[i], tfidf_matrix[j]).item())
