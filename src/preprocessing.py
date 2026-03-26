from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

@dataclass
class PreprocessConfig:
    csv_path: str
    id_col: str = "profile_id"
    age_col: str = "age"
    gender_col: str = "sex"
    orientation_col: str = "orientation"
    location_col: str = "location"
    essay_cols: Tuple[str, ...] = ("essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9")
    lifestyle_cols: Tuple[str, ...] = ("body_type","diet","drinks","drugs","education","ethnicity","job","offspring","pets","religion","sign","smokes","speaks","status")
    numeric_optional_cols: Tuple[str, ...] = ("height","income")
    min_age: int = 18
    max_age: int = 70
    min_text_chars: int = 50
    max_users: Optional[int] = 2000
    tfidf_max_features: int = 5000
    random_seed: int = 42

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_and_clean_profiles(cfg: PreprocessConfig) -> pd.DataFrame:
    csv_path = Path(cfg.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if cfg.id_col not in df.columns:
        df[cfg.id_col] = np.arange(len(df))
    required = [cfg.age_col, cfg.gender_col, cfg.orientation_col]
    df = df.dropna(subset=required).copy()
    df[cfg.age_col] = pd.to_numeric(df[cfg.age_col], errors="coerce")
    df = df.dropna(subset=[cfg.age_col])
    df = df[(df[cfg.age_col] >= cfg.min_age) & (df[cfg.age_col] <= cfg.max_age)]

    essay_cols = [c for c in cfg.essay_cols if c in df.columns]
    if not essay_cols:
        raise ValueError("No essay columns found.")
    df["profile_text"] = df[essay_cols].fillna("").astype(str).agg(" ".join, axis=1).map(clean_text)
    df = df[df["profile_text"].str.len() >= cfg.min_text_chars]

    for col in cfg.lifestyle_cols:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str).str.lower().str.strip()
    for col in cfg.numeric_optional_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df[cfg.gender_col] = df[cfg.gender_col].astype(str).str.lower().str.strip()
    df[cfg.orientation_col] = df[cfg.orientation_col].astype(str).str.lower().str.strip()
    if cfg.location_col in df.columns:
        df[cfg.location_col] = df[cfg.location_col].fillna("").astype(str).str.lower().str.strip()

    if cfg.max_users is not None and len(df) > cfg.max_users:
        df = df.sample(cfg.max_users, random_state=cfg.random_seed).reset_index(drop=True)

    return df.reset_index(drop=True)

def build_tfidf_matrix(df: pd.DataFrame, cfg: PreprocessConfig):
    vec = TfidfVectorizer(max_features=cfg.tfidf_max_features, stop_words="english", ngram_range=(1,2))
    matrix = vec.fit_transform(df["profile_text"])
    return vec, matrix
