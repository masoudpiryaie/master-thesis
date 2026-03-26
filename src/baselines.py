from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = ["age_sim","orientation_match","text_sim","lifestyle_sim","same_location","height_diff","income_diff"]

@dataclass
class TrainConfig:
    test_size: float = 0.2
    random_seed: int = 42

def train_test_pairs(pairs: pd.DataFrame, cfg: TrainConfig):
    X = pairs[FEATURE_COLS].fillna(0.0).copy()
    y = pairs["label"].astype(int).copy()
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, pairs.index, test_size=cfg.test_size, stratify=y, random_state=cfg.random_seed
    )
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)
    return X_train_scaled, X_test_scaled, y_train, y_test, idx_train, idx_test, scaler

def fit_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_auc(model: LogisticRegression, X_test: pd.DataFrame, y_test: pd.Series):
    probs = model.predict_proba(X_test)[:, 1]
    return {"auc": float(roc_auc_score(y_test, probs))}
