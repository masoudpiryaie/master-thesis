from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from src.preprocessing import PreprocessConfig, load_and_clean_profiles, build_tfidf_matrix
from src.labels import LabelConfig, build_pair_dataset
from src.baselines import TrainConfig, train_test_pairs, fit_logistic_regression, evaluate_auc
from src.twostage_model import TwoStageConfig, stage1_candidates, stage2_rank

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--max_users", type=int, default=2000)
    parser.add_argument("--candidate_top_n", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1] Loading and cleaning profiles...")
    pcfg = PreprocessConfig(csv_path=args.csv_path, max_users=args.max_users)
    df = load_and_clean_profiles(pcfg)
    print("Users after cleaning:", df.shape)

    print("[2] Building TF-IDF matrix...")
    tfidf_vec, tfidf_matrix = build_tfidf_matrix(df, pcfg)

    print("[3] Building pair dataset...")
    lcfg = LabelConfig()
    pairs = build_pair_dataset(df, tfidf_matrix, pcfg.lifestyle_cols, lcfg, id_col=pcfg.id_col)
    print("Pairs shape:", pairs.shape)
    print("Label distribution:")
    print(pairs["label"].value_counts())

    pairs.to_csv(out_dir / "pairs_dataset.csv", index=False)

    print("[4] Training logistic regression baseline...")
    tcfg = TrainConfig()
    X_train, X_test, y_train, y_test, idx_train, idx_test, scaler = train_test_pairs(pairs, tcfg)
    log_reg = fit_logistic_regression(X_train, y_train)
    metrics = evaluate_auc(log_reg, X_test, y_test)
    print("Baseline metrics:", metrics)
    pd.DataFrame([metrics]).to_csv(out_dir / "metrics_summary.csv", index=False)

    print("[5] Generating example recommendations...")
    user_index = {uid: idx for idx, uid in enumerate(df[pcfg.id_col].tolist())}
    sample_user = df.iloc[0][pcfg.id_col]
    candidates = stage1_candidates(sample_user, df, user_index, tfidf_matrix, candidate_top_n=args.candidate_top_n, id_col=pcfg.id_col)
    ranked = stage2_rank(sample_user, candidates, df, user_index, tfidf_matrix, scaler, log_reg, TwoStageConfig(candidate_top_n=args.candidate_top_n), pcfg.lifestyle_cols, age_col=pcfg.age_col)
    ranked.to_csv(out_dir / "recommendation_examples.csv", index=False)
    print(ranked.head(10))

    print("[DONE] Saved files in:", out_dir.resolve())

if __name__ == "__main__":
    main()
