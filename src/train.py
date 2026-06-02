#!/usr/bin/env python3
"""
Train and evaluate two collaborative filtering models on MovieLens 100K.

Models:
  1. SVD (scipy) — truncated SVD on the user-item matrix
  2. User-based CF — cosine similarity + weighted k-NN (k=20)

Split: 80/20 random. Metrics: RMSE, MAE on held-out ratings.
Save:  models/results.json  +  models/user_sim.npy  +  models/svd_components.npz
"""
from __future__ import annotations
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity

from src.features import load_ratings, build_matrix, split

MODEL_DIR   = pathlib.Path(__file__).parent.parent / "models"
RESULTS_PATH = MODEL_DIR / "results.json"

N_FACTORS = 50
K_NEIGHBORS = 20


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


# ── SVD ──────────────────────────────────────────────────────────────────────

def train_svd(train_mat: np.ndarray, n_factors: int = N_FACTORS):
    mean_user = np.where(train_mat > 0, train_mat, np.nan)
    row_means  = np.nanmean(mean_user, axis=1, keepdims=True)
    row_means  = np.nan_to_num(row_means, nan=0.0)
    demeaned   = train_mat - row_means * (train_mat > 0)

    k = min(n_factors, min(demeaned.shape) - 1)
    U, sigma, Vt = svds(demeaned.astype(np.float64), k=k)
    return U, sigma, Vt, row_means


def predict_svd(U, sigma, Vt, row_means, user_idx, item_idx):
    pred_mat = U @ np.diag(sigma) @ Vt + row_means
    return float(np.clip(pred_mat[user_idx, item_idx], 1, 5))


# ── User-based CF ─────────────────────────────────────────────────────────────

def train_ubcf(train_mat: np.ndarray):
    sim = cosine_similarity(train_mat)
    np.fill_diagonal(sim, 0)
    return sim


def predict_ubcf(sim: np.ndarray, train_mat: np.ndarray, user_idx: int,
                 item_idx: int, k: int = K_NEIGHBORS) -> float:
    item_col = train_mat[:, item_idx]
    raters   = np.where(item_col > 0)[0]
    if len(raters) == 0:
        return float(train_mat[user_idx][train_mat[user_idx] > 0].mean() or 3.0)
    sims = sim[user_idx, raters]
    top_k = np.argsort(sims)[-k:]
    weights = sims[top_k]
    ratings = item_col[raters[top_k]]
    denom = weights.sum()
    if denom == 0:
        return 3.0
    return float(np.clip((weights @ ratings) / denom, 1, 5))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    MODEL_DIR.mkdir(exist_ok=True)

    print("Loading MovieLens 100K...")
    df = load_ratings()
    train_df, test_df = split(df)
    print(f"  train={len(train_df):,}  test={len(test_df):,}")

    train_mat = build_matrix(train_df)

    print(f"\nTraining SVD (k={N_FACTORS} factors)...")
    U, sigma, Vt, row_means = train_svd(train_mat)
    np.savez(MODEL_DIR / "svd_components.npz", U=U, sigma=sigma, Vt=Vt,
             row_means=row_means)

    print(f"Training User-based CF (k={K_NEIGHBORS} neighbours)...")
    sim = train_ubcf(train_mat)
    np.save(MODEL_DIR / "user_sim.npy", sim)

    print("\nEvaluating on test set...")
    svd_preds, ubcf_preds, actuals = [], [], []
    for row in test_df.itertuples(index=False):
        u, i, r = row.user_id - 1, row.item_id - 1, row.rating
        if u >= train_mat.shape[0] or i >= train_mat.shape[1]:
            continue
        svd_preds.append(predict_svd(U, sigma, Vt, row_means, u, i))
        ubcf_preds.append(predict_ubcf(sim, train_mat, u, i))
        actuals.append(r)

    actuals   = np.array(actuals)
    svd_preds = np.array(svd_preds)
    ubcf_preds = np.array(ubcf_preds)

    results = {
        "svd":  {"rmse": round(rmse(actuals, svd_preds), 4),
                 "mae":  round(mae(actuals,  svd_preds), 4),
                 "n_factors": N_FACTORS},
        "ubcf": {"rmse": round(rmse(actuals, ubcf_preds), 4),
                 "mae":  round(mae(actuals,  ubcf_preds), 4),
                 "k_neighbours": K_NEIGHBORS},
        "n_users": int(df["user_id"].max()),
        "n_items": int(df["item_id"].max()),
        "train_ratings": len(train_df),
        "test_ratings":  len(test_df),
    }

    print(f"\nSVD   — RMSE: {results['svd']['rmse']}  MAE: {results['svd']['mae']}")
    print(f"UBCF  — RMSE: {results['ubcf']['rmse']}  MAE: {results['ubcf']['mae']}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
