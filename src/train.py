from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

from src.features import build_user_item_matrix, load_movies, load_ratings, split_ratings

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
RESULTS_PATH = MODELS_DIR / "results.json"
N_FACTORS = 50
K_NEIGHBORS = 30
RANDOM_STATE = 42


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def fit_biases(matrix: np.ndarray, reg: float = 10.0, n_iter: int = 12) -> tuple[float, np.ndarray, np.ndarray]:
    observed = matrix > 0
    global_mean = float(matrix[observed].mean())
    user_bias = np.zeros(matrix.shape[0], dtype=np.float32)
    item_bias = np.zeros(matrix.shape[1], dtype=np.float32)
    user_counts = observed.sum(axis=1).astype(np.float32)
    item_counts = observed.sum(axis=0).astype(np.float32)

    for _ in range(n_iter):
        residual = np.where(observed, matrix - global_mean - item_bias[None, :], 0.0)
        user_bias = residual.sum(axis=1) / (reg + user_counts)
        residual = np.where(observed, matrix - global_mean - user_bias[:, None], 0.0)
        item_bias = residual.sum(axis=0) / (reg + item_counts)

    return global_mean, user_bias, item_bias


def train_svd(matrix: np.ndarray, n_factors: int = N_FACTORS) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
    observed = matrix > 0
    global_mean, user_bias, item_bias = fit_biases(matrix)
    baseline = global_mean + user_bias[:, None] + item_bias[None, :]
    residual = np.where(observed, matrix - baseline, 0.0)
    k = min(n_factors, min(matrix.shape) - 1)
    u, sigma, vt = svds(csr_matrix(residual), k=k)
    order = np.argsort(sigma)[::-1]
    u = u[:, order]
    sigma = sigma[order]
    vt = vt[order, :]
    reconstructed = (u @ np.diag(sigma) @ vt) + baseline
    predictions = np.clip(reconstructed, 1.0, 5.0).astype(np.float32)
    params = {
        "global_mean": global_mean,
        "user_bias": user_bias,
        "item_bias": item_bias,
        "u": u.astype(np.float32),
        "sigma": sigma.astype(np.float32),
        "vt": vt.astype(np.float32),
    }
    return predictions, params


def train_user_cf(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    observed = matrix > 0
    counts = observed.sum(axis=1)
    sums = matrix.sum(axis=1)
    user_means = np.divide(sums, counts, out=np.full_like(sums, 3.0, dtype=np.float32), where=counts > 0)
    centered = np.where(observed, matrix - user_means[:, None], 0.0)
    similarity = cosine_similarity(centered)
    np.fill_diagonal(similarity, 0.0)
    return similarity.astype(np.float32), user_means.astype(np.float32)


def predict_user_cf(
    similarity: np.ndarray,
    matrix: np.ndarray,
    user_means: np.ndarray,
    user_idx: int,
    item_idx: int,
    k: int = K_NEIGHBORS,
) -> float:
    item_ratings = matrix[:, item_idx]
    raters = np.where(item_ratings > 0)[0]
    if len(raters) == 0:
        return float(user_means[user_idx])
    weights = similarity[user_idx, raters]
    if len(weights) > k:
        top = np.argpartition(np.abs(weights), -k)[-k:]
        raters = raters[top]
        weights = weights[top]
    denom = np.abs(weights).sum()
    if denom <= 1e-9:
        return float(user_means[user_idx])
    residuals = item_ratings[raters] - user_means[raters]
    return float(np.clip(user_means[user_idx] + (weights @ residuals) / denom, 1.0, 5.0))


def evaluate_user_cf(test_df: pd.DataFrame, matrix: np.ndarray, similarity: np.ndarray, user_means: np.ndarray) -> np.ndarray:
    preds = []
    for row in test_df.itertuples(index=False):
        preds.append(
            predict_user_cf(
                similarity,
                matrix,
                user_means,
                int(row.user_id) - 1,
                int(row.item_id) - 1,
            )
        )
    return np.asarray(preds, dtype=np.float32)


def write_readme(results: dict) -> None:
    svd = results["models"]["svd"]
    ubcf = results["models"]["user_based_cf"]
    readme = f"""# MovieLens Recommender System

Portfolio-quality collaborative filtering project using the MovieLens 100K ratings dataset.

## Objective

Compare two recommendation approaches on 100,000 movie ratings from 943 users and 1,682 movies:

- SVD matrix factorization using `scipy.sparse.linalg.svds`
- User-based collaborative filtering with cosine similarity

The target benchmark is about 0.92 RMSE on a held-out 20% test split.

## Results

| Model | RMSE | MAE | Notes |
|---|---:|---:|---|
| SVD matrix factorization | {svd["rmse"]:.4f} | {svd["mae"]:.4f} | {svd["n_factors"]} latent factors plus user/item bias baseline |
| User-based collaborative filtering | {ubcf["rmse"]:.4f} | {ubcf["mae"]:.4f} | cosine similarity, top {ubcf["k_neighbors"]} neighbors |

Baseline reference: SVD is expected to land near 0.92 RMSE on this split. Actual metrics are saved in `models/results.json`.

## Architecture

```text
MovieLens 100K zip
    |
    v
src/features.py
    |-- download/load ratings
    |-- build user-item matrix
    |-- create 80/20 train-test split
    v
src/train.py
    |-- SVD factorization
    |-- user-based CF
    |-- evaluation metrics
    |-- model artifacts in models/
    |
    +--> api/serve.py  FastAPI /recommend endpoint
    |
    +--> app.py        Streamlit recommendation demo
```

## Project Structure

```text
recommender-system/
├── api/serve.py
├── app.py
├── models/results.json
├── requirements.txt
├── src/features.py
└── src/train.py
```

## How to Run

```bash
cd /home/daniel/Code/recommender-system
python3 -m pip install -r requirements.txt
python3 -m src.train
uvicorn api.serve:app --reload
streamlit run app.py
```

The training command downloads MovieLens 100K automatically if `data/raw/ml-100k/u.data` is not present.

## API Example

```bash
curl -X POST http://127.0.0.1:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{{"user_id": 42, "top_n": 10}}'
```

Response:

```json
{{
  "user_id": 42,
  "recommendations": [
    {{"item_id": 318, "title": "Schindler's List (1993)", "predicted_rating": 4.73}}
  ]
}}
```

## Notes

- Ratings are one-indexed in the raw data; model arrays are zero-indexed internally.
- Recommendations filter out movies the user already rated in the training split.
- If a user has rated every candidate in the training matrix, the API falls back to ranking all movies by predicted score.
"""
    (ROOT / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ratings = load_ratings()
    movies = load_movies()
    n_users = int(ratings["user_id"].max())
    n_items = int(ratings["item_id"].max())
    train_df, test_df = split_ratings(ratings, test_size=0.2, random_state=RANDOM_STATE)
    train_matrix = build_user_item_matrix(train_df, n_users=n_users, n_items=n_items)
    actual = test_df["rating"].to_numpy(dtype=np.float32)
    test_users = test_df["user_id"].to_numpy(dtype=int) - 1
    test_items = test_df["item_id"].to_numpy(dtype=int) - 1

    print(f"MovieLens 100K: train={len(train_df):,} test={len(test_df):,} users={n_users:,} items={n_items:,}")
    print(f"Training SVD with {N_FACTORS} factors...")
    svd_predictions, svd_params = train_svd(train_matrix)
    svd_test = svd_predictions[test_users, test_items]

    print(f"Training user-based collaborative filtering with {K_NEIGHBORS} neighbors...")
    similarity, user_means = train_user_cf(train_matrix)
    ubcf_test = evaluate_user_cf(test_df, train_matrix, similarity, user_means)

    seen_mask = train_matrix > 0
    movies = movies.sort_values("item_id")
    movie_lookup_path = MODELS_DIR / "movie_lookup.csv"
    movies.to_csv(movie_lookup_path, index=False)

    np.save(MODELS_DIR / "prediction_matrix.npy", svd_predictions)
    np.save(MODELS_DIR / "seen_mask.npy", seen_mask)
    np.save(MODELS_DIR / "user_similarity.npy", similarity)
    np.savez_compressed(
        MODELS_DIR / "svd_components.npz",
        global_mean=np.asarray([svd_params["global_mean"]], dtype=np.float32),
        user_bias=svd_params["user_bias"],
        item_bias=svd_params["item_bias"],
        u=svd_params["u"],
        sigma=svd_params["sigma"],
        vt=svd_params["vt"],
    )

    results = {
        "dataset": {
            "name": "MovieLens 100K",
            "ratings": int(len(ratings)),
            "users": n_users,
            "items": n_items,
            "train_ratings": int(len(train_df)),
            "test_ratings": int(len(test_df)),
            "split": "80/20 stratified by rating",
            "random_state": RANDOM_STATE,
        },
        "models": {
            "svd": {
                "rmse": round(rmse(actual, svd_test), 4),
                "mae": round(mae(actual, svd_test), 4),
                "n_factors": N_FACTORS,
            },
            "user_based_cf": {
                "rmse": round(rmse(actual, ubcf_test), 4),
                "mae": round(mae(actual, ubcf_test), 4),
                "k_neighbors": K_NEIGHBORS,
            },
        },
        "artifacts": {
            "prediction_matrix": "models/prediction_matrix.npy",
            "seen_mask": "models/seen_mask.npy",
            "user_similarity": "models/user_similarity.npy",
            "svd_components": "models/svd_components.npz",
            "movie_lookup": "models/movie_lookup.csv",
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_readme(results)
    print(json.dumps(results["models"], indent=2))
    print(f"Saved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
