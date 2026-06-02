from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

app = FastAPI(title="MovieLens Recommender API", version="1.0.0")


class RecommendRequest(BaseModel):
    user_id: int = Field(..., ge=1, description="MovieLens user id, 1 through 943")
    top_n: int = Field(10, ge=1, le=50, description="Number of recommendations")


class Recommendation(BaseModel):
    item_id: int
    title: str
    predicted_rating: float


class RecommendResponse(BaseModel):
    user_id: int
    top_n: int
    recommendations: list[Recommendation]


@lru_cache(maxsize=1)
def load_artifacts() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    pred_path = MODELS_DIR / "prediction_matrix.npy"
    seen_path = MODELS_DIR / "seen_mask.npy"
    movie_path = MODELS_DIR / "movie_lookup.csv"
    missing = [str(p) for p in (pred_path, seen_path, movie_path) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing model artifacts. Run `python3 -m src.train` first. Missing: " + ", ".join(missing))
    predictions = np.load(pred_path)
    seen_mask = np.load(seen_path)
    movies = pd.read_csv(movie_path).set_index("item_id")
    return predictions, seen_mask, movies


def recommend_for_user(user_id: int, top_n: int) -> list[dict]:
    try:
        predictions, seen_mask, movies = load_artifacts()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if user_id < 1 or user_id > predictions.shape[0]:
        raise HTTPException(status_code=400, detail=f"user_id must be between 1 and {predictions.shape[0]}")

    user_idx = user_id - 1
    scores = predictions[user_idx].copy()
    candidates = np.where(~seen_mask[user_idx])[0]
    if candidates.size == 0:
        candidates = np.arange(scores.shape[0])

    order = candidates[np.argsort(scores[candidates])[::-1]][:top_n]
    recs = []
    for item_idx in order:
        item_id = int(item_idx + 1)
        title = str(movies.loc[item_id, "title"]) if item_id in movies.index else f"Movie {item_id}"
        recs.append(
            {
                "item_id": item_id,
                "title": title,
                "predicted_rating": round(float(scores[item_idx]), 3),
            }
        )
    return recs


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(payload: RecommendRequest) -> RecommendResponse:
    recs = recommend_for_user(payload.user_id, payload.top_n)
    return RecommendResponse(user_id=payload.user_id, top_n=payload.top_n, recommendations=recs)
