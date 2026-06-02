#!/usr/bin/env python3
"""FastAPI recommendation endpoint."""
from __future__ import annotations
import json
import pathlib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_DIR = pathlib.Path(__file__).parent.parent / "models"

app = FastAPI(title="Movie Recommender", version="1.0")
_state: dict = {}


def _load():
    if _state:
        return
    req = [MODEL_DIR / "svd_components.npz", MODEL_DIR / "user_sim.npy",
           MODEL_DIR / "results.json"]
    if not all(p.exists() for p in req):
        raise HTTPException(503, "Models not trained. Run src/train.py first.")
    svd = np.load(MODEL_DIR / "svd_components.npz")
    _state["U"]         = svd["U"]
    _state["sigma"]     = svd["sigma"]
    _state["Vt"]        = svd["Vt"]
    _state["row_means"] = svd["row_means"]
    _state["sim"]       = np.load(MODEL_DIR / "user_sim.npy")
    _state["pred_mat"]  = (_state["U"] @ np.diag(_state["sigma"])
                           @ _state["Vt"] + _state["row_means"])


class RecommendRequest(BaseModel):
    user_id: int
    top_n: int = 10


class RecommendResponse(BaseModel):
    user_id: int
    recommendations: list[dict]


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    _load()
    u = req.user_id - 1
    if u < 0 or u >= _state["pred_mat"].shape[0]:
        raise HTTPException(400, f"user_id must be 1–{_state['pred_mat'].shape[0]}")
    scores = np.clip(_state["pred_mat"][u], 1, 5)
    top_items = np.argsort(scores)[::-1][: req.top_n]
    recs = [{"item_id": int(i + 1), "predicted_rating": round(float(scores[i]), 2)}
            for i in top_items]
    return RecommendResponse(user_id=req.user_id, recommendations=recs)


@app.get("/health")
def health():
    return {"status": "ok"}
