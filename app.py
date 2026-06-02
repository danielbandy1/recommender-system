#!/usr/bin/env python3
"""Streamlit demo for the MovieLens recommender."""
import pathlib
import numpy as np
import streamlit as st

MODEL_DIR = pathlib.Path(__file__).parent / "models"

st.set_page_config(page_title="Movie Recommender", page_icon="🎬")
st.title("🎬 Movie Recommender")
st.caption("SVD collaborative filtering on MovieLens 100K · 943 users · 1,682 movies")

@st.cache_resource
def load_model():
    svd = np.load(MODEL_DIR / "svd_components.npz")
    pred = svd["U"] @ np.diag(svd["sigma"]) @ svd["Vt"] + svd["row_means"]
    return np.clip(pred, 1, 5)

if not (MODEL_DIR / "svd_components.npz").exists():
    st.error("Model not found. Run `python -m src.train` first.")
    st.stop()

pred_mat = load_model()
n_users  = pred_mat.shape[0]

user_id = st.number_input("User ID (1–943):", min_value=1, max_value=n_users, value=1)
top_n   = st.slider("Number of recommendations:", 5, 20, 10)

if st.button("Get Recommendations", type="primary"):
    scores    = pred_mat[user_id - 1]
    top_items = np.argsort(scores)[::-1][:top_n]
    st.subheader(f"Top {top_n} for User {user_id}")
    for rank, item in enumerate(top_items, 1):
        st.write(f"**{rank}.** Movie {item + 1} — ⭐ {scores[item]:.2f} predicted")
