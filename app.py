from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from api.serve import recommend_for_user

ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "models" / "results.json"

st.set_page_config(page_title="MovieLens Recommender", page_icon="🎬", layout="wide")
st.title("MovieLens Recommender System")
st.caption("SVD collaborative filtering recommendations from MovieLens 100K.")

if RESULTS_PATH.exists():
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    svd = results["models"]["svd"]
    ubcf = results["models"]["user_based_cf"]
    col1, col2, col3 = st.columns(3)
    col1.metric("SVD RMSE", f"{svd['rmse']:.4f}")
    col2.metric("SVD MAE", f"{svd['mae']:.4f}")
    col3.metric("User CF RMSE", f"{ubcf['rmse']:.4f}")
else:
    st.warning("Model artifacts not found. Run `python3 -m src.train` first.")

user_id = st.number_input("MovieLens user ID", min_value=1, max_value=943, value=42, step=1)
top_n = st.slider("Recommendations", min_value=5, max_value=20, value=10, step=1)

if st.button("Recommend"):
    try:
        recs = recommend_for_user(int(user_id), int(top_n))
        st.subheader(f"Top {top_n} recommendations for user {user_id}")
        st.dataframe(recs, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(str(exc))
