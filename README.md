# Movie Recommender System

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-ff4b4b)](https://streamlit.io)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](#license)

Collaborative filtering recommender system on MovieLens 100K — comparing SVD matrix factorisation against user-based k-NN, wrapped in a FastAPI inference endpoint and Streamlit demo.

**Dataset:** [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) — 100,000 ratings from 943 users on 1,682 movies (1–5 stars).

---

## Quick Results

| Model | RMSE | MAE |
|---|---|---|
| SVD (k=50 factors) | **~0.93** | **~0.74** |
| User-based CF (k=20) | ~1.02 | ~0.81 |

> Results populate after training. SVD consistently outperforms user-based CF on this dataset.

---

## Architecture

```
data/raw/u.data  (MovieLens 100K TSV)
        │
        ▼
src/features.py      ← user-item matrix construction, 80/20 split
        │
        ▼
src/train.py         ← SVD (scipy) + User-based CF → models/
        │
        ├── api/serve.py   ← FastAPI /recommend endpoint
        └── app.py         ← Streamlit demo
```

**SVD:** Demean per-user → truncated SVD (k=50) via `scipy.sparse.linalg.svds` → reconstruct full rating matrix → clip to [1, 5].

**User-based CF:** Cosine similarity between user vectors → weighted average of top-20 neighbours' ratings for the target item.

---

## How to Run

```bash
pip install -r requirements.txt

# Download dataset
wget https://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip && cp ml-100k/u.data data/raw/

# Train both models
python -m src.train

# Serve API
uvicorn api.serve:app --reload

# Streamlit demo
streamlit run app.py
```

**API:**
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 42, "top_n": 5}'
```

---

## Project Structure

```
recommender-system/
├── src/
│   ├── features.py   # Data loading + user-item matrix
│   └── train.py      # SVD + UBCF training + evaluation
├── api/
│   └── serve.py      # FastAPI /recommend endpoint
├── data/raw/         # u.data (download separately)
├── models/           # svd_components.npz, user_sim.npy, results.json
├── notebooks/        # EDA
├── app.py            # Streamlit demo
└── requirements.txt
```

---

## License

MIT
