# MovieLens Recommender System

Portfolio-quality collaborative filtering project using the MovieLens 100K ratings dataset.

## Objective

Compare two recommendation approaches on 100,000 movie ratings from 943 users and 1,682 movies:

- SVD matrix factorization using `scipy.sparse.linalg.svds`
- User-based collaborative filtering with cosine similarity

The target benchmark is about 0.92 RMSE on a held-out 20% test split.

## Results

| Model | RMSE | MAE | Notes |
|---|---:|---:|---|
| SVD matrix factorization | 0.9350 | 0.7365 | 50 latent factors plus user/item bias baseline |
| User-based collaborative filtering | 0.9359 | 0.7318 | cosine similarity, top 30 neighbors |

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
curl -X POST http://127.0.0.1:8000/recommend   -H "Content-Type: application/json"   -d '{"user_id": 42, "top_n": 10}'
```

Response:

```json
{
  "user_id": 42,
  "recommendations": [
    {"item_id": 318, "title": "Schindler's List (1993)", "predicted_rating": 4.73}
  ]
}
```

## Notes

- Ratings are one-indexed in the raw data; model arrays are zero-indexed internally.
- Recommendations filter out movies the user already rated in the training split.
- If a user has rated every candidate in the training matrix, the API falls back to ranking all movies by predicted score.
