from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
ML_DIR = DATA_DIR / "ml-100k"
RATINGS_PATH = ML_DIR / "u.data"
ITEMS_PATH = ML_DIR / "u.item"
DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def ensure_movielens(data_dir: Path | None = None) -> Path:
    data_dir = Path(data_dir or DATA_DIR)
    ml_dir = data_dir / "ml-100k"
    ratings_path = ml_dir / "u.data"
    if ratings_path.exists() and (ml_dir / "u.item").exists():
        return ml_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "ml-100k.zip"
    if not zip_path.exists():
        print(f"Downloading MovieLens 100K to {zip_path}...")
        urlretrieve(DATA_URL, zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            target = (data_dir / member).resolve()
            if not str(target).startswith(str(data_dir.resolve())):
                raise RuntimeError(f"Unsafe zip member: {member}")
        zf.extractall(data_dir)

    if not ratings_path.exists():
        raise FileNotFoundError(f"MovieLens ratings file not found: {ratings_path}")
    return ml_dir


def load_ratings(data_dir: Path | None = None) -> pd.DataFrame:
    ml_dir = ensure_movielens(data_dir)
    df = pd.read_csv(
        ml_dir / "u.data",
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )
    return df.astype({"user_id": int, "item_id": int, "rating": float, "timestamp": int})


def load_movies(data_dir: Path | None = None) -> pd.DataFrame:
    ml_dir = ensure_movielens(data_dir)
    cols = [
        "item_id",
        "title",
        "release_date",
        "video_release_date",
        "imdb_url",
    ]
    genre_cols = [f"genre_{i}" for i in range(19)]
    movies = pd.read_csv(
        ml_dir / "u.item",
        sep="|",
        names=cols + genre_cols,
        encoding="latin-1",
        engine="python",
    )
    return movies[["item_id", "title"]].astype({"item_id": int})


def split_ratings(
    ratings: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        ratings,
        test_size=test_size,
        random_state=random_state,
        stratify=ratings["rating"],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_user_item_matrix(
    ratings: pd.DataFrame,
    n_users: int | None = None,
    n_items: int | None = None,
) -> np.ndarray:
    n_users = int(n_users or ratings["user_id"].max())
    n_items = int(n_items or ratings["item_id"].max())
    matrix = np.zeros((n_users, n_items), dtype=np.float32)
    matrix[
        ratings["user_id"].to_numpy(dtype=int) - 1,
        ratings["item_id"].to_numpy(dtype=int) - 1,
    ] = ratings["rating"].to_numpy(dtype=np.float32)
    return matrix
