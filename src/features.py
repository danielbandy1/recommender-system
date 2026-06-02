#!/usr/bin/env python3
"""Load MovieLens 100K and build user-item matrix."""
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR  = pathlib.Path(__file__).parent.parent / "data" / "raw"
RATINGS   = DATA_DIR / "u.data"

COLUMNS   = ["user_id", "item_id", "rating", "timestamp"]


def load_ratings() -> pd.DataFrame:
    return pd.read_csv(RATINGS, sep="\t", names=COLUMNS, engine="python")


def build_matrix(df: pd.DataFrame):
    n_users = df["user_id"].max()
    n_items = df["item_id"].max()
    mat = np.zeros((n_users, n_items), dtype=np.float32)
    for row in df.itertuples(index=False):
        mat[row.user_id - 1, row.item_id - 1] = row.rating
    return mat


def split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    train, test = train_test_split(df, test_size=test_size, random_state=seed)
    return train.reset_index(drop=True), test.reset_index(drop=True)
