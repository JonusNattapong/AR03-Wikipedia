# app/utils.py
import pandas as pd
import json
import os
from typing import Any

def load_json(filepath: str) -> Any:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data: Any, filepath: str):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_parquet(filepath: str) -> pd.DataFrame:
    return pd.read_parquet(filepath)

def save_parquet(df: pd.DataFrame, filepath: str):
    df.to_parquet(filepath, index=False)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def batch(iterable, size=10):
    for i in range(0, len(iterable), size):
        yield iterable[i:i+size]
