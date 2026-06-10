# app/ingestion.py

import os
import pandas as pd

REQUIRED_COLUMNS = ["date", "description", "amount"]


def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV: {str(e)}")
    df.columns = [c.lower().strip() for c in df.columns]
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["description"] = df["description"].astype(str).str.strip()
    df = df.dropna(subset=["date", "amount"])
    df = df.sort_values("date").reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid data found after cleaning")
    return df