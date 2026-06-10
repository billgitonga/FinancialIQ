# app/pipeline.py

from __future__ import annotations

import os
from typing import Tuple, Dict, Any

import pandas as pd

from app.ingestion import load_data
from app.categorizer import categorize_transactions
from app.trends import analyze_trends
from app.anomaly import detect_anomalies
from app.database import (
    save_transactions,
    save_trends,
    save_anomalies,
)

try:
    from app.logger import logger
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("FinanceIQPipeline")

# ============================================================
# CONFIG
# ============================================================

MAX_AMOUNT = float(os.getenv("MAX_TRANSACTION_AMOUNT", "1000000000"))
MIN_AMOUNT = float(os.getenv("MIN_TRANSACTION_AMOUNT", "-1000000000"))

REQUIRED_COLUMNS = [
    "date",
    "amount",
    "description",
]

# ============================================================
# VALIDATION
# ============================================================

def validate_dataframe(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

# ============================================================
# CLEANING
# ============================================================

def clean_data(
    df: pd.DataFrame,
    warn: bool = True
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    validate_dataframe(df)

    original_rows = len(df)

    stats = {
        "original_rows": original_rows,
        "invalid_amount_rows": 0,
        "invalid_date_rows": 0,
        "duplicate_rows_removed": 0,
        "outlier_rows_removed": 0,
        "empty_description_rows": 0,
        "final_rows": 0,
    }

    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]

    # Clean description
    df["description"] = (
        df["description"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    empty_desc_mask = (df["description"] == "") | (df["description"].str.lower() == "nan")
    stats["empty_description_rows"] = int(empty_desc_mask.sum())

    # Convert amounts
    df["amount"] = (
        df["amount"]
        .astype(str)
        .str.replace(r"[^\d\-,.]", "", regex=True)
        .str.replace(",", "", regex=False)
    )
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    invalid_amount_mask = df["amount"].isna()
    stats["invalid_amount_rows"] = int(invalid_amount_mask.sum())
    df = df.dropna(subset=["amount"])

    # Remove outliers
    outlier_mask = (df["amount"] > MAX_AMOUNT) | (df["amount"] < MIN_AMOUNT)
    stats["outlier_rows_removed"] = int(outlier_mask.sum())
    df = df[~outlier_mask]

    # Parse dates (make timezone naive)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if hasattr(df["date"].dt, 'tz') and df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    invalid_date_mask = df["date"].isna()
    stats["invalid_date_rows"] = int(invalid_date_mask.sum())
    df = df.dropna(subset=["date"])

    # Remove duplicates
    before_dupes = len(df)
    df = df.drop_duplicates()
    stats["duplicate_rows_removed"] = before_dupes - len(df)

    # Sort chronologically
    df = df.sort_values("date").reset_index(drop=True)
    stats["final_rows"] = len(df)

    if warn:
        if stats["invalid_amount_rows"] > 0:
            logger.warning("Dropped %s rows with invalid amounts", stats["invalid_amount_rows"])
        if stats["invalid_date_rows"] > 0:
            logger.warning("Dropped %s rows with invalid dates", stats["invalid_date_rows"])
        if stats["duplicate_rows_removed"] > 0:
            logger.info("Removed %s duplicate rows", stats["duplicate_rows_removed"])
        if stats["outlier_rows_removed"] > 0:
            logger.warning("Removed %s outlier rows", stats["outlier_rows_removed"])

    return df, stats

# ============================================================
# PIPELINE
# ============================================================

def run_pipeline(
    path: str,
    username: str = "default"
) -> Tuple[pd.DataFrame, Any, Any]:
    """
    Full FinanceIQ processing pipeline.
    """
    logger.info("Starting pipeline for file: %s", path)

    # LOAD
    try:
        logger.info("Loading data...")
        df = load_data(path)
    except Exception as e:
        logger.exception("Failed to load data: %s", str(e))
        raise

    # CLEAN
    logger.info("Cleaning data...")
    df, stats = clean_data(df)
    logger.info("Cleaning summary: %s", stats)

    if df.empty:
        logger.error("No valid rows remain after cleaning")
        raise ValueError("No valid transaction data after cleaning")

    logger.info("Cleaned rows: %s", len(df))

    # CATEGORIZATION
    try:
        logger.info("Categorizing transactions...")
        df = categorize_transactions(df)
    except Exception as e:
        logger.exception("Categorization failed: %s", str(e))
        if "category" not in df.columns:
            df["category"] = "uncategorized"

    # TREND ANALYSIS
    try:
        logger.info("Analyzing trends...")
        trends = analyze_trends(df)
    except Exception as e:
        logger.exception("Trend analysis failed: %s", str(e))
        trends = None

    # ANOMALY DETECTION
    try:
        logger.info("Detecting anomalies...")
        anomalies = detect_anomalies(df)
    except Exception as e:
        logger.exception("Anomaly detection failed: %s", str(e))
        anomalies = pd.DataFrame()

    # SAVE RESULTS - silently handle errors (don't break the pipeline)
    try:
        save_success, save_msg = save_transactions(df, username)
        if save_success:
            logger.info("Transactions saved: %s", save_msg)
    except Exception as e:
        logger.warning("Transaction save error (non-critical): %s", str(e))

    logger.info("Pipeline completed successfully")
    return df, trends, anomalies


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    sample_file = "transactions.csv"
    try:
        transactions, trends, anomalies = run_pipeline(sample_file)
        print("\n✅ Pipeline completed successfully")
        print(f"Transactions: {len(transactions)}")
        print(transactions.head())
    except Exception as e:
        logger.exception("Pipeline execution failed: %s", str(e))
        print(f"\n❌ Pipeline failed: {str(e)}")