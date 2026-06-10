# app/anomaly.py
# ============================================================
# FinanceIQ Intelligent Anomaly Detection Engine (SME Optimized)
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_Z_THRESHOLD = 3.5
DEFAULT_CONTAMINATION = 0.03
ROLLING_WINDOW = 7

# SME: Limit lookback period for performance
DEFAULT_LOOKBACK_DAYS = int(os.getenv("FINANCEIQ_ANOMALY_LOOKBACK_DAYS", "90"))

# ============================================================
# DATA VALIDATION
# ============================================================

def validate_dataframe(df):

    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty.")

    required_columns = ["amount"]

    for col in required_columns:

        if col not in df.columns:
            raise ValueError(
                f"Missing required column: {col}"
            )

    df = df.copy()

    # Ensure numeric amount
    df["amount"] = pd.to_numeric(
        df["amount"],
        errors="coerce"
    )

    df = df.dropna(subset=["amount"])

    # Ensure datetime - handle timezone by making naive
    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )
        # Convert to timezone-naive if it has timezone info
        if hasattr(df["date"].dt, 'tz') and df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_localize(None)
    else:
        df["date"] = pd.NaT

    # Default category
    if "category" not in df.columns:
        df["category"] = "other"

    # Default transaction type
    if "type" not in df.columns:
        df["type"] = "expense"

    return df

# ============================================================
# ROBUST Z-SCORE (MAD BASED)
# ============================================================

def robust_zscore(series):

    median = np.median(series)

    mad = np.median(
        np.abs(series - median)
    )

    if mad == 0:

        return np.zeros(len(series))

    modified_z = (
        0.6745 * (series - median) / mad
    )

    return modified_z

# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(df):

    features = pd.DataFrame(index=df.index)

    # ========================================================
    # LOG AMOUNT
    # ========================================================

    features["log_amount"] = np.log1p(
        np.abs(df["amount"])
    )

    # ========================================================
    # WEEKDAY
    # ========================================================

    # Handle possible NaT in date
    if df["date"].notna().any():
        features["weekday"] = (
            df["date"]
            .dt.weekday
            .fillna(0)
        )
    else:
        features["weekday"] = 0

    # ========================================================
    # MONTHDAY
    # ========================================================

    if df["date"].notna().any():
        features["monthday"] = (
            df["date"]
            .dt.day
            .fillna(1)
        )
    else:
        features["monthday"] = 1

    # ========================================================
    # CATEGORY FREQUENCY
    # ========================================================

    cat_freq = (
        df["category"]
        .value_counts(normalize=True)
    )

    features["category_freq"] = (
        df["category"]
        .map(cat_freq)
        .fillna(0)
    )

    # ========================================================
    # ROLLING MEAN
    # ========================================================

    rolling_mean = (
        df["amount"]
        .rolling(
            window=ROLLING_WINDOW,
            min_periods=1
        )
        .mean()
    )

    features["rolling_mean"] = rolling_mean

    # ========================================================
    # ROLLING STD
    # ========================================================

    rolling_std = (
        df["amount"]
        .rolling(
            window=ROLLING_WINDOW,
            min_periods=1
        )
        .std()
        .fillna(0)
    )

    features["rolling_std"] = rolling_std

    # ========================================================
    # DEVIATION FROM ROLLING AVG
    # ========================================================

    features["rolling_deviation"] = (
        np.abs(
            df["amount"] - rolling_mean
        ) / (rolling_std + 1)
    )

    return features

# ============================================================
# ROBUST Z-SCORE ANOMALIES
# ============================================================

def robust_zscore_anomalies(
    df,
    threshold=DEFAULT_Z_THRESHOLD
):

    df = df.copy()

    # Separate income vs expense
    expense_mask = df["amount"] < 0
    income_mask = df["amount"] >= 0

    z_scores = np.zeros(len(df))

    if expense_mask.any():

        z_scores[expense_mask] = robust_zscore(
            df.loc[expense_mask, "amount"]
        )

    if income_mask.any():

        z_scores[income_mask] = robust_zscore(
            df.loc[income_mask, "amount"]
        )

    df["robust_zscore"] = z_scores

    df["z_anomaly"] = (
        np.abs(z_scores) > threshold
    )

    return df

# ============================================================
# ISOLATION FOREST
# ============================================================

def isolation_forest_anomalies(
    df,
    contamination=DEFAULT_CONTAMINATION
):

    df = df.copy()

    features = build_features(df)

    scaler = RobustScaler()

    X = scaler.fit_transform(features)

    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=200
    )

    preds = model.fit_predict(X)

    scores = model.decision_function(X)

    df["if_anomaly"] = preds == -1

    df["if_score"] = -scores

    return df

# ============================================================
# CATEGORY DEVIATION
# ============================================================

def compute_category_deviation(df):

    df = df.copy()

    cat_mean = (
        df.groupby("category")["amount"]
        .transform("mean")
    )

    cat_median = (
        df.groupby("category")["amount"]
        .transform("median")
    )

    cat_std = (
        df.groupby("category")["amount"]
        .transform("std")
        .fillna(1)
        .replace(0, 1)
    )

    df["category_deviation"] = (
        np.abs(df["amount"] - cat_median)
        / cat_std
    )

    return df

# ============================================================
# WEEKDAY DEVIATION
# ============================================================

def compute_weekday_deviation(df):

    df = df.copy()

    # Only compute if we have valid dates
    if df["date"].notna().any():
        weekday_avg = (
            df.groupby(
                df["date"].dt.weekday
            )["amount"]
            .transform("mean")
        )
    else:
        weekday_avg = df["amount"].median()

    weekday_avg = weekday_avg.fillna(
        df["amount"].median()
    )

    df["weekday_deviation"] = (
        np.abs(df["amount"] - weekday_avg)
        / (np.abs(weekday_avg) + 1)
    )

    return df

# ============================================================
# COMBINED ANOMALY SCORE
# ============================================================

def compute_anomaly_score(df):

    df = df.copy()

    # normalized components
    z_component = np.clip(
        np.abs(df["robust_zscore"]) / 5,
        0,
        1
    )

    cat_component = np.clip(
        df["category_deviation"] / 5,
        0,
        1
    )

    weekday_component = np.clip(
        df["weekday_deviation"] / 5,
        0,
        1
    )

    if_component = np.clip(
        df["if_score"],
        0,
        1
    )

    # weighted combination
    df["anomaly_score"] = (

        0.35 * z_component +

        0.25 * cat_component +

        0.20 * weekday_component +

        0.20 * if_component
    )

    return df

# ============================================================
# EXPLANATION ENGINE
# ============================================================

def generate_explanations(df):

    explanations = []

    for _, row in df.iterrows():

        reasons = []

        if row["z_anomaly"]:

            reasons.append(
                f"Amount deviates strongly "
                f"(robust z-score="
                f"{row['robust_zscore']:.2f})"
            )

        if row["if_anomaly"]:

            reasons.append(
                "Isolation Forest detected "
                "unusual spending behavior"
            )

        if row["category_deviation"] > 3:

            reasons.append(
                "Transaction is unusual "
                "for its category"
            )

        if row["weekday_deviation"] > 2:

            reasons.append(
                "Transaction deviates "
                "from normal weekday behavior"
            )

        if not reasons:

            explanations.append("")

        else:

            explanations.append(
                "; ".join(reasons)
            )

    df["explanation"] = explanations

    return df

# ============================================================
# MAIN DETECTION PIPELINE
# ============================================================

def detect_anomalies(
    df,
    z_threshold=DEFAULT_Z_THRESHOLD,
    contamination=DEFAULT_CONTAMINATION,
    return_full=False,
    lookback_days=None
):
    # Use env default if not provided
    if lookback_days is None:
        lookback_days = DEFAULT_LOOKBACK_DAYS

    # ========================================================
    # VALIDATION
    # ========================================================

    df = validate_dataframe(df)

    # ========================================================
    # APPLY LOOKBACK FILTER (SME optimization)
    # ========================================================
    if lookback_days > 0 and "date" in df.columns and not df["date"].isna().all():
        # Make cutoff timezone-naive for comparison
        cutoff = datetime.now()
        cutoff = cutoff - timedelta(days=lookback_days)
        df = df[df["date"] >= cutoff]

    if df.empty:
        # Return empty DataFrame with expected columns
        empty_columns = list(df.columns) + ["anomaly", "anomaly_score", "explanation"]
        empty_df = pd.DataFrame(columns=empty_columns)
        return empty_df

    # ========================================================
    # SORT TEMPORALLY
    # ========================================================

    df = df.sort_values(
        by="date",
        ascending=True
    )

    # ========================================================
    # ROBUST Z-SCORE
    # ========================================================

    df = robust_zscore_anomalies(
        df,
        threshold=z_threshold
    )

    # ========================================================
    # ISOLATION FOREST
    # ========================================================

    df = isolation_forest_anomalies(
        df,
        contamination=contamination
    )

    # ========================================================
    # CATEGORY DEVIATION
    # ========================================================

    df = compute_category_deviation(df)

    # ========================================================
    # WEEKDAY DEVIATION
    # ========================================================

    df = compute_weekday_deviation(df)

    # ========================================================
    # FINAL ANOMALY SCORE
    # ========================================================

    df = compute_anomaly_score(df)

    # ========================================================
    # COMBINED ANOMALY DECISION
    # ========================================================

    df["anomaly"] = (

        df["z_anomaly"]

        |

        df["if_anomaly"]

        |

        (df["anomaly_score"] > 0.70)
    )

    # ========================================================
    # EXPLANATIONS
    # ========================================================

    df = generate_explanations(df)

    # ========================================================
    # SORT BY SEVERITY
    # ========================================================

    df = df.sort_values(
        by="anomaly_score",
        ascending=False
    )

    # ========================================================
    # RETURN
    # ========================================================

    if return_full:

        return df

    anomalies = df[
        df["anomaly"]
    ].copy()

    return anomalies

# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":

    sample = pd.DataFrame({

        "date": pd.date_range(
            start="2025-01-01",
            periods=30
        ),

        "amount": [

            50, 45, 52, 48, 51,
            49, 47, 1000, 46, 52,
            51, 49, 53, 47, 48,
            5000, 49, 50, 52, 51,
            47, 49, 46, 48, 50,
            7000, 52, 51, 49, 50
        ],

        "category": [

            "food"
        ] * 30,

        "type": [

            "expense"
        ] * 30
    })

    anomalies = detect_anomalies(
        sample,
        return_full=False
    )

    print("\n================================================")
    print("ANOMALIES DETECTED")
    print("================================================\n")

    if not anomalies.empty:
        print(anomalies[[
            "date",
            "amount",
            "anomaly_score",
            "explanation"
        ]])
    else:
        print("No anomalies found")