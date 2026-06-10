# app/predictor.py

from __future__ import annotations

import hashlib
import logging
import os
from functools import lru_cache
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger("FinanceIQPredictor")

# ============================================================
# OPTIONAL DEPENDENCIES
# ============================================================

try:
    from sklearn.linear_model import LinearRegression

    SKLEARN_AVAILABLE = True

except Exception as e:

    logger.warning(
        "Scikit-learn unavailable: %s",
        str(e)
    )

    SKLEARN_AVAILABLE = False

# SME Optimization: Disable LSTM if requested
LSTM_ENABLED = os.getenv("LSTM_ENABLED", "true").lower() == "true"

try:
    if LSTM_ENABLED:
        from app.lstm_predictor import (
            predict_next as lstm_predict,
            forecast as lstm_forecast,
        )
    else:
        lstm_predict = None
        lstm_forecast = None
        logger.info("LSTM predictor disabled by LSTM_ENABLED=false")
    LSTM_AVAILABLE = LSTM_ENABLED

except Exception as e:

    logger.warning(
        "LSTM predictor unavailable: %s",
        str(e)
    )

    LSTM_AVAILABLE = False

# ============================================================
# CONFIG
# ============================================================

DEFAULT_FORECAST_STEPS = 7

PREDICTOR_ORDER = [
    "lstm",
    "linear",
    "simple",
]

# Override default method from env
PREDICTOR_DEFAULT_METHOD = os.getenv("PREDICTOR_DEFAULT_METHOD", "auto")

# ============================================================
# VALIDATION
# ============================================================

def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and normalize input DataFrame.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Input must be a pandas DataFrame"
        )

    if df.empty:
        raise ValueError(
            "Prediction DataFrame is empty"
        )

    if "amount" not in df.columns:
        raise ValueError(
            "Missing required column: amount"
        )

    df = df.copy()

    # --------------------------------------------------------
    # Convert amounts safely
    # --------------------------------------------------------

    df["amount"] = pd.to_numeric(
        df["amount"],
        errors="coerce"
    )

    df = df.dropna(subset=["amount"])

    if df.empty:
        raise ValueError(
            "No valid numeric amounts available"
        )

    # --------------------------------------------------------
    # Optional chronological sorting
    # --------------------------------------------------------

    if "date" in df.columns:

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

        df = df.sort_values(
            "date"
        ).reset_index(drop=True)

    else:

        df = df.reset_index(drop=True)

    return df

# ============================================================
# DATA FINGERPRINT
# ============================================================

def dataframe_fingerprint(df: pd.DataFrame) -> str:
    """
    Create stable hash for caching models.
    """

    values = (
        df["amount"]
        .astype(str)
        .tolist()
    )

    joined = "|".join(values)

    return hashlib.md5(
        joined.encode("utf-8")
    ).hexdigest()

# ============================================================
# SIMPLE PREDICTION
# ============================================================

def simple_prediction(
    df: pd.DataFrame,
    window: int = 5
) -> Optional[float]:
    """
    Predict using rolling average.
    """

    try:

        df = validate_dataframe(df)

        recent = (
            df["amount"]
            .tail(window)
        )

        prediction = float(
            recent.mean()
        )

        prediction = max(
            prediction,
            0.0
        )

        logger.info(
            "Simple prediction: %.2f",
            prediction
        )

        return prediction

    except Exception as e:

        logger.exception(
            "Simple prediction failed: %s",
            str(e)
        )

        return None

# ============================================================
# LINEAR REGRESSION CACHE
# ============================================================

_LINEAR_MODEL_CACHE: Dict[
    str,
    Any
] = {}

# ============================================================
# LINEAR REGRESSION PREDICTION
# ============================================================

def linear_regression_prediction(
    df: pd.DataFrame
) -> Optional[float]:
    """
    Predict next spending value using
    time-aware linear regression.
    """

    if not SKLEARN_AVAILABLE:

        logger.warning(
            "Linear regression unavailable"
        )

        return None

    try:

        df = validate_dataframe(df)

        if len(df) < 2:

            logger.warning(
                "Insufficient rows for regression"
            )

            return None

        # ----------------------------------------------------
        # Feature engineering
        # ----------------------------------------------------

        if "date" in df.columns:

            base_date = df["date"].min()

            X = (
                (
                    df["date"] - base_date
                )
                .dt.days
                .values
                .reshape(-1, 1)
            )

            next_x = np.array([
                [
                    X[-1][0] + 1
                ]
            ])

        else:

            X = np.arange(
                len(df)
            ).reshape(-1, 1)

            next_x = np.array([
                [len(df)]
            ])

        y = df["amount"].values

        # ----------------------------------------------------
        # Cache model
        # ----------------------------------------------------

        fingerprint = dataframe_fingerprint(df)

        if fingerprint in _LINEAR_MODEL_CACHE:

            model = _LINEAR_MODEL_CACHE[
                fingerprint
            ]

        else:

            model = LinearRegression()

            model.fit(X, y)

            _LINEAR_MODEL_CACHE[
                fingerprint
            ] = model

        prediction = float(
            model.predict(next_x)[0]
        )

        # Prevent negative spending prediction

        prediction = max(
            prediction,
            0.0
        )

        logger.info(
            "Linear regression prediction: %.2f",
            prediction
        )

        return prediction

    except Exception as e:

        logger.exception(
            "Linear regression prediction failed: %s",
            str(e)
        )

        return None

# ============================================================
# LSTM PREDICTION
# ============================================================

def lstm_prediction(
    df: pd.DataFrame
) -> Optional[float]:
    """
    Predict using optional LSTM model.
    """

    if not LSTM_AVAILABLE:

        logger.info(
            "LSTM predictor not available"
        )

        return None

    try:

        df = validate_dataframe(df)

        prediction = lstm_predict(df)

        if prediction is None:

            logger.warning(
                "LSTM returned no prediction"
            )

            return None

        prediction = max(
            float(prediction),
            0.0
        )

        logger.info(
            "LSTM prediction: %.2f",
            prediction
        )

        return prediction

    except Exception as e:

        logger.exception(
            "LSTM prediction failed: %s",
            str(e)
        )

        return None

# ============================================================
# FORECAST
# ============================================================

def forecast_spending(
    df: pd.DataFrame,
    steps: int = DEFAULT_FORECAST_STEPS,
    method: str = "auto"
) -> List[float]:
    """
    Generate multi-step spending forecast.

    Methods:
    - auto
    - lstm
    - linear
    - simple
    """

    try:

        df = validate_dataframe(df)

    except Exception as e:

        logger.exception(
            "Forecast validation failed: %s",
            str(e)
        )

        return []

    # --------------------------------------------------------
    # LSTM FORECAST
    # --------------------------------------------------------

    if method in ("auto", "lstm"):

        if LSTM_AVAILABLE:

            try:

                preds = lstm_forecast(
                    df,
                    steps=steps
                )

                if preds:

                    preds = [
                        max(float(p), 0.0)
                        for p in preds
                    ]

                    logger.info(
                        "LSTM forecast successful"
                    )

                    return preds

            except Exception as e:

                logger.exception(
                    "LSTM forecast failed: %s",
                    str(e)
                )

    # --------------------------------------------------------
    # LINEAR FORECAST
    # --------------------------------------------------------

    if method in ("auto", "linear"):

        try:

            first_pred = linear_regression_prediction(
                df
            )

            if first_pred is not None:

                preds = [
                    float(first_pred)
                    for _ in range(steps)
                ]

                logger.info(
                    "Linear forecast successful"
                )

                return preds

        except Exception as e:

            logger.exception(
                "Linear forecast failed: %s",
                str(e)
            )

    # --------------------------------------------------------
    # SIMPLE FORECAST
    # --------------------------------------------------------

    avg = simple_prediction(df)

    if avg is None:

        logger.warning(
            "Forecast fallback failed"
        )

        return []

    logger.info(
        "Using simple forecast fallback"
    )

    return [
        float(avg)
        for _ in range(steps)
    ]

# ============================================================
# MAIN PREDICTION ROUTER
# ============================================================

def predict_next_spending(
    df: pd.DataFrame,
    preferred_method: str = None
) -> Optional[float]:
    """
    Main prediction entry point.

    Methods:
    - auto
    - lstm
    - linear
    - simple
    """

    if preferred_method is None:
        preferred_method = PREDICTOR_DEFAULT_METHOD

    try:

        df = validate_dataframe(df)

    except Exception as e:

        logger.exception(
            "Prediction validation failed: %s",
            str(e)
        )

        return None

    # --------------------------------------------------------
    # AUTO MODE
    # --------------------------------------------------------

    if preferred_method == "auto":

        for method in PREDICTOR_ORDER:

            if method == "lstm" and LSTM_AVAILABLE:

                pred = lstm_prediction(df)

            elif method == "linear":

                pred = linear_regression_prediction(df)

            elif method == "simple":

                pred = simple_prediction(df)

            else:
                continue

            if pred is not None:

                logger.info(
                    "Prediction selected from %s",
                    method
                )

                return pred

        return None

    # --------------------------------------------------------
    # MANUAL METHOD SELECTION
    # --------------------------------------------------------

    if preferred_method == "lstm":
        if LSTM_AVAILABLE:
            return lstm_prediction(df)
        else:
            logger.warning("LSTM requested but disabled")
            return linear_regression_prediction(df)

    if preferred_method == "linear":
        return linear_regression_prediction(df)

    if preferred_method == "simple":
        return simple_prediction(df)

    logger.warning(
        "Unknown prediction method: %s",
        preferred_method
    )

    return None

# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    sample_df = pd.DataFrame({
        "date": pd.date_range(
            start="2025-01-01",
            periods=10,
            freq="D"
        ),
        "amount": [
            120,
            140,
            135,
            160,
            155,
            170,
            180,
            175,
            190,
            200,
        ]
    })

    print(
        "\nNext prediction:",
        predict_next_spending(sample_df)
    )

    print(
        "\nForecast:",
        forecast_spending(
            sample_df,
            steps=5
        )
    )