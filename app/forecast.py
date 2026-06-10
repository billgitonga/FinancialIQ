# app/forecast.py
"""
Simple cashflow forecasting helpers for FinanceIQ.

Uses monthly aggregation and a linear trend fit to project next N months.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Dict

import numpy as np
import pandas as pd


def forecast_cashflow(df: pd.DataFrame, periods: int = 6, date_col: str = "date", amount_col: str = "amount") -> Dict:
    """Forecast monthly cashflow (net) for the next `periods` months.

    Input `df` should contain a date-like column and an amount column.
    Returns a dict with monthly history and forecasted values.
    """
    if df is None or df.empty:
        return {"history": [], "forecast": []}

    # normalize dates
    dates = pd.to_datetime(df[date_col], errors="coerce")
    mask = ~dates.isna()
    df = df.loc[mask].copy()
    df["_month"] = dates.dt.to_period("M").dt.to_timestamp()

    monthly = df.groupby("_month")[amount_col].sum().reset_index()

    if monthly.empty:
        return {"history": [], "forecast": []}

    # x = months index, y = amounts
    monthly = monthly.sort_values("_month")
    x = np.arange(len(monthly))
    y = monthly[amount_col].values.astype(float)

    # simple linear fit
    try:
        coeffs = np.polyfit(x, y, 1)
        slope, intercept = coeffs[0], coeffs[1]
    except Exception:
        slope, intercept = 0.0, float(np.mean(y))

    forecast_vals = []
    for i in range(1, periods + 1):
        xi = len(monthly) + i - 1
        pred = intercept + slope * xi
        forecast_vals.append(float(pred))

    history = [
        {"month": ts.strftime("%Y-%m"), "amount": float(val)}
        for ts, val in zip(monthly['_month'].tolist(), monthly[amount_col].tolist())
    ]

    forecast = []
    last_month = monthly['_month'].iloc[-1]
    for i, val in enumerate(forecast_vals, start=1):
        next_month = (last_month + pd.DateOffset(months=i))
        forecast.append({"month": next_month.strftime("%Y-%m"), "amount": float(val)})

    return {"history": history, "forecast": forecast}
