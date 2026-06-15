# app/health.py

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any


DEFAULT_CONFIG = {
    "weights": {
        "stability": 0.25,
        "balance": 0.20,
        "essential_spending": 0.20,
        "spending_spikes": 0.15,
        "spending_velocity": 0.20
    },
    "thresholds": {
        "low_volatility": 0.5,
        "medium_volatility": 1.0,
        "high_volatility": 1.5,
        "spike_multiplier": 2.0,
        "minimum_transactions": 3
    },
    "essential_categories": [
        "food",
        "groceries",
        "rent",
        "housing",
        "utilities",
        "transport",
        "healthcare",
        "education"
    ]
}


def _validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean input DataFrame.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    if df.empty:
        return pd.DataFrame(columns=["amount", "category", "date"])

    required_columns = ["amount"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: '{col}'")

    cleaned = df.copy()

    cleaned["amount"] = pd.to_numeric(cleaned["amount"], errors="coerce")
    cleaned = cleaned.dropna(subset=["amount"])

    if "category" not in cleaned.columns:
        cleaned["category"] = "uncategorized"

    return cleaned


def _filter_current_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter records for the current month.
    """
    if df.empty or "date" not in df.columns:
        return df.copy()

    filtered = df.copy()

    filtered["date"] = pd.to_datetime(
        filtered["date"],
        errors="coerce"
    )

    filtered = filtered.dropna(subset=["date"])

    now = datetime.now()

    return filtered[
        (filtered["date"].dt.month == now.month) &
        (filtered["date"].dt.year == now.year)
    ].copy()


def _category_distribution(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate spending distribution percentages by category.
    Uses only positive expenses.
    """
    if df.empty:
        return {}

    expenses = df[df["amount"] > 0]

    if expenses.empty:
        return {}

    total = expenses["amount"].sum()

    if total <= 0:
        return {}

    grouped = (
        expenses.groupby("category")["amount"]
        .sum()
        .sort_values(ascending=False)
    )

    return {
        str(category): round((amount / total) * 100, 2)
        for category, amount in grouped.items()
    }


def _spending_stability(
    df: pd.DataFrame,
    config: Dict[str, Any]
) -> float:
    """
    Measure spending stability using coefficient of variation.
    """
    if df.empty:
        return 0.0

    expenses = df[df["amount"] > 0]

    if len(expenses) < config["thresholds"]["minimum_transactions"]:
        return 50.0

    amounts = expenses["amount"]

    mean = amounts.mean()
    std = amounts.std(ddof=0) if len(amounts) > 1 else 0.0

    if mean <= 0 or np.isnan(mean):
        return 50.0

    volatility = std / mean

    low = config["thresholds"]["low_volatility"]
    medium = config["thresholds"]["medium_volatility"]
    high = config["thresholds"]["high_volatility"]

    if volatility < low:
        return 90.0
    elif volatility < medium:
        return 75.0
    elif volatility < high:
        return 60.0
    else:
        return 40.0


def _category_balance(dist: Dict[str, float]) -> float:
    """
    Measure how balanced spending is across categories.
    """
    if not dist:
        return 50.0

    max_category = max(dist.values())

    if max_category < 30:
        return 90.0
    elif max_category < 50:
        return 75.0
    elif max_category < 70:
        return 60.0
    else:
        return 40.0


def _essential_vs_nonessential(
    dist: Dict[str, float],
    essential_categories: List[str]
) -> float:
    """
    Evaluate proportion of spending in essential categories.
    """
    if not dist:
        return 50.0

    essential_categories = {
        category.lower()
        for category in essential_categories
    }

    essential_pct = sum(
        percentage
        for category, percentage in dist.items()
        if category.lower() in essential_categories
    )

    if essential_pct > 70:
        return 90.0
    elif essential_pct > 50:
        return 75.0
    elif essential_pct > 30:
        return 60.0
    else:
        return 40.0


def _high_spend_penalty(df: pd.DataFrame, iqr_multiplier: float) -> float:
    """
    Detect unusually large spending spikes using IQR-based outlier detection.
    """
    if df.empty:
        return 50.0

    expenses = df[df["amount"] > 0]

    if len(expenses) < 4:
        return 75.0

    amounts = expenses["amount"]

    q1 = amounts.quantile(0.25)
    q3 = amounts.quantile(0.75)

    iqr = q3 - q1

    if iqr == 0:
        return 90.0

    upper_bound = q3 + (iqr_multiplier * iqr)

    spikes = expenses[expenses["amount"] > upper_bound]

    spike_ratio = len(spikes) / len(expenses)

    if spike_ratio == 0:
        return 90.0
    elif spike_ratio < 0.05:
        return 75.0
    elif spike_ratio < 0.15:
        return 60.0
    else:
        return 40.0


def _calculate_spending_velocity(df: pd.DataFrame) -> float:
    """
    Measure spending velocity (burn rate) relative to typical patterns.
    Higher velocity = faster spending, may indicate risk.
    """
    if df.empty or "date" not in df.columns:
        return 50.0

    expenses = df[df["amount"] > 0]
    if expenses.empty:
        return 50.0

    expenses = expenses.copy()
    expenses["date"] = pd.to_datetime(expenses["date"], errors="coerce")
    expenses = expenses.dropna(subset=["date"])

    if expenses.empty:
        return 50.0

    daily_spending = expenses.groupby(expenses["date"].dt.date)["amount"].sum()
    mean_daily = daily_spending.mean()

    if mean_daily <= 0 or np.isnan(mean_daily):
        return 50.0

    daily_std = daily_spending.std(ddof=0) if len(daily_spending) > 1 else 0.0
    daily_volatility = daily_std / mean_daily if mean_daily > 0 else 0

    if daily_volatility < 0.3:
        return 90.0
    elif daily_volatility < 0.6:
        return 75.0
    elif daily_volatility < 1.0:
        return 60.0
    else:
        return 40.0


def _calculate_components(
    df: pd.DataFrame,
    config: Dict[str, Any]
) -> Dict[str, float]:
    """
    Calculate all health score components.
    """
    distribution = _category_distribution(df)

    stability = _spending_stability(df, config)
    balance = _category_balance(distribution)
    essential_score = _essential_vs_nonessential(
        distribution,
        config["essential_categories"]
    )
    spike_score = _high_spend_penalty(
        df,
        config["thresholds"]["spike_multiplier"]
    )
    velocity_score = _calculate_spending_velocity(df)

    return {
        "stability": stability,
        "balance": balance,
        "essential_spending": essential_score,
        "spending_spikes": spike_score,
        "spending_velocity": velocity_score,
        "distribution": distribution,
    }


def _compute_weighted_score(
    components: Dict[str, float],
    weights: Dict[str, float]
) -> float:
    """
    Compute weighted score from components.
    """
    return round(
        sum(
            components[k] * weights[k]
            for k in [
                "stability", "balance",
                "essential_spending", "spending_spikes",
                "spending_velocity"
            ]
            if k in components
        ),
        2
    )


def calculate_health_score(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None
) -> Optional[float]:
    """
    Calculate overall financial health score.
    """
    config = config or DEFAULT_CONFIG

    df = _validate_dataframe(df)
    df = _filter_current_month(df)

    if df.empty:
        return None

    components = _calculate_components(df, config)

    return _compute_weighted_score(components, config["weights"])


def health_report(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate a detailed financial health report.
    """
    config = config or DEFAULT_CONFIG

    df = _validate_dataframe(df)
    df = _filter_current_month(df)

    if df.empty:
        return {
            "score": None,
            "message": "No valid transaction data available",
            "components": {},
            "category_distribution": {}
        }

    components = _calculate_components(df, config)
    score = _compute_weighted_score(components, config["weights"])

    return {
        "score": score,
        "components": {
            "stability": components["stability"],
            "balance": components["balance"],
            "essential_spending": components["essential_spending"],
            "spending_spikes": components["spending_spikes"],
            "spending_velocity": components["spending_velocity"]
        },
        "category_distribution": components["distribution"],
        "transaction_count": len(df),
        "generated_at": datetime.now().isoformat()
    }


def health_insights(
    df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    Generate human-readable financial health insights.
    """
    report = health_report(df, config)

    if report["score"] is None:
        return ["No sufficient financial data available"]

    insights = []

    score = report["score"]

    if score >= 80:
        insights.append("✅ Excellent financial health")
    elif score >= 60:
        insights.append("👍 Good financial health with room for improvement")
    else:
        insights.append("⚠️ Financial health requires attention")

    components = report.get("components", {})

    if components.get("stability", 0) < 60:
        insights.append(
            "Your spending pattern is highly volatile. "
            "Consider maintaining more consistent expenses."
        )

    if components.get("balance", 0) < 60:
        insights.append(
            "A large portion of your spending is concentrated "
            "in one category."
        )

    if components.get("essential_spending", 0) < 50:
        insights.append(
            "High discretionary spending detected. "
            "Review non-essential expenses."
        )

    if components.get("spending_spikes", 0) < 60:
        insights.append(
            "Frequent unusually large transactions detected."
        )

    if components.get("spending_velocity", 0) < 60:
        insights.append(
            "Spending velocity is high. "
            "Monitor daily spending patterns."
        )

    top_categories = report.get("category_distribution", {})

    if top_categories:
        sorted_cats = sorted(
            top_categories.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        if sorted_cats:
            top_cat_str = ", ".join(
                f"{cat} ({pct:.1f}%)" for cat, pct in sorted_cats
            )
            insights.append(f"Top spending categories: {top_cat_str}")

    return insights