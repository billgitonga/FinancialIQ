# app/financial_ai.py

from typing import Dict, List, Optional, Any

import pandas as pd


_full_analysis_cache: Dict[str, Dict[str, Any]] = {}


def _df_hash(df: pd.DataFrame) -> str:
    if df.empty:
        return "empty"
    try:
        return str(hash(df.to_json()))
    except Exception:
        return "empty"


def calculate_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {
            "total": 0.0,
            "average": 0.0,
            "transaction_count": 0,
            "top_category": "unknown",
            "category_breakdown": {},
        }
    amounts = df["amount"]
    total = float(amounts.sum())
    avg = float(amounts.mean())
    breakdown = {}
    if "category" in df.columns:
        breakdown = df.groupby("category")["amount"].sum().round(2).to_dict()
    top_category = max(breakdown, key=breakdown.get) if breakdown else "unknown"
    return {
        "total": total,
        "average": avg,
        "transaction_count": int(len(df)),
        "top_category": top_category,
        "category_breakdown": breakdown,
    }


def detect_risk(df: pd.DataFrame) -> List[Dict[str, str]]:
    risks = []
    if df.empty:
        return risks
    metrics = calculate_metrics(df)
    breakdown = metrics.get("category_breakdown", {})
    for category, amount in sorted(breakdown.items(), key=lambda x: x[1], reverse=True)[:3]:
        pct = amount / metrics["total"] * 100 if metrics["total"] > 0 else 0
        if pct > 50:
            risks.append({
                "severity": "medium",
                "message": f"Category '{category}' accounts for {pct:.1f}% of all spending.",
            })
    high_avg = metrics.get("average", 0) > 1000
    if high_avg:
        risks.append({
            "severity": "low",
            "message": "Average transaction amount is notably high.",
        })
    return risks


def generate_ai_insights(metrics: Dict[str, Any], risks: List[Dict[str, str]]) -> List[str]:
    insights = [
        f"Total spending analysed: {metrics['total']:.2f}",
        f"Average transaction value: {metrics['average']:.2f}",
        f"Transactions analysed: {metrics['transaction_count']}",
    ]
    if metrics.get("top_category"):
        insights.append(f"Highest spending category: {metrics['top_category']}")
    for risk in risks:
        insights.append(f"Risk detected ({risk['severity']}): {risk['message']}")
    return insights


def analyze_future(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {"next": 0.0, "forecast": [], "trend": "unknown"}
    metrics = calculate_metrics(df)
    return {
        "next": round(metrics.get("average", 0.0), 2),
        "forecast": [],
        "trend": "stable",
    }


def optimize_and_explain(df: pd.DataFrame):
    if df.empty:
        return {}, []
    metrics = calculate_metrics(df)
    recommendations = [
        "Review recurring subscriptions to reduce monthly expenses.",
        "Set category budgets to keep spending in check.",
        "Track small daily expenses to identify savings opportunities.",
    ]
    return {}, recommendations


def _cached_full_analysis(hash_str: str, df: pd.DataFrame) -> Dict[str, Any]:
    if hash_str in _full_analysis_cache:
        return _full_analysis_cache[hash_str]
    metrics = calculate_metrics(df)
    risks = detect_risk(df)
    insights = generate_ai_insights(metrics, risks)
    future = analyze_future(df)
    simulation, optimization = optimize_and_explain(df)
    result = {
        "metrics": metrics,
        "risks": risks,
        "insights": insights,
        "forecast": future,
        "optimization": {"simulation": simulation, "recommendations": optimization[1] if isinstance(optimization, tuple) else optimization},
    }
    if hash_str != "empty":
        _full_analysis_cache[hash_str] = result
    return result


def full_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    h = _df_hash(df)
    return _cached_full_analysis(h, df)


def financial_advice(df: pd.DataFrame, analysis: Optional[Dict[str, Any]] = None) -> List[str]:
    if analysis is None:
        analysis = full_analysis(df)
    advice = []
    metrics = analysis["metrics"]
    forecast = analysis["forecast"]
    risks = analysis["risks"]
    category_breakdown = metrics.get("category_breakdown", {})
    if metrics["average"] > 1000:
        advice.append("Your average transaction amount is high. Review discretionary spending.")
    if forecast.get("trend") == "increasing":
        advice.append("Your spending trend is increasing. Consider setting monthly category budgets.")
    elif forecast.get("trend") == "decreasing":
        advice.append("Your spending trend is improving. Maintain your current budgeting discipline.")
    if category_breakdown:
        top_category = max(category_breakdown, key=category_breakdown.get)
        advice.append(f"You spend the most on '{top_category}'. Review whether expenses in this category can be optimized.")
    for risk in risks:
        advice.append(f"Risk detected ({risk['severity']}): {risk['message']}")
    for rec in analysis.get("optimization", {}).get("recommendations", []):
        advice.append(str(rec))
    return list(dict.fromkeys(advice))
