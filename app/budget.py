# app/budget.py

import os
import json
import tempfile
import threading
from decimal import Decimal, InvalidOperation
from datetime import datetime, timedelta
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

BUDGET_FILE = os.getenv("FINANCEIQ_BUDGET_FILE", "budget_store.json")
DEFAULT_PERIOD = "monthly"
VALID_PERIODS = ["daily", "weekly", "monthly", "yearly"]

_budget_lock = threading.Lock()

def _safe_decimal(value):
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError):
        return Decimal("0")

def _atomic_write_json(filepath, data):
    try:
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", delete=False, dir=directory if directory else ".", encoding="utf-8") as tmp_file:
            json.dump(data, tmp_file, indent=2)
            temp_name = tmp_file.name
        os.replace(temp_name, filepath)
        return True, "Budget data saved successfully."
    except Exception as e:
        return False, str(e)

def _load_all_budgets():
    if not os.path.exists(BUDGET_FILE):
        return {}
    try:
        with open(BUDGET_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def _save_all_budgets(data):
    with _budget_lock:
        return _atomic_write_json(BUDGET_FILE, data)

def normalize_category(category):
    if not category:
        return "general"
    return str(category).strip().lower()

def validate_budget_amount(amount):
    value = _safe_decimal(amount)
    if value <= 0:
        return False, "Budget amount must be greater than zero."
    return True, value

def set_budget(user, category, amount, period=DEFAULT_PERIOD):
    if not user:
        return False, "User is required."
    category = normalize_category(category)
    if period not in VALID_PERIODS:
        return False, f"Invalid period. Choose from: {VALID_PERIODS}"
    valid, validated_amount = validate_budget_amount(amount)
    if not valid:
        return False, validated_amount
    data = _load_all_budgets()
    if user not in data:
        data[user] = {}
    previous = data[user].get(category)
    data[user][category] = {
        "amount": float(validated_amount),
        "period": period,
        "updated_at": datetime.now().isoformat(),
        "previous_amount": previous["amount"] if isinstance(previous, dict) else None
    }
    success, message = _save_all_budgets(data)
    if not success:
        return False, f"Failed to save budget: {message}"
    return True, f"Budget set successfully.\nCategory: {category}\nAmount: {validated_amount}\nPeriod: {period}"

def get_user_budgets(user):
    data = _load_all_budgets()
    budgets = data.get(user, {})
    cleaned = {}
    for category, value in budgets.items():
        try:
            if isinstance(value, dict):
                cleaned[normalize_category(category)] = value
            else:
                cleaned[normalize_category(category)] = {"amount": float(value), "period": DEFAULT_PERIOD}
        except:
            continue
    return cleaned

def filter_budget_period(df, period=DEFAULT_PERIOD):
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    if "date" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    now = datetime.now()
    if period == "daily":
        start = now.replace(hour=0, minute=0, second=0)
    elif period == "weekly":
        start = now - timedelta(days=now.weekday())
    elif period == "yearly":
        start = datetime(now.year, 1, 1)
    else:
        start = datetime(now.year, now.month, 1)
    return df[df["date"] >= start]

def normalize_dataframe_categories(df):
    df = df.copy()
    if "category" not in df.columns:
        df["category"] = "general"
    df["category"] = df["category"].astype(str).str.lower().str.strip()
    return df

def check_budget(df, user, category=None):
    budgets = get_user_budgets(user)
    if not budgets:
        return ["No budgets set."]
    if category is not None:
        category = normalize_category(category)
        if category not in budgets:
            return [f"No budget set for {category}."]
        budgets = {category: budgets[category]}
    if df is None or df.empty:
        return ["No transaction data available."]
    df = normalize_dataframe_categories(df)
    results = []
    for cat, info in budgets.items():
        limit = _safe_decimal(info.get("amount", 0))
        period = info.get("period", DEFAULT_PERIOD)
        df_period = filter_budget_period(df, period)
        spent = Decimal("0")
        if not df_period.empty:
            matching = df_period[df_period["category"] == cat]
            if not matching.empty:
                spent = _safe_decimal(matching["amount"].sum())
        percent = (spent / limit * 100) if limit > 0 else Decimal("0")
        remaining = limit - spent
        if spent > limit:
            status = f"⚠️ {cat}: Overspent by {(spent-limit):.2f} ({percent:.1f}%)"
        elif percent >= 80:
            status = f"⚠️ {cat}: Near limit ({percent:.1f}%)"
        else:
            status = f"✅ {cat}: OK ({percent:.1f}%)"
        results.append(status)
        results.append(f"   Budget: {limit:.2f} | Spent: {spent:.2f} | Remaining: {remaining:.2f}")
    return results

def budget_summary(df, user, category=None):
    budgets = get_user_budgets(user)
    if not budgets:
        return []
    if category is not None:
        category = normalize_category(category)
        if category not in budgets:
            return []
        budgets = {category: budgets[category]}
    if df is None or df.empty:
        summary = []
        for cat, info in budgets.items():
            summary.append({
                "category": cat,
                "budget": float(info.get("amount", 0)),
                "spent": 0.0,
                "remaining": float(info.get("amount", 0)),
                "percent": 0.0,
                "period": info.get("period", DEFAULT_PERIOD)
            })
        return summary
    df = normalize_dataframe_categories(df)
    summary = []
    for cat, info in budgets.items():
        limit = _safe_decimal(info.get("amount", 0))
        period = info.get("period", DEFAULT_PERIOD)
        df_period = filter_budget_period(df, period)
        spent = Decimal("0")
        if not df_period.empty:
            matching = df_period[df_period["category"] == cat]
            if not matching.empty:
                spent = _safe_decimal(matching["amount"].sum())
        remaining = limit - spent
        percent = float((spent / limit) * 100) if limit > 0 else 0
        summary.append({
            "category": cat,
            "budget": float(limit),
            "spent": float(spent),
            "remaining": float(remaining),
            "percent": round(percent, 2),
            "period": period
        })
    return summary