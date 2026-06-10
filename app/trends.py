# app/trends.py

import pandas as pd


def category_trends(df):
    return df.groupby("category")["amount"].sum().sort_values(ascending=False)


def monthly_trends(df):
    df["month"] = df["date"].dt.to_period("M").astype(str)
    monthly = df.groupby("month")["amount"].sum().reset_index()
    return monthly


def income_vs_expense(df):
    income = df[df["amount"] > 0]["amount"].sum()
    expenses = df[df["amount"] < 0]["amount"].sum()
    return {"income": income, "expenses": abs(expenses), "net": income + expenses}


def category_share(df):
    total = df["amount"].sum()
    if total == 0:
        return {}
    shares = df.groupby("category")["amount"].sum().apply(lambda x: (x / total) * 100).sort_values(ascending=False)
    return shares.to_dict()


def growth_trend(df):
    monthly = monthly_trends(df)
    monthly["growth"] = monthly["amount"].pct_change() * 100
    return monthly


def analyze_trends(df):
    return {
        "category": category_trends(df),
        "monthly": monthly_trends(df),
        "income_vs_expense": income_vs_expense(df),
        "category_share": category_share(df),
        "growth": growth_trend(df)
    }