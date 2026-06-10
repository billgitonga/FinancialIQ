# app/retriever.py

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "description",
    "category",
    "amount"
}


# ============================================================
# VALIDATION
# ============================================================

def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[set] = None
) -> None:

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    if df.empty:
        raise ValueError("DataFrame is empty")

    if required_columns:
        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}"
            )


# ============================================================
# SAFE STRING SERIES
# ============================================================

def safe_string_series(series: pd.Series) -> pd.Series:

    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )


# ============================================================
# KEYWORD SEARCH
# ============================================================

def keyword_search(
    df: pd.DataFrame,
    query: str,
    top_k: int = 5
) -> pd.DataFrame:

    validate_dataframe(
        df,
        {"description", "category"}
    )

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be a non-empty string")

    query = query.strip().lower()

    desc = safe_string_series(df["description"])
    cat = safe_string_series(df["category"])

    mask = (
        desc.str.contains(query, na=False, regex=False)
        |
        cat.str.contains(query, na=False, regex=False)
    )

    results = df[mask].copy()

    if results.empty:
        return results

    # Simple relevance scoring
    results["_score"] = 0

    results.loc[
        desc.str.startswith(query),
        "_score"
    ] += 3

    results.loc[
        cat.str.startswith(query),
        "_score"
    ] += 2

    results.loc[
        desc.str.contains(query, na=False, regex=False),
        "_score"
    ] += 1

    results = (
        results
        .sort_values("_score", ascending=False)
        .drop(columns=["_score"])
    )

    return results.head(top_k)


# ============================================================
# CATEGORY TRANSACTIONS
# ============================================================

def get_category_transactions(
    df: pd.DataFrame,
    category: str
) -> pd.DataFrame:

    validate_dataframe(df, {"category"})

    if not isinstance(category, str):
        raise ValueError("Category must be a string")

    category_series = safe_string_series(df["category"])

    return df[
        category_series == category.lower().strip()
    ].copy()


# ============================================================
# TOP SPENDING
# ============================================================

def top_spending(
    df: pd.DataFrame,
    top_k: int = 5,
    expenses_only: bool = True
) -> pd.DataFrame:

    validate_dataframe(df, {"amount"})

    df = df.copy()

    df["amount"] = pd.to_numeric(
        df["amount"],
        errors="coerce"
    )

    df = df.dropna(subset=["amount"])

    if expenses_only:
        df = df[df["amount"] > 0]

    return (
        df
        .sort_values(by="amount", ascending=False)
        .head(top_k)
    )


# ============================================================
# RECENT TRANSACTIONS
# ============================================================

def recent_transactions(
    df: pd.DataFrame,
    top_k: int = 5
) -> pd.DataFrame:

    validate_dataframe(df)

    if "date" not in df.columns:
        logger.warning(
            "Date column missing in recent_transactions"
        )
        return pd.DataFrame(columns=df.columns)

    temp = df.copy()

    temp["date"] = pd.to_datetime(
        temp["date"],
        errors="coerce"
    )

    temp = temp.dropna(subset=["date"])

    return (
        temp
        .sort_values(by="date", ascending=False)
        .head(top_k)
    )


# ============================================================
# FIND SIMILAR TRANSACTIONS
# ============================================================

def find_similar_transactions(
    df: pd.DataFrame,
    amount: float,
    tolerance: float = 0.10
) -> pd.DataFrame:

    validate_dataframe(df, {"amount"})

    if not isinstance(amount, (int, float)):
        raise ValueError("Amount must be numeric")

    if tolerance < 0:
        raise ValueError("Tolerance cannot be negative")

    df = df.copy()

    df["amount"] = pd.to_numeric(
        df["amount"],
        errors="coerce"
    )

    df = df.dropna(subset=["amount"])

    abs_amount = abs(amount)

    delta = max(abs_amount * tolerance, 1.0)

    lower = amount - delta
    upper = amount + delta

    return df[
        (df["amount"] >= lower)
        &
        (df["amount"] <= upper)
    ].copy()


# ============================================================
# CONTEXT RETRIEVAL
# ============================================================

def retrieve_context(
    df: pd.DataFrame,
    query: str,
    keyword_top_k: int = 3,
    spending_top_k: int = 2,
    recent_top_k: int = 2
) -> pd.DataFrame:

    validate_dataframe(df)

    frames = []

    try:
        keyword_results = keyword_search(
            df,
            query,
            top_k=keyword_top_k
        )

        if not keyword_results.empty:
            keyword_results["_source"] = "keyword"
            frames.append(keyword_results)

    except Exception as e:
        logger.warning(
            f"Keyword search failed: {str(e)}"
        )

    try:
        top_results = top_spending(
            df,
            top_k=spending_top_k
        )

        if not top_results.empty:
            top_results["_source"] = "top_spending"
            frames.append(top_results)

    except Exception as e:
        logger.warning(
            f"Top spending retrieval failed: {str(e)}"
        )

    try:
        recent = recent_transactions(
            df,
            top_k=recent_top_k
        )

        if not recent.empty:
            recent["_source"] = "recent"
            frames.append(recent)

    except Exception as e:
        logger.warning(
            f"Recent transaction retrieval failed: {str(e)}"
        )

    if not frames:
        return pd.DataFrame(columns=df.columns)

    context = pd.concat(
        frames,
        ignore_index=True
    )

    context = context.drop_duplicates()

    return context.reset_index(drop=True)


# ============================================================
# FORMAT TRANSACTIONS
# ============================================================

def format_transactions(
    df: pd.DataFrame
) -> str:

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Input must be a pandas DataFrame"
        )

    if df.empty:
        return "No relevant transactions found."

    formatted = df.copy()

    if "date" in formatted.columns:
        formatted["date"] = (
            pd.to_datetime(
                formatted["date"],
                errors="coerce"
            )
            .dt.strftime("%Y-%m-%d")
            .fillna("")
        )

    if "amount" in formatted.columns:
        formatted["amount"] = pd.to_numeric(
            formatted["amount"],
            errors="coerce"
        ).fillna(0)

        formatted["amount"] = (
            formatted["amount"]
            .map(lambda x: f"{x:.2f}")
        )

    required = [
        "date",
        "category",
        "amount",
        "description"
    ]

    for col in required:
        if col not in formatted.columns:
            formatted[col] = ""

    lines = (
        formatted[required]
        .astype(str)
        .agg(" | ".join, axis=1)
        .tolist()
    )

    return "\n".join(lines)