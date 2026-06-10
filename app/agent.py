# app/agent.py
# ============================================================
# FinanceIQ Intelligent Financial Agent
# ============================================================

import re
from typing import Optional, Dict, Any

from app.intent import classify_intent
from app.budget import (
    set_budget,
    check_budget,
    budget_summary
)

# ============================================================
# CONFIG
# ============================================================

VALID_CATEGORIES = [
    "food",
    "groceries",
    "transport",
    "shopping",
    "rent",
    "utilities",
    "health",
    "medical",
    "education",
    "entertainment",
    "fuel",
    "travel",
    "airtime",
    "internet",
    "salary",
    "investment",
    "savings",
    "business",
    "dining out",
    "subscriptions",
    "insurance",
    "general"
]

# ============================================================
# HELPERS
# ============================================================

def normalize_text(text: str) -> str:

    if not text:
        return ""

    text = text.lower().strip()

    text = re.sub(r"\s+", " ", text)

    return text

# ============================================================
# NUMBER PARSER
# ============================================================

def parse_amount(text: str) -> Optional[float]:

    if not text:
        return None

    text = text.replace(",", "")

    try:

        value = float(text)

        if value <= 0:
            return None

        return value

    except:
        return None

# ============================================================
# EXTRACT AMOUNT
# ============================================================

def extract_amount(query: str) -> Optional[float]:

    if not query:
        return None

    query = normalize_text(query)

    patterns = [

        r'(\d+(?:,\d{3})*(?:\.\d+)?)',

        r'(?:ksh|kes|\$|usd)\s*(\d+(?:,\d{3})*(?:\.\d+)?)',

        r'budget\s+(?:of\s+)?(\d+(?:,\d{3})*(?:\.\d+)?)',

        r'limit\s+(?:of\s+)?(\d+(?:,\d{3})*(?:\.\d+)?)',

        r'set\s+(?:my\s+)?budget\s+(?:to\s+)?(\d+(?:,\d{3})*(?:\.\d+)?)'
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            query,
            re.IGNORECASE
        )

        if match:

            amount = parse_amount(
                match.group(1)
            )

            if amount:
                return amount

    return None

# ============================================================
# EXTRACT CATEGORY
# ============================================================

def extract_category(query: str) -> str:

    if not query:
        return "general"

    query = normalize_text(query)

    # ========================================================
    # DIRECT CATEGORY MATCH
    # ========================================================

    sorted_categories = sorted(
        VALID_CATEGORIES,
        key=len,
        reverse=True
    )

    for category in sorted_categories:

        if category in query:
            return category

    # ========================================================
    # PATTERN MATCHING
    # ========================================================

    patterns = [

        r'for\s+([a-zA-Z\s]+?)\s+(?:of|to|\d|$)',

        r'on\s+([a-zA-Z\s]+?)\s+(?:of|to|\d|$)',

        r'budget\s+([a-zA-Z\s]+?)\s+(?:of|to|\d|$)',

        r'category\s+([a-zA-Z\s]+)',

        r'spending\s+on\s+([a-zA-Z\s]+)',

        r'allocate\s+([a-zA-Z\s]+?)\s+(?:budget|money|funds)',

        r'limit\s+([a-zA-Z\s]+?)\s+(?:to|at|\d|$)'
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            query,
            re.IGNORECASE
        )

        if match:

            category = match.group(1).strip()

            category = re.sub(
                r'\s+',
                ' ',
                category
            )

            if len(category) > 1:
                return category

    return "general"

# ============================================================
# COMBINED EXTRACTION
# ============================================================

def extract_budget_data(query: str) -> Dict[str, Any]:

    query = normalize_text(query)

    amount = extract_amount(query)

    category = extract_category(query)

    return {
        "amount": amount,
        "category": category
    }

# ============================================================
# RESPONSE HELPERS
# ============================================================

def success_response(message: str):
    # Return just the message string for chatbot compatibility
    return message

def error_response(message: str):
    return f"❌ {message}"

# ============================================================
# HANDLE SET BUDGET
# ============================================================

def handle_set_budget(username, query):

    data = extract_budget_data(query)

    amount = data["amount"]

    category = data["category"]

    if amount is None:
        return error_response("Please specify a valid budget amount.")

    if amount <= 0:
        return error_response("Budget amount must be greater than zero.")

    try:

        result = set_budget(
            username,
            category,
            amount
        )

        if isinstance(result, tuple):
            success = result[0]
            message = result[1]
            if success:
                return f"✅ Budget set successfully.\n\nCategory: {category}\nAmount: {amount:.2f}\n\n{message}"
            return error_response(f"Failed to set budget.\n\n{message}")

        elif result is True:
            return f"✅ Budget set successfully.\n\nCategory: {category}\nAmount: {amount:.2f}"

        return error_response("Failed to set budget.")

    except Exception as e:
        return error_response(f"Budget system error: {str(e)}")

# ============================================================
# HANDLE CHECK BUDGET
# ============================================================

def handle_check_budget(username, query, df=None):

    category = extract_category(query)

    try:

        result = check_budget(
            df,
            username,
            category
        )

        message = "\n".join(result) if isinstance(result, list) else str(result)

        return f"📊 Budget Status\n\nCategory: {category}\n\n{message}"

    except Exception as e:
        return error_response(f"Could not check budget:\n{str(e)}")

# ============================================================
# HANDLE BUDGET SUMMARY
# ============================================================

def handle_budget_summary(username, df=None):

    try:

        summary_data = budget_summary(
            df,
            username
        )

        if not summary_data:
            return "No budgets set."

        lines = []
        for item in summary_data:
            lines.append(
                f"{item['category']}: spent {item['spent']:.2f} / budget {item['budget']:.2f} ({item['percent']}%)"
            )

        return "📈 Budget Summary\n\n" + "\n".join(lines)

    except Exception as e:
        return error_response(f"Could not retrieve summary:\n{str(e)}")

# -----------------------------------------------------------------------------
# FALLBACK HANDLER
# -----------------------------------------------------------------------------

def fallback_response(intent, query, df=None):

    if intent == "general":
        # Provide intelligent response based on available data
        if df is not None and not df.empty:
            try:
                total = df["amount"].sum() if "amount" in df.columns else 0
                count = len(df)
                top_cat = "unknown"
                if "category" in df.columns:
                    top_cat = df["category"].mode().iloc[0] if not df["category"].mode().empty else "unknown"
                avg = total / count if count > 0 else 0
                return (
                    f"🤖 FinanceIQ Summary\n\n"
                    f"• Total spending: {total:.2f}\n"
                    f"• Transactions: {count}\n"
                    f"• Top category: {top_cat}\n"
                    f"• Average: {avg:.2f}\n\n"
                    f"Ask me about: 'forecast', 'anomalies', 'advice', or specific categories!"
                )
            except Exception:
                pass
        return (
            "I'm not fully sure what you want to do.\n\n"
            "You can ask things like:\n"
            "- Set food budget to 5000\n"
            "- Check my transport budget\n"
            "- Show my budget summary"
        )

    return f"I recognized the intent '{intent}' but no handler exists yet."

# ============================================================
# MAIN AGENT
# ============================================================

def run_agent(query, username, df=None):

    if not query:
        return error_response("Empty query received.")

    query = normalize_text(query)

    try:
        intent = classify_intent(query)
    except Exception as e:
        return error_response(f"Intent classification failed:\n{str(e)}")

    # ========================================================
    # SET BUDGET
    # ========================================================

    if intent == "set_budget":
        return handle_set_budget(username, query)

    # ========================================================
    # CHECK BUDGET
    # ========================================================

    elif intent == "check_budget":
        return handle_check_budget(username, query, df)

    # ========================================================
    # BUDGET SUMMARY
    # ========================================================

    elif intent == "budget_summary":
        return handle_budget_summary(username, df)

    # ========================================================
    # FALLBACK
    # ========================================================

    return fallback_response(intent, query, df)