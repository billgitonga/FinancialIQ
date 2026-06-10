# app/intent.py

"""
Intent classification module for FinanceIQ.

Provides lightweight and transformer-based intent classification
with lazy loading, configurable thresholds, query preprocessing,
multi-intent support, and confidence scoring.
"""

from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from typing import Dict, List, Tuple, Optional

# Use numpy only if needed for ML classification
_ML_ENABLED = os.getenv("FINANCEIQ_INTENT_ML_ENABLED", "true").lower() == "true"
if _ML_ENABLED:
    import numpy as np
else:
    np = None


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_THRESHOLD = 0.45
MULTI_INTENT_THRESHOLD = 0.40
MAX_MULTI_INTENTS = 3

MODEL_NAME = "all-MiniLM-L6-v2"


# -----------------------------------------------------------------------------
# Intent Definitions
# -----------------------------------------------------------------------------

INTENT_DEFINITIONS: Dict[str, List[str]] = {
    "set_budget": [
        "set a budget for food",
        "create a spending limit",
        "allocate budget",
        "budget for transport",
        "limit my spending on entertainment",
        "set monthly grocery budget",
        "reduce entertainment expenses"
    ],

    "forecast": [
        "predict next month spending",
        "forecast future expenses",
        "what will I spend next week",
        "estimate future costs",
        "predict upcoming bills",
        "future spending trends"
    ],

    "anomaly_check": [
        "detect suspicious transactions",
        "find unusual spending",
        "any anomalies in my account",
        "fraud detection",
        "identify suspicious payments",
        "unexpected transaction"
    ],

    "advice": [
        "how can I save money",
        "give me financial advice",
        "improve my spending habits",
        "optimize my budget",
        "how do I reduce expenses",
        "tips to save money"
    ],

    "report": [
        "show my spending report",
        "generate a financial summary",
        "export transactions",
        "monthly spending report",
        "download financial report"
    ],

    "categorization_feedback": [
        "this is food",
        "no it is transport",
        "change category to bills",
        "correct category",
        "this transaction belongs to groceries",
        "update transaction category"
    ],

    "check_balance": [
        "what is my balance",
        "show account balance",
        "how much money do i have",
        "current balance"
    ],

    "savings_goal": [
        "set a savings goal",
        "track my savings",
        "save for vacation",
        "monthly savings target"
    ],

    "compare_spending": [
        "compare spending this month",
        "compare expenses",
        "month over month analysis",
        "compare categories"
    ],

    "delete_transaction": [
        "remove transaction",
        "delete this payment",
        "erase transaction",
        "cancel this entry"
    ]
}


# -----------------------------------------------------------------------------
# Lightweight Keyword Fallback
# -----------------------------------------------------------------------------

KEYWORD_MAP = {
    "budget": "set_budget",
    "forecast": "forecast",
    "predict": "forecast",
    "future": "forecast",
    "anomaly": "anomaly_check",
    "fraud": "anomaly_check",
    "suspicious": "anomaly_check",
    "advice": "advice",
    "save": "advice",
    "report": "report",
    "summary": "report",
    "export": "report",
    "balance": "check_balance",
    "goal": "savings_goal",
    "compare": "compare_spending",
    "delete": "delete_transaction",
    "remove": "delete_transaction",
    "category": "categorization_feedback"
}


# -----------------------------------------------------------------------------
# Lazy-loaded globals
# -----------------------------------------------------------------------------

_model = None
_util = None
_intent_embeddings = {}
_model_available = False


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def preprocess_query(query: str) -> str:
    """
    Clean and normalize a user query.

    Args:
        query: Raw user query

    Returns:
        Cleaned query string
    """
    if not isinstance(query, str):
        raise ValueError("Query must be a string")

    query = query.lower().strip()

    # Remove excessive whitespace
    query = re.sub(r"\s+", " ", query)

    # Remove punctuation except apostrophes
    query = re.sub(r"[^\w\s']", "", query)

    return query


def _load_model() -> None:
    """
    Lazy-load transformer model and embeddings.
    """
    global _model
    global _util
    global _intent_embeddings
    global _model_available

    if _model is not None:
        return

    if not _ML_ENABLED:
        logger.info("ML intent classification disabled via FINANCEIQ_INTENT_ML_ENABLED")
        _model_available = False
        return

    try:
        from sentence_transformers import SentenceTransformer, util

        logger.info("Loading sentence transformer model...")

        _model = SentenceTransformer(MODEL_NAME)
        _util = util

        for intent, examples in INTENT_DEFINITIONS.items():
            _intent_embeddings[intent] = _model.encode(
                examples,
                convert_to_tensor=True
            )

        _model_available = True

        logger.info("Intent model loaded successfully.")

    except Exception as e:
        logger.warning(
            f"Transformer model unavailable. "
            f"Falling back to keyword matching. Error: {e}"
        )

        _model_available = False


# -----------------------------------------------------------------------------
# Keyword-based fallback
# -----------------------------------------------------------------------------

def keyword_intent_classifier(query: str) -> Tuple[str, float]:
    """
    Lightweight keyword-based intent classification.

    Args:
        query: User query

    Returns:
        Tuple of (intent, confidence)
    """
    cleaned = preprocess_query(query)

    for keyword, intent in KEYWORD_MAP.items():
        if keyword in cleaned:
            return intent, 0.60

    return "general", 0.0


# -----------------------------------------------------------------------------
# Transformer Classification
# -----------------------------------------------------------------------------

@lru_cache(maxsize=512)
def classify_intent(
    query: str,
    threshold: float = DEFAULT_THRESHOLD
) -> str:
    """
    Classify a single best-matching intent.

    Args:
        query: User query
        threshold: Minimum similarity threshold

    Returns:
        Best intent label or 'general'
    """
    intents = classify_multiple_intents(
        query=query,
        threshold=threshold,
        max_intents=1
    )

    if not intents:
        return "general"

    return intents[0][0]


def classify_multiple_intents(
    query: str,
    threshold: float = MULTI_INTENT_THRESHOLD,
    max_intents: int = MAX_MULTI_INTENTS
) -> List[Tuple[str, float]]:
    """
    Detect one or more intents from a query.

    Args:
        query: User query
        threshold: Similarity threshold
        max_intents: Maximum intents to return

    Returns:
        List of (intent, confidence_score)
    """
    if not query or not isinstance(query, str):
        return [("general", 0.0)]

    cleaned_query = preprocess_query(query)

    # Lazy-load model (will skip if ML is disabled)
    _load_model()

    # Fallback to keyword matcher
    if not _model_available:
        fallback_intent, confidence = keyword_intent_classifier(
            cleaned_query
        )
        return [(fallback_intent, confidence)]

    # If np is None (ML disabled), still use keyword fallback
    if np is None:
        fallback_intent, confidence = keyword_intent_classifier(
            cleaned_query
        )
        return [(fallback_intent, confidence)]

    try:
        query_embedding = _model.encode(
            cleaned_query,
            convert_to_tensor=True
        )

        results = []

        for intent, emb_tensor in _intent_embeddings.items():

            scores = _util.cos_sim(
                query_embedding,
                emb_tensor
            )

            max_score = float(scores.max().item())

            if max_score >= threshold:
                results.append((intent, round(max_score, 4)))

        # Sort descending by confidence
        results.sort(key=lambda x: x[1], reverse=True)

        # Remove weak ambiguous matches
        if results:
            best_score = results[0][1]

            results = [
                item for item in results
                if item[1] >= best_score * 0.85
            ]

        if not results:
            return [("general", 0.0)]

        return results[:max_intents]

    except Exception as e:
        logger.error(f"Intent classification failed: {e}")

        fallback_intent, confidence = keyword_intent_classifier(
            cleaned_query
        )

        return [(fallback_intent, confidence)]


# -----------------------------------------------------------------------------
# Confidence Utilities
# -----------------------------------------------------------------------------

def classify_with_confidence(
    query: str,
    threshold: float = DEFAULT_THRESHOLD
) -> Dict[str, Optional[float]]:
    """
    Return best intent with confidence score.

    Args:
        query: User query
        threshold: Confidence threshold

    Returns:
        Dictionary with intent classification details
    """
    intents = classify_multiple_intents(
        query=query,
        threshold=threshold,
        max_intents=1
    )

    intent, confidence = intents[0]

    return {
        "intent": intent,
        "confidence": round(float(confidence), 4),
        "accepted": confidence >= threshold
    }


# -----------------------------------------------------------------------------
# Dynamic Intent Updates
# -----------------------------------------------------------------------------

def add_intent_examples(
    intent_name: str,
    examples: List[str]
) -> None:
    """
    Dynamically add examples to an existing intent.

    Args:
        intent_name: Intent label
        examples: Example utterances
    """
    if intent_name not in INTENT_DEFINITIONS:
        INTENT_DEFINITIONS[intent_name] = []

    INTENT_DEFINITIONS[intent_name].extend(examples)

    logger.info(
        f"Added {len(examples)} examples to intent '{intent_name}'"
    )

    # Rebuild embeddings if model loaded
    if _model_available and _model is not None:
        _intent_embeddings[intent_name] = _model.encode(
            INTENT_DEFINITIONS[intent_name],
            convert_to_tensor=True
        )


# -----------------------------------------------------------------------------
# Intent Listing
# -----------------------------------------------------------------------------

def list_supported_intents() -> List[str]:
    """
    Return all supported intent labels.

    Returns:
        List of intent names
    """
    return sorted(INTENT_DEFINITIONS.keys())


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    queries = [
        "forecast my spending next month",
        "set a budget for groceries",
        "detect suspicious activity",
        "compare my spending this month",
        "how can I save more money",
        "delete this transaction"
    ]

    for q in queries:

        result = classify_with_confidence(q)

        print("\nQuery:", q)
        print("Intent:", result["intent"])
        print("Confidence:", result["confidence"])