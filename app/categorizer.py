# app/categorizer.py
# ============================================================
# FinanceIQ Intelligent Transaction Categorizer (SME Optimized)
# ============================================================

import os
import re
import json
import time
import queue
import joblib
import threading
import warnings

import numpy as np
import pandas as pd

from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

MODELS_DIR = os.getenv(
    "FINANCEIQ_MODEL_DIR",
    "models"
)

MODEL_PATH = os.path.join(
    MODELS_DIR,
    "category_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    MODELS_DIR,
    "vectorizer.pkl"
)

METADATA_PATH = os.path.join(
    MODELS_DIR,
    "category_metadata.json"
)

FEEDBACK_FILE = os.path.join(
    MODELS_DIR,
    "feedback.csv"
)

MODEL_VERSION = "2.0"

MIN_FEEDBACK_FOR_TRAINING = 30

RETRAIN_BATCH_SIZE = 25

DEFAULT_CONFIDENCE_THRESHOLD = 0.55

ENABLE_ASYNC_RETRAIN = True

# SME Optimization: Disable ML categorizer if requested
ENABLE_ML = os.getenv("FINANCEIQ_ML_CATEGORIZER_ENABLED", "true").lower() == "true"

# ============================================================
# THREAD SAFETY
# ============================================================

_model_lock = threading.RLock()
_feedback_lock = threading.RLock()

# ============================================================
# GLOBAL MODEL
# ============================================================

model_global = None
vectorizer_global = None
metadata_global = {}

# ============================================================
# BACKGROUND EXECUTOR
# ============================================================

executor = ThreadPoolExecutor(max_workers=1)

# ============================================================
# EXPANDED CATEGORY RULES (SME focused)
# ============================================================

DEFAULT_CATEGORY_RULES = {

    "food": [
        "restaurant", "cafe", "coffee", "pizza", "burger", "lunch", "breakfast",
        "dinner", "kfc", "ubereats", "glovo", "swiggy", "snack", "chips", "milk",
        "bread", "food", "eat", "meal", "groceries", "grocery", "supermarket",
        "vegetables", "fruits", "meat", "chicken", "fish", "rice", "cooking",
        "kitchen", "dining", "takeout", "takeaway", "delivery", "resto", "tea"
    ],

    "transport": [
        "uber", "bolt", "taxi", "fuel", "petrol", "diesel", "bus", "matatu",
        "train", "parking", "fare", "transport", "travel", "commute", "boda",
        "motorbike", "car", "vehicle", "maintenance", "repair", "tyre", "oil",
        "toll", "highway", "gas", "station", "shell", "total"
    ],

    "shopping": [
        "mall", "amazon", "shop", "store", "supermarket", "clothes", "shoes",
        "electronics", "jumia", "shopping", "retail", "walmart", "target",
        "fashion", "apparel", "accessories", "gadget", "phone", "laptop",
        "computer", "tv", "television", "furniture", "home", "decor"
    ],

    "bills": [
        "electricity", "water", "internet", "wifi", "rent", "insurance",
        "airtime", "phone", "subscription", "bill", "utility", "gas",
        "trash", "sewer", "maintenance", "hoa", "mortgage", "loan"
    ],

    "health": [
        "hospital", "clinic", "doctor", "medicine", "pharmacy", "chemist",
        "health", "medical", "dentist", "optical", "eye", "prescription",
        "drugs", "treatment", "checkup", "vaccination", "insurance"
    ],

    "education": [
        "school", "university", "college", "course", "tuition", "exam",
        "book", "education", "training", "workshop", "seminar", "class",
        "online course", "certification", "degree", "student"
    ],

    "entertainment": [
        "movie", "spotify", "netflix", "concert", "game", "party", "cinema",
        "entertainment", "music", "video", "streaming", "hulu", "disney",
        "sports", "theatre", "theater", "comedy", "show", "event"
    ],

    "salary": [
        "salary", "payroll", "income", "deposit", "payment", "wage",
        "earning", "revenue", "commission", "bonus", "dividend", "interest"
    ],

    "fuel": [
        "fuel", "petrol", "diesel", "gasoline", "gas station", "shell",
        "total", "bp", "caltex", "filling station", "refuel"
    ],

    "rent": [
        "rent", "lease", "tenancy", "apartment", "housing", "accommodation"
    ],

    "utilities": [
        "electricity", "water", "power", "sewer", "trash", "garbage",
        "recycling", "utility", "kplc", "nairobi water", "token"
    ],

    "airtime": [
        "airtime", "safaricom", "airtel", "telkom", "mobile", "prepaid",
        "postpaid", "vodafone", "orange", "mtn", "etisalat"
    ],

    "travel": [
        "travel", "hotel", "flight", "airline", "airport", "lodge",
        "accommodation", "vacation", "holiday", "tour", "safari", "booking"
    ],

    "investment": [
        "investment", "stock", "bond", "mutual fund", "etf", "crypto",
        "bitcoin", "forex", "trading", "dividend", "capital"
    ],

    "savings": [
        "savings", "deposit", "account", "bank", "sacco", "cooperative"
    ],

    "business": [
        "business", "supplies", "office", "stationery", "printer", "paper",
        "equipment", "marketing", "advertising", "software", "subscription"
    ],

    "dining out": [
        "restaurant", "cafe", "coffee shop", "bar", "pub", "eatery",
        "dinner", "lunch", "breakfast", "brunch", "buffet", "tea"
    ],

    "subscriptions": [
        "subscription", "netflix", "spotify", "amazon prime", "disney",
        "hulu", "hbo", "apple music", "youtube", "premium"
    ],

    "insurance": [
        "insurance", "premium", "cover", "policy", "aia", "allianz",
        "jubilee", "britam", "cia", "health insurance", "car insurance"
    ],

    "other": [
        "misc", "other", "unknown", "general"
    ]
}

# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower().strip()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()

# ============================================================
# LOAD METADATA
# ============================================================

def load_metadata():

    if os.path.exists(METADATA_PATH):

        try:

            with open(
                METADATA_PATH,
                "r"
            ) as f:

                return json.load(f)

        except:
            pass

    return {

        "version": MODEL_VERSION,

        "trained_at": None,

        "categories": []
    }

# ============================================================
# SAVE METADATA
# ============================================================

def save_metadata(data):

    os.makedirs(
        MODELS_DIR,
        exist_ok=True
    )

    with open(
        METADATA_PATH,
        "w"
    ) as f:

        json.dump(
            data,
            f,
            indent=2
        )

# ============================================================
# LOAD MODEL
# ============================================================

def _load_global_model():

    global model_global
    global vectorizer_global
    global metadata_global

    with _model_lock:

        if (
            os.path.exists(MODEL_PATH)
            and
            os.path.exists(VECTORIZER_PATH)
        ):

            try:

                model_global = joblib.load(
                    MODEL_PATH
                )

                vectorizer_global = joblib.load(
                    VECTORIZER_PATH
                )

                metadata_global = load_metadata()

                print(
                    "✅ Categorizer model loaded"
                )

                return True

            except Exception as e:

                print(
                    "⚠️ Failed to load categorizer:",
                    str(e)
                )

    return False

# ============================================================
# INITIALIZE
# ============================================================

_load_global_model()

# ============================================================
# LOAD FEEDBACK DATA
# ============================================================

def load_feedback_data():

    if not os.path.exists(FEEDBACK_FILE):

        return pd.DataFrame(
            columns=[
                "description",
                "amount",
                "date",
                "category"
            ]
        )

    try:

        df = pd.read_csv(FEEDBACK_FILE)

        return df

    except Exception as e:

        print(
            "⚠️ Failed to load feedback:",
            str(e)
        )

        return pd.DataFrame(
            columns=[
                "description",
                "amount",
                "date",
                "category"
            ]
        )

# ============================================================
# VALID CATEGORY
# ============================================================

def valid_category(category):

    category = normalize_text(category)

    allowed = set(
        DEFAULT_CATEGORY_RULES.keys()
    )

    allowed.add("other")

    return category in allowed

# ============================================================
# SAVE FEEDBACK
# ============================================================

def save_feedback(
    description,
    category,
    amount=None,
    date=None
):

    if not valid_category(category):

        return False, "Invalid category"

    row = {

        "description":
            normalize_text(description),

        "amount":
            amount if amount is not None else 0,

        "date":
            str(date) if date else "",

        "category":
            normalize_text(category)
    }

    with _feedback_lock:

        try:

            df = load_feedback_data()

            df = pd.concat(
                [
                    df,
                    pd.DataFrame([row])
                ],
                ignore_index=True
            )

            os.makedirs(
                MODELS_DIR,
                exist_ok=True
            )

            df.to_csv(
                FEEDBACK_FILE,
                index=False
            )

            # Async retraining
            if (
                len(df)
                >=
                MIN_FEEDBACK_FOR_TRAINING
                and
                len(df)
                %
                RETRAIN_BATCH_SIZE
                ==
                0
            ):

                if ENABLE_ASYNC_RETRAIN:

                    executor.submit(
                        retrain_model
                    )

                else:

                    retrain_model()

            return True, "Feedback saved"

        except Exception as e:

            return False, str(e)

# ============================================================
# AMOUNT BUCKET
# ============================================================

def categorize_amount(amount):

    try:

        amount = float(amount)

    except:
        amount = 0

    if amount < 0:

        return "refund"

    elif amount == 0:

        return "zero"

    elif amount < 50:

        return "small"

    elif amount < 200:

        return "medium"

    elif amount < 1000:

        return "large"

    else:

        return "huge"

# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_feature_string(
    description,
    amount=None,
    date=None
):

    desc = normalize_text(description)

    parts = [desc]

    if amount is not None:

        parts.append(
            categorize_amount(amount)
        )

    try:

        if date is not None:

            dt = pd.to_datetime(date)

            parts.append(
                dt.day_name().lower()
            )

            parts.append(
                f"month_{dt.month}"
            )

    except:
        pass

    return " ".join(parts)

# ============================================================
# RULE-BASED CATEGORY (expanded and improved)
# ============================================================

@lru_cache(maxsize=5000)
def rule_based_category(description):

    desc = normalize_text(description)

    # First try exact word matching
    words = desc.split()

    for category, keywords in DEFAULT_CATEGORY_RULES.items():
        for keyword in keywords:
            # Check if keyword is in the description
            if keyword in desc:
                return category
            # Also check individual words
            for word in words:
                if keyword == word or keyword in word:
                    return category

    return "other"

# ============================================================
# TRAIN MODEL
# ============================================================

def retrain_model():

    global model_global
    global vectorizer_global

    with _model_lock:

        try:

            df = load_feedback_data()

            if len(df) < MIN_FEEDBACK_FOR_TRAINING:

                return False

            df = df.copy()

            df["description"] = df[
                "description"
            ].fillna("").apply(
                normalize_text
            )

            df["features"] = df.apply(

                lambda row:
                build_feature_string(
                    row["description"],
                    row.get("amount"),
                    row.get("date")
                ),

                axis=1
            )

            X_text = df["features"]

            y = df["category"]

            vectorizer = TfidfVectorizer(

                max_features=5000,

                ngram_range=(1, 2),

                stop_words="english"
            )

            X = vectorizer.fit_transform(
                X_text
            )

            base_model = SGDClassifier(

                loss="log_loss",

                random_state=42,

                max_iter=1000
            )

            model = CalibratedClassifierCV(
                base_model
            )

            model.fit(X, y)

            os.makedirs(
                MODELS_DIR,
                exist_ok=True
            )

            joblib.dump(
                model,
                MODEL_PATH
            )

            joblib.dump(
                vectorizer,
                VECTORIZER_PATH
            )

            metadata = {

                "version":
                    MODEL_VERSION,

                "trained_at":
                    time.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                "categories":
                    sorted(
                        list(
                            set(y)
                        )
                    ),

                "samples":
                    int(len(df))
            }

            save_metadata(metadata)

            model_global = model

            vectorizer_global = vectorizer

            print(
                "✅ Categorizer retrained"
            )

            return True

        except Exception as e:

            print(
                "❌ Retraining failed:",
                str(e)
            )

            return False

# ============================================================
# ML CATEGORY
# ============================================================

@lru_cache(maxsize=10000)
def ml_category_cached(
    description,
    amount_bucket,
    weekday
):

    global model_global
    global vectorizer_global

    if (
        model_global is None
        or
        vectorizer_global is None
    ):

        return None

    try:

        feature_str = (
            f"{normalize_text(description)} "
            f"{amount_bucket} "
            f"{weekday}"
        )

        X = vectorizer_global.transform(
            [feature_str]
        )

        probs = model_global.predict_proba(X)[0]

        best_idx = np.argmax(probs)

        confidence = probs[best_idx]

        prediction = model_global.classes_[
            best_idx
        ]

        if confidence < DEFAULT_CONFIDENCE_THRESHOLD:

            return None

        return prediction

    except Exception as e:

        print(
            "⚠️ ML prediction failed:",
            str(e)
        )

        return None

# ============================================================
# MAIN ML CATEGORY
# ============================================================

def ml_category(
    description,
    amount=None,
    date=None
):

    amount_bucket = categorize_amount(
        amount or 0
    )

    weekday = ""

    try:

        if date is not None:

            weekday = pd.to_datetime(
                date
            ).day_name().lower()

    except:
        pass

    return ml_category_cached(
        normalize_text(description),
        amount_bucket,
        weekday
    )

# ============================================================
# SINGLE ITEM CATEGORY
# ============================================================

def categorize_item_description(
    description,
    amount=None,
    date=None
):
    # First try rule-based (always runs, gives reasonable defaults)
    rule_result = rule_based_category(description)
    
    # If ML is enabled and rule-based returned "other", try ML
    if ENABLE_ML and rule_result == "other":
        pred = ml_category(
            description,
            amount,
            date
        )
        if pred and pred != "other":
            return pred
    
    return rule_result

# ============================================================
# BULK TRANSACTION CATEGORIZATION
# ============================================================

def categorize_transactions(df):

    if df.empty:
        return df

    df = df.copy()

    if "description" not in df.columns:

        df["description"] = ""

    if "amount" not in df.columns:

        df["amount"] = 0

    categories = df.apply(

        lambda row:
        categorize_item_description(
            row.get("description"),
            row.get("amount"),
            row.get("date")
        ),

        axis=1
    )

    df["category"] = categories

    return df

# ============================================================
# ADD CUSTOM RULE
# ============================================================

def add_rule(
    category,
    keyword
):

    category = normalize_text(category)

    keyword = normalize_text(keyword)

    if category not in DEFAULT_CATEGORY_RULES:

        DEFAULT_CATEGORY_RULES[category] = []

    if keyword not in DEFAULT_CATEGORY_RULES[category]:

        DEFAULT_CATEGORY_RULES[category].append(
            keyword
        )

    return True

# ============================================================
# GET AVAILABLE CATEGORIES
# ============================================================

def get_categories():

    return sorted(
        list(
            DEFAULT_CATEGORY_RULES.keys()
        )
    )

# ============================================================
# MODEL STATUS
# ============================================================

def model_status():

    return {

        "model_loaded":
            model_global is not None,

        "vectorizer_loaded":
            vectorizer_global is not None,

        "model_version":
            metadata_global.get(
                "version"
            ),

        "trained_at":
            metadata_global.get(
                "trained_at"
            ),

        "categories":
            metadata_global.get(
                "categories",
                []
            )
    }
