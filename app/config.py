# app/config.py

import os
import logging
import secrets
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

# ---------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# ---------------------------------------------------
load_dotenv()

# ---------------------------------------------------
# LOGGER
# ---------------------------------------------------
logger = logging.getLogger("financeiq.config")

if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )


# ---------------------------------------------------
# HELPERS
# ---------------------------------------------------
def parse_bool(value, default=False):
    """
    Robust boolean parser.
    Accepts: true, 1, yes, on  → True
            false, 0, no, off → False
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    true_values = {"true", "1", "yes", "on"}
    false_values = {"false", "0", "no", "off"}
    if value in true_values:
        return True
    if value in false_values:
        return False
    return default


def parse_float(value, default, minimum=None, maximum=None):
    try:
        result = float(value)
        if minimum is not None and result < minimum:
            logger.warning(
                f"Value {result} below minimum {minimum}. "
                f"Using default {default}."
            )
            return default
        if maximum is not None and result > maximum:
            logger.warning(
                f"Value {result} above maximum {maximum}. "
                f"Using default {default}."
            )
            return default
        return result
    except (TypeError, ValueError):
        logger.warning(
            f"Invalid float value '{value}'. "
            f"Using default {default}."
        )
        return default


def ensure_directory(path):
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Cannot create directory '{path}': {str(e)}")


def validate_url(url, name="URL"):
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid {name}: {url}")
    return True


def validate_database_url(url):
    if url.startswith("sqlite:///"):
        return True
    parsed = urlparse(url)
    if not parsed.scheme:
        raise ValueError("Database URL missing scheme.")
    return True


# ---------------------------------------------------
# CONFIG CLASS
# ---------------------------------------------------
class Config:

    # ------------------------------------------------
    # APP
    # ------------------------------------------------
    APP_NAME = os.getenv(
        "FINANCEIQ_APP_NAME",
        "FinanceIQ"
    )

    # Default DEBUG=True for development (no secret key required)
    DEBUG = parse_bool(
        os.getenv("FINANCEIQ_DEBUG", "True")
    )

    # ------------------------------------------------
    # SECURITY
    # ------------------------------------------------
    SECRET_KEY = os.getenv(
        "FINANCEIQ_SECRET_KEY",
        None
    )

    # ------------------------------------------------
    # DATABASE
    # ------------------------------------------------
    DATABASE_URL = os.getenv(
        "FINANCEIQ_DATABASE_URL",
        os.getenv(
            "DATABASE_URL",
            "sqlite:///finance.db"
        )
    )

    # ------------------------------------------------
    # OLLAMA
    # ------------------------------------------------
    OLLAMA_URL = os.getenv(
        "FINANCEIQ_OLLAMA_URL",
        os.getenv(
            "OLLAMA_URL",
            "http://localhost:11434/api/generate"
        )
    )

    OLLAMA_MODEL = os.getenv(
        "FINANCEIQ_OLLAMA_MODEL",
        os.getenv("OLLAMA_MODEL", "tinyllama")
    )

    # ------------------------------------------------
    # OPENAI
    # ------------------------------------------------
    OPENAI_API_KEY = os.getenv(
        "FINANCEIQ_OPENAI_API_KEY",
        os.getenv("OPENAI_API_KEY")
    )

    # ------------------------------------------------
    # RESEARCH MODE
    # ------------------------------------------------
    RESEARCH_MODE = parse_bool(
        os.getenv(
            "FINANCEIQ_RESEARCH_MODE",
            "False"
        )
    )

    EVALUATION_OUTPUT_DIR = os.getenv(
        "FINANCEIQ_EVALUATION_OUTPUT_DIR",
        "evaluation_results"
    )

    # ------------------------------------------------
    # ANOMALY DETECTION
    # ------------------------------------------------
    ANOMALY_CONTAMINATION = parse_float(
        os.getenv(
            "FINANCEIQ_ANOMALY_CONTAMINATION",
            "0.05"
        ),
        default=0.05,
        minimum=0.001,
        maximum=0.5
    )

    ZSCORE_THRESHOLD = parse_float(
        os.getenv(
            "FINANCEIQ_ZSCORE_THRESHOLD",
            "3.0"
        ),
        default=3.0,
        minimum=1.0,
        maximum=10.0
    )

    # ------------------------------------------------
    # CATEGORIZATION
    # ------------------------------------------------
    CATEGORIZATION_CONFIDENCE_THRESHOLD = parse_float(
        os.getenv(
            "FINANCEIQ_CATEGORIZATION_CONFIDENCE_THRESHOLD",
            "0.6"
        ),
        default=0.6,
        minimum=0.1,
        maximum=0.99
    )

    # ------------------------------------------------
    # CACHE
    # ------------------------------------------------
    CACHE_ENABLED = parse_bool(
        os.getenv(
            "FINANCEIQ_CACHE_ENABLED",
            "True"
        )
    )

    CACHE_TTL = int(
        parse_float(
            os.getenv(
                "FINANCEIQ_CACHE_TTL",
                "300"
            ),
            default=300,
            minimum=1
        )
    )

    # ------------------------------------------------
    # BATCH PROCESSING
    # ------------------------------------------------
    BATCH_PROCESSING_ENABLED = parse_bool(
        os.getenv(
            "FINANCEIQ_BATCH_PROCESSING_ENABLED",
            "True"
        )
    )

    # ------------------------------------------------
    # VALIDATION
    # ------------------------------------------------
    @classmethod
    def validate(cls):

        logger.info("Validating FinanceIQ configuration...")

        # ----------------------------
        # SECRET KEY
        # ----------------------------
        if not cls.SECRET_KEY:
            if cls.DEBUG:
                # Development mode: generate a random key
                cls.SECRET_KEY = secrets.token_hex(32)
                logger.warning(
                    "No SECRET_KEY set. Generated a random development key. "
                    "For production, set FINANCEIQ_SECRET_KEY in .env."
                )
            else:
                raise ValueError(
                    "SECRET_KEY is required in production. "
                    "Set FINANCEIQ_SECRET_KEY in your .env file."
                )

        # ----------------------------
        # DATABASE URL
        # ----------------------------
        validate_database_url(cls.DATABASE_URL)

        # ----------------------------
        # OLLAMA URL
        # ----------------------------
        validate_url(
            cls.OLLAMA_URL,
            "OLLAMA_URL"
        )

        # ----------------------------
        # OUTPUT DIR
        # ----------------------------
        ensure_directory(
            cls.EVALUATION_OUTPUT_DIR
        )

        # ----------------------------
        # RESEARCH MODE
        # ----------------------------
        if cls.RESEARCH_MODE:
            logger.info(
                "Research mode enabled."
            )

        # ----------------------------
        # OPENAI WARNING
        # ----------------------------
        if not cls.OPENAI_API_KEY:
            logger.info(
                "OPENAI_API_KEY not configured "
                "(OK if using Ollama only)."
            )

        logger.info(
            "Configuration validation completed successfully."
        )

    # ------------------------------------------------
    # DEBUG PRINT
    # ------------------------------------------------
    @classmethod
    def summary(cls):

        return {
            "APP_NAME": cls.APP_NAME,
            "DEBUG": cls.DEBUG,
            "DATABASE_URL": cls.DATABASE_URL,
            "OLLAMA_MODEL": cls.OLLAMA_MODEL,
            "RESEARCH_MODE": cls.RESEARCH_MODE,
            "ANOMALY_CONTAMINATION": cls.ANOMALY_CONTAMINATION,
            "ZSCORE_THRESHOLD": cls.ZSCORE_THRESHOLD,
            "CACHE_ENABLED": cls.CACHE_ENABLED,
            "CACHE_TTL": cls.CACHE_TTL
        }

    # ------------------------------------------------
    # RELOAD SUPPORT
    # ------------------------------------------------
    @classmethod
    def reload(cls):

        logger.info("Reloading configuration...")

        load_dotenv(override=True)

        cls.validate()

        logger.info("Configuration reloaded.")


# ---------------------------------------------------
# VALIDATE ON IMPORT
# ---------------------------------------------------
Config.validate()


# ---------------------------------------------------
# EXPORTS
# ---------------------------------------------------
APP_NAME = Config.APP_NAME
DEBUG = Config.DEBUG
SECRET_KEY = Config.SECRET_KEY

DATABASE_URL = Config.DATABASE_URL

OLLAMA_URL = Config.OLLAMA_URL
OLLAMA_MODEL = Config.OLLAMA_MODEL

OPENAI_API_KEY = Config.OPENAI_API_KEY

RESEARCH_MODE = Config.RESEARCH_MODE
EVALUATION_OUTPUT_DIR = Config.EVALUATION_OUTPUT_DIR

ANOMALY_CONTAMINATION = Config.ANOMALY_CONTAMINATION
ZSCORE_THRESHOLD = Config.ZSCORE_THRESHOLD

CATEGORIZATION_CONFIDENCE_THRESHOLD = (
    Config.CATEGORIZATION_CONFIDENCE_THRESHOLD
)

CACHE_ENABLED = Config.CACHE_ENABLED
CACHE_TTL = Config.CACHE_TTL

BATCH_PROCESSING_ENABLED = (
    Config.BATCH_PROCESSING_ENABLED
)