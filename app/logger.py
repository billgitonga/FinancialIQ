# app/logger.py

"""
Centralized logging configuration for FinanceIQ.

Features:
- Environment-based configuration
- Rotating log files
- Console + file handlers
- Structured and detailed formatting
- Automatic log directory creation
- Configurable log levels
- Third-party log suppression
- UTC timestamps
- Thread-safe handlers
"""

import os
import sys
import logging
import logging.config
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# -------------------------------------------------------------------
# Environment Configuration
# -------------------------------------------------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_FILE = os.getenv("LOG_FILE", "financeiq.log")

LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", 5 * 1024 * 1024))  # 5 MB
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", 5))

ENABLE_CONSOLE_LOG = (
    os.getenv("ENABLE_CONSOLE_LOG", "true").lower() == "true"
)

SUPPRESS_THIRD_PARTY_LOGS = (
    os.getenv("SUPPRESS_THIRD_PARTY_LOGS", "true").lower() == "true"
)

# -------------------------------------------------------------------
# Ensure Log Directory Exists
# -------------------------------------------------------------------

log_dir_path = Path(LOG_DIR)

try:
    log_dir_path.mkdir(parents=True, exist_ok=True)
except Exception as e:
    raise RuntimeError(f"Failed to create log directory: {e}")

log_file_path = log_dir_path / LOG_FILE

# -------------------------------------------------------------------
# Log Format
# -------------------------------------------------------------------

LOG_FORMAT = (
    "%(asctime)s | "
    "%(levelname)s | "
    "%(name)s | "
    "%(filename)s:%(lineno)d | "
    "%(message)s"
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S UTC"

# -------------------------------------------------------------------
# Formatter
# -------------------------------------------------------------------

formatter = logging.Formatter(
    fmt=LOG_FORMAT,
    datefmt=DATE_FORMAT
)

# Force UTC timestamps
formatter.converter = __import__("time").gmtime

# -------------------------------------------------------------------
# File Handler (Rotating)
# -------------------------------------------------------------------

file_handler = RotatingFileHandler(
    filename=log_file_path,
    maxBytes=LOG_MAX_BYTES,
    backupCount=LOG_BACKUP_COUNT,
    encoding="utf-8"
)

file_handler.setLevel(LOG_LEVEL)
file_handler.setFormatter(formatter)

# -------------------------------------------------------------------
# Console Handler
# -------------------------------------------------------------------

console_handler: Optional[logging.StreamHandler] = None

if ENABLE_CONSOLE_LOG:
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(formatter)

# -------------------------------------------------------------------
# Root Logger Configuration
# -------------------------------------------------------------------

logger = logging.getLogger("FinanceIQ")

# Prevent duplicate handlers during reloads
if not logger.handlers:

    logger.setLevel(LOG_LEVEL)

    logger.addHandler(file_handler)

    if console_handler:
        logger.addHandler(console_handler)

    logger.propagate = False

# -------------------------------------------------------------------
# Suppress Noisy Third-Party Logs
# -------------------------------------------------------------------

if SUPPRESS_THIRD_PARTY_LOGS:

    noisy_loggers = [
        "urllib3",
        "matplotlib",
        "tensorflow",
        "PIL",
        "asyncio",
        "sentence_transformers",
        "transformers"
    ]

    for noisy_logger in noisy_loggers:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------


def get_logger(name: str = "FinanceIQ") -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name:
            Logger name/module name.

    Returns:
        Configured logger object.
    """
    return logging.getLogger(name)


def set_log_level(level: str):
    """
    Dynamically update log level.

    Args:
        level:
            DEBUG, INFO, WARNING, ERROR, CRITICAL
    """

    level = level.upper()

    valid_levels = {
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL"
    }

    if level not in valid_levels:
        raise ValueError(f"Invalid log level: {level}")

    logger.setLevel(level)

    for handler in logger.handlers:
        handler.setLevel(level)

    logger.info("Log level updated to %s", level)


def log_exception(
    exc: Exception,
    message: str = "Unhandled exception occurred"
):
    """
    Log an exception with stack trace.

    Args:
        exc:
            Exception object.

        message:
            Context message.
    """
    logger.exception("%s | %s", message, str(exc))


# -------------------------------------------------------------------
# Startup Log
# -------------------------------------------------------------------

logger.info("FinanceIQ logging initialized.")
logger.info("Log file: %s", log_file_path.resolve())
logger.info("Log level: %s", LOG_LEVEL)