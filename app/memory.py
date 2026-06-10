# app/memory.py

"""
Persistent conversation memory management for FinanceIQ.

Features:
- Safe user file handling (prevents path traversal)
- Atomic writes to avoid corruption
- Optional TTL support for old messages
- Message size validation
- Structured logging
- Thread-safe file operations
- Robust JSON validation

Author: Bill K. Gitonga (P108/1840g/20)
"""

import json
import logging
import os
import re
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from typing import Dict, List, Optional

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

MEMORY_DIR = Path(os.getenv("MEMORY_DIR", "memory_store"))
MAX_HISTORY = int(os.getenv("MAX_HISTORY", 50))
MAX_MESSAGE_LENGTH = int(os.getenv("MAX_MESSAGE_LENGTH", 5000))
DEFAULT_TTL_DAYS = int(os.getenv("MEMORY_TTL_DAYS", 90))

VALID_ROLES = {"user", "assistant", "system"}

logger = logging.getLogger("FinanceIQ.Memory")

# Thread lock for safe concurrent access
_memory_lock = RLock()


# -----------------------------------------------------------------------------
# Internal Utilities
# -----------------------------------------------------------------------------

def _ensure_memory_dir() -> None:
    """Ensure the memory directory exists."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)


def _sanitize_user(user: str) -> str:
    """
    Sanitize user identifier to prevent path traversal attacks.

    Allowed:
    - letters
    - numbers
    - underscore
    - dash

    Args:
        user: Raw user identifier

    Returns:
        Sanitized username

    Raises:
        ValueError: If username is invalid
    """
    if not isinstance(user, str) or not user.strip():
        raise ValueError("User identifier must be a non-empty string")

    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", user.strip())

    if len(sanitized) > 100:
        raise ValueError("User identifier too long")

    return sanitized


def _get_user_file(user: str) -> Path:
    """
    Get secure path to user's memory file.

    Args:
        user: User identifier

    Returns:
        Path object for user's memory file
    """
    safe_user = _sanitize_user(user)
    return MEMORY_DIR / f"{safe_user}.json"


def _validate_history(history: List[Dict]) -> List[Dict]:
    """
    Validate and clean memory history structure.

    Args:
        history: Raw loaded history

    Returns:
        Clean validated history
    """
    validated = []

    for item in history:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        message = item.get("message")
        timestamp = item.get("time")

        if (
            isinstance(role, str)
            and isinstance(message, str)
            and isinstance(timestamp, str)
            and role in VALID_ROLES
        ):
            validated.append({
                "role": role,
                "message": message[:MAX_MESSAGE_LENGTH],
                "time": timestamp
            })

    return validated[-MAX_HISTORY:]


def _filter_expired_messages(
    history: List[Dict],
    ttl_days: Optional[int] = DEFAULT_TTL_DAYS
) -> List[Dict]:
    """
    Remove messages older than TTL.

    Args:
        history: Message history
        ttl_days: Time-to-live in days

    Returns:
        Filtered history
    """
    if ttl_days is None or ttl_days <= 0:
        return history

    cutoff = datetime.now(timezone.utc) - timedelta(days=ttl_days)

    filtered = []

    for item in history:
        try:
            timestamp = datetime.fromisoformat(
                item["time"].replace("Z", "+00:00")
            )

            if timestamp >= cutoff:
                filtered.append(item)

        except Exception:
            logger.warning("Skipping malformed timestamp: %s", item)

    return filtered


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def load_memory(
    user: str,
    ttl_days: Optional[int] = DEFAULT_TTL_DAYS
) -> List[Dict]:
    """
    Load user memory safely.

    Args:
        user: User identifier
        ttl_days: Optional retention period

    Returns:
        List of conversation messages
    """
    _ensure_memory_dir()

    file_path = _get_user_file(user)

    if not file_path.exists():
        return []

    with _memory_lock:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                logger.warning(
                    "Invalid memory structure for user '%s'",
                    user
                )
                return []

            history = _validate_history(data)
            history = _filter_expired_messages(history, ttl_days)

            return history[-MAX_HISTORY:]

        except json.JSONDecodeError as e:
            logger.error(
                "Corrupted memory file for user '%s': %s",
                user,
                str(e)
            )
            return []

        except Exception as e:
            logger.exception(
                "Unexpected memory load error for user '%s': %s",
                user,
                str(e)
            )
            return []


def save_memory(user: str, history: List[Dict]) -> bool:
    """
    Save memory safely using atomic writes.

    Args:
        user: User identifier
        history: Conversation history

    Returns:
        True if successful, False otherwise
    """
    _ensure_memory_dir()

    file_path = _get_user_file(user)

    # Validate before saving
    history = _validate_history(history)
    history = history[-MAX_HISTORY:]

    with _memory_lock:
        temp_file = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                delete=False,
                dir=MEMORY_DIR,
                suffix=".tmp"
            ) as tmp:
                json.dump(history, tmp, indent=2, ensure_ascii=False)
                temp_file = tmp.name

            # Atomic replace
            os.replace(temp_file, file_path)

            logger.debug(
                "Saved %d messages for user '%s'",
                len(history),
                user
            )

            return True

        except Exception as e:
            logger.exception(
                "Memory save failed for user '%s': %s",
                user,
                str(e)
            )

            # Cleanup temp file
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception:
                    pass

            return False


def add_message(
    user: str,
    role: str,
    message: str
) -> bool:
    """
    Add a new message to user memory.

    Args:
        user: User identifier
        role: Message role (user/assistant/system)
        message: Message text

    Returns:
        True if saved successfully
    """
    if role not in VALID_ROLES:
        raise ValueError(
            f"Invalid role '{role}'. "
            f"Allowed roles: {VALID_ROLES}"
        )

    if not isinstance(message, str) or not message.strip():
        raise ValueError("Message cannot be empty")

    message = message.strip()

    if len(message) > MAX_MESSAGE_LENGTH:
        logger.warning(
            "Message truncated for user '%s' "
            "(length=%d)",
            user,
            len(message)
        )
        message = message[:MAX_MESSAGE_LENGTH]

    with _memory_lock:
        history = load_memory(user)

        history.append({
            "role": role,
            "message": message,
            "time": datetime.now(timezone.utc).isoformat()
        })

        return save_memory(user, history)


def get_context(
    user: str,
    limit: int = 5
) -> List[Dict]:
    """
    Retrieve recent conversation context.

    Args:
        user: User identifier
        limit: Number of recent messages

    Returns:
        Recent conversation messages
    """
    if limit <= 0:
        return []

    history = load_memory(user)

    return history[-limit:]


def clear_memory(user: str) -> bool:
    """
    Delete user memory file.

    Args:
        user: User identifier

    Returns:
        True if deleted successfully
    """
    file_path = _get_user_file(user)

    with _memory_lock:
        try:
            if file_path.exists():
                file_path.unlink()

                logger.info(
                    "Cleared memory for user '%s'",
                    user
                )

            return True

        except Exception as e:
            logger.exception(
                "Failed to clear memory for user '%s': %s",
                user,
                str(e)
            )

            return False


def search_memory(
    user: str,
    keyword: str,
    role: Optional[str] = None,
    limit: int = 10
) -> List[Dict]:
    """
    Search conversation memory.

    Args:
        user: User identifier
        keyword: Search term
        role: Optional role filter
        limit: Maximum results

    Returns:
        Matching memory entries
    """
    if not keyword or not isinstance(keyword, str):
        return []

    keyword = keyword.lower().strip()

    history = load_memory(user)

    results = []

    for item in history:
        try:
            if role and item["role"] != role:
                continue

            if keyword in item["message"].lower():
                results.append(item)

        except Exception:
            continue

    return results[-limit:]


def memory_stats(user: str) -> Dict:
    """
    Return memory statistics for a user.

    Args:
        user: User identifier

    Returns:
        Memory usage statistics
    """
    history = load_memory(user)

    if not history:
        return {
            "messages": 0,
            "oldest": None,
            "newest": None,
            "roles": {}
        }

    role_counts = {}

    for item in history:
        role = item["role"]
        role_counts[role] = role_counts.get(role, 0) + 1

    return {
        "messages": len(history),
        "oldest": history[0]["time"],
        "newest": history[-1]["time"],
        "roles": role_counts
    }