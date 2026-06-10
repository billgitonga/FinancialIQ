# app/auth.py

import bcrypt
import re
from typing import Tuple
from app.database import create_user_record, get_user_password, get_user_role as db_get_role, is_user_blocked, is_user_active


def check_password_strength(password: str) -> Tuple[bool, str]:
    """Check password strength requirements."""
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    if not re.search(r"[A-Z]", password):
        return False, "Include at least one uppercase letter."
    if not re.search(r"[0-9]", password):
        return False, "Include at least one digit."
    return True, ""


def create_user(
    username: str,
    password: str,
    role: str = "cashier",
    full_name: str = None,
    email: str = None,
    phone: str = None
) -> Tuple[bool, str]:
    if not username or not password:
        return False, "Username and password required"

    valid, msg = check_password_strength(password)
    if not valid:
        return False, msg

    try:
        password_bytes = password.encode("utf-8")
        hashed_bytes = bcrypt.hashpw(password_bytes, bcrypt.gensalt())
        hashed_str = hashed_bytes.decode("utf-8").strip()

        success, message = create_user_record(username, hashed_str, role, full_name, email, phone)
        if success:
            return True, f"User '{username}' created successfully! You can now login."
        else:
            return False, message
    except Exception as e:
        return False, f"Error creating user: {str(e)}"


def authenticate(username: str, password: str) -> bool:
    if not username or not password:
        return False

    if is_user_blocked(username):
        return False

    if not is_user_active(username):
        return False

    try:
        stored_hash = get_user_password(username)
        if not stored_hash:
            return False

        stored_hash = str(stored_hash).strip()
        password_bytes = password.encode("utf-8")
        stored_bytes = stored_hash.encode("utf-8")

        return bcrypt.checkpw(password_bytes, stored_bytes)
    except Exception:
        return False


def get_user_role(username: str) -> str:
    return db_get_role(username)