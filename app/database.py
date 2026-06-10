# app/database.py

import json
import logging
from contextlib import contextmanager
from decimal import Decimal
from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import QueuePool

from app.config import DATABASE_URL

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

engine = create_engine(
    DATABASE_URL,
    echo=False,
    future=True,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)


@contextmanager
def get_connection():
    with engine.begin() as conn:
        yield conn


def init_db():
    try:
        with get_connection() as conn:
            # Users table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    role TEXT DEFAULT 'cashier',
                    full_name TEXT,
                    email TEXT,
                    phone TEXT,
                    blocked INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            try:
                conn.execute(text("ALTER TABLE users ADD COLUMN blocked INTEGER DEFAULT 0"))
            except Exception:
                pass

            try:
                conn.execute(text("ALTER TABLE users ADD COLUMN full_name TEXT"))
            except Exception:
                pass

            # Transactions
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    date DATE,
                    description TEXT,
                    amount REAL,
                    category TEXT,
                    payment_method TEXT,
                    reference TEXT,
                    month TEXT,
                    status TEXT DEFAULT 'approved',
                    approved_by TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(username) REFERENCES users(username)
                )
            """))

            # Daily items
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS daily_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_name TEXT NOT NULL,
                    date DATE NOT NULL,
                    description TEXT NOT NULL,
                    amount REAL NOT NULL,
                    category TEXT,
                    item_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_name) REFERENCES users(username)
                )
            """))

            # Products
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS products (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_name TEXT NOT NULL,
                    name TEXT NOT NULL,
                    sku TEXT UNIQUE,
                    category TEXT,
                    buying_price REAL,
                    selling_price REAL,
                    current_stock REAL DEFAULT 0,
                    min_stock_level REAL DEFAULT 0,
                    unit TEXT DEFAULT 'pcs',
                    supplier_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_name) REFERENCES users(username)
                )
            """))

            # Stock movements
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS stock_movements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_name TEXT NOT NULL,
                    product_id INTEGER,
                    date DATE,
                    quantity REAL,
                    movement_type TEXT,
                    reference TEXT,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_name) REFERENCES users(username),
                    FOREIGN KEY(product_id) REFERENCES products(id)
                )
            """))

            # Customers
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS customers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_name TEXT NOT NULL,
                    name TEXT NOT NULL,
                    phone TEXT,
                    email TEXT,
                    address TEXT,
                    credit_limit REAL DEFAULT 0,
                    current_balance REAL DEFAULT 0,
                    credit_status TEXT DEFAULT 'active',
                    approved_by TEXT,
                    approved_at TIMESTAMP,
                    last_credit_review DATE,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_name) REFERENCES users(username)
                )
            """))

# Credit sales
             conn.execute(text("""
                 CREATE TABLE IF NOT EXISTS credit_sales (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     user_name TEXT NOT NULL,
                     customer_id INTEGER,
                     date DATE,
                     amount REAL,
                     description TEXT,
                     due_date DATE,
                     paid_amount REAL DEFAULT 0,
                     status TEXT DEFAULT 'pending',
                     approved_by TEXT,
                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                     FOREIGN KEY(user_name) REFERENCES users(username),
                     FOREIGN KEY(customer_id) REFERENCES customers(id)
                 )
             """))

            # Credit payments
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS credit_payments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_name TEXT NOT NULL,
                    credit_sale_id INTEGER,
                    payment_date DATE,
                    amount REAL,
                    payment_method TEXT,
                    reference TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_name) REFERENCES users(username),
                    FOREIGN KEY(credit_sale_id) REFERENCES credit_sales(id)
                )
            """))

            # Suppliers
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS suppliers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_name TEXT NOT NULL,
                    name TEXT NOT NULL,
                    phone TEXT,
                    email TEXT,
                    address TEXT,
                    payment_terms TEXT,
                    average_delivery_days INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_name) REFERENCES users(username)
                )
            """))

            # Supplier transactions
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS supplier_transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_name TEXT NOT NULL,
                    supplier_id INTEGER,
                    date DATE,
                    amount REAL,
                    transaction_type TEXT,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_name) REFERENCES users(username),
                    FOREIGN KEY(supplier_id) REFERENCES suppliers(id)
                )
            """))

            # Expense approvals
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS expense_approvals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_name TEXT NOT NULL,
                    requester TEXT,
                    amount REAL,
                    category TEXT,
                    description TEXT,
                    receipt_path TEXT,
                    status TEXT DEFAULT 'pending',
                    approved_by TEXT,
                    approved_at TIMESTAMP,
                    rejected_reason TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_name) REFERENCES users(username)
                )
            """))

            # Budgets
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS budgets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_name TEXT NOT NULL,
                    category TEXT,
                    amount REAL,
                    period TEXT,
                    year INTEGER,
                    month INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_name) REFERENCES users(username)
                )
            """))

            # Memory (chat)
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    role TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(username) REFERENCES users(username)
                )
            """))

            # User messages
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS user_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    from_user TEXT NOT NULL,
                    to_user TEXT NOT NULL,
                    subject TEXT,
                    body TEXT,
                    is_read INTEGER DEFAULT 0,
                    parent_message_id INTEGER,
                    message_group_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(from_user) REFERENCES users(username),
                    FOREIGN KEY(to_user) REFERENCES users(username),
                    FOREIGN KEY(parent_message_id) REFERENCES user_messages(id)
                )
            """))

            try:
                conn.execute(text("ALTER TABLE user_messages ADD COLUMN parent_message_id INTEGER"))
            except Exception:
                pass
            try:
                conn.execute(text("ALTER TABLE user_messages ADD COLUMN message_group_id TEXT"))
            except Exception:
                pass

            # Invoices
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS invoices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_name TEXT NOT NULL,
                    customer_id INTEGER,
                    invoice_number TEXT,
                    invoice_date DATE,
                    due_date DATE,
                    subtotal REAL DEFAULT 0,
                    tax_amount REAL DEFAULT 0,
                    total_amount REAL DEFAULT 0,
                    status TEXT DEFAULT 'draft',
                    payment_status TEXT DEFAULT 'unpaid',
                    paid_amount REAL DEFAULT 0,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_name) REFERENCES users(username)
                )
            """))

            # Cash Flow
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS cash_flow (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_name TEXT NOT NULL,
                    date DATE,
                    description TEXT,
                    amount REAL,
                    flow_type TEXT,
                    category TEXT,
                    reference TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_name) REFERENCES users(username)
                )
            """))

            # Payroll
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS payroll (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_name TEXT NOT NULL,
                    employee_name TEXT,
                    employee_id TEXT,
                    department TEXT,
                    basic_salary REAL,
                    allowances REAL DEFAULT 0,
                    deductions REAL DEFAULT 0,
                    net_pay REAL,
                    payment_date DATE,
                    payment_method TEXT DEFAULT 'bank',
                    status TEXT DEFAULT 'paid',
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_name) REFERENCES users(username)
                )
            """))

            # Tax
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS tax_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_name TEXT NOT NULL,
                    tax_type TEXT,
                    period TEXT,
                    amount REAL,
                    paid_amount REAL DEFAULT 0,
                    status TEXT DEFAULT 'pending',
                    due_date DATE,
                    filed_date DATE,
                    receipt_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_name) REFERENCES users(username)
                )
            """))
            try:
                conn.execute(text("ALTER TABLE user_messages ADD COLUMN message_group_id TEXT"))
            except Exception:
                pass

            # Trends
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS trends (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    trend_type TEXT,
                    trend_data TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(username) REFERENCES users(username)
                )
            """))

            # Anomalies
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS anomalies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    date DATE,
                    description TEXT,
                    amount REAL,
                    category TEXT,
                    anomaly_type TEXT,
                    explanation TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(username) REFERENCES users(username)
                )
            """))

            # Receipts
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS receipts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_name TEXT NOT NULL,
                    image_path TEXT,
                    extracted_text TEXT,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_name) REFERENCES users(username)
                )
            """))

            # Indexes
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_transactions_user_date ON transactions(username, date)",
                "CREATE INDEX IF NOT EXISTS idx_daily_items_user_date ON daily_items(user_name, date)",
                "CREATE INDEX IF NOT EXISTS idx_products_user_name ON products(user_name)",
                "CREATE INDEX IF NOT EXISTS idx_customers_user_name ON customers(user_name)",
                "CREATE INDEX IF NOT EXISTS idx_credit_sales_user_customer ON credit_sales(user_name, customer_id)",
                "CREATE INDEX IF NOT EXISTS idx_suppliers_user_name ON suppliers(user_name)",
                "CREATE INDEX IF NOT EXISTS idx_expense_approvals_status ON expense_approvals(status)",
            ]
            for idx in indexes:
                conn.execute(text(idx))

        logger.info("✅ Database initialised successfully")
        return True
    except SQLAlchemyError as e:
        logger.exception("❌ Database initialisation failed")
        return False


def create_user_record(username, password_hash, role="cashier", full_name=None, email=None, phone=None):
    try:
        with get_connection() as conn:
            existing = conn.execute(text("SELECT id FROM users WHERE username = :u"), {"u": username}).fetchone()
            if existing:
                return False, "User already exists"
            owner_exists = conn.execute(text("SELECT COUNT(*) FROM users WHERE role = 'owner'")).fetchone()[0] > 0
            is_active = 1 if (role == "owner" and not owner_exists) else 0
            clean_hash = str(password_hash).strip()
            conn.execute(text("""
                INSERT INTO users (username, password, role, full_name, email, phone, blocked, is_active)
                VALUES (:u, :p, :r, :fn, :e, :ph, 0, :ia)
            """), {"u": username, "p": clean_hash, "r": role, "fn": full_name, "e": email, "ph": phone, "ia": is_active})
        return True, f"User '{username}' created" + (" and pending approval" if not is_active else "")
    except Exception as e:
        return False, str(e)


def unblock_user(username):
    try:
        with get_connection() as conn:
            try:
                conn.execute(text("UPDATE users SET blocked = 0 WHERE username = :u"), {"u": username})
            except Exception:
                try:
                    conn.execute(text("ALTER TABLE users ADD COLUMN blocked INTEGER DEFAULT 0"))
                    conn.execute(text("UPDATE users SET blocked = 0 WHERE username = :u"), {"u": username})
                except Exception as e:
                    return False, str(e)
        return True, "User unblocked"
    except Exception as e:
        return False, str(e)


def is_user_active(username):
    try:
        with engine.connect() as conn:
            try:
                result = conn.execute(text("SELECT is_active FROM users WHERE username = :u"), {"u": username}).fetchone()
            except Exception:
                try:
                    conn.execute(text("ALTER TABLE users ADD COLUMN is_active INTEGER DEFAULT 1"))
                    result = conn.execute(text("SELECT is_active FROM users WHERE username = :u"), {"u": username}).fetchone()
                except Exception:
                    return True
        return bool(result[0]) if result else True
    except Exception:
        return True


def activate_user(username):
    try:
        with get_connection() as conn:
            try:
                conn.execute(text("UPDATE users SET is_active = 1 WHERE username = :u"), {"u": username})
            except Exception:
                conn.execute(text("ALTER TABLE users ADD COLUMN is_active INTEGER DEFAULT 1"))
                conn.execute(text("UPDATE users SET is_active = 1 WHERE username = :u"), {"u": username})
        return True, "User activated"
    except Exception as e:
        return False, str(e)


def reject_user(username):
    try:
        with get_connection() as conn:
            conn.execute(text("DELETE FROM users WHERE username = :u"), {"u": username})
        return True, "Signup rejected and user removed"
    except Exception as e:
        return False, str(e)


def block_user(username):
    try:
        with get_connection() as conn:
            try:
                conn.execute(text("UPDATE users SET blocked = 1 WHERE username = :u"), {"u": username})
            except Exception:
                try:
                    conn.execute(text("ALTER TABLE users ADD COLUMN blocked INTEGER DEFAULT 0"))
                    conn.execute(text("UPDATE users SET blocked = 1 WHERE username = :u"), {"u": username})
                except Exception as e:
                    return False, str(e)
        return True, "User blocked"
    except Exception as e:
        return False, str(e)


def get_pending_users():
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT username, role, full_name, email, phone, created_at FROM users WHERE is_active = 0 ORDER BY created_at")).fetchall()
        return [{"username": r[0], "role": r[1], "name": r[2], "email": r[3], "phone": r[4], "created_at": r[5]} for r in result]
    except Exception:
        try:
            with engine.connect() as conn:
                result = conn.execute(text("SELECT username, role, full_name, email, phone, created_at FROM users WHERE blocked = 0 AND is_active = 0 ORDER BY created_at")).fetchall()
            return [{"username": r[0], "role": r[1], "name": r[2], "email": r[3], "phone": r[4], "created_at": r[5]} for r in result]
        except Exception:
            return []


def get_user_role(username):
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT role FROM users WHERE username = :u"), {"u": username}).fetchone()
        return result[0] if result else "cashier"
    except Exception:
        return "cashier"


def is_user_blocked(username):
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT blocked FROM users WHERE username = :u"), {"u": username}).fetchone()
        return bool(result[0]) if result else False
    except Exception:
        return False


def get_user_password(username):
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT password FROM users WHERE username = :u"), {"u": username}).fetchone()
        return result[0] if result else None
    except Exception:
        return None


def get_user_info(username):
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT username, role, full_name, email, phone FROM users WHERE username = :u"), {"u": username}).fetchone()
        if result:
            return {"username": result[0], "role": result[1], "full_name": result[2], "email": result[3], "phone": result[4]}
        return None
    except Exception:
        return None


def get_all_users(current_user=None):
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT username, role, full_name, email, phone, blocked, is_active FROM users ORDER BY created_at")).fetchall()
        return [{"username": r[0], "role": r[1], "name": r[2], "email": r[3], "phone": r[4], "blocked": bool(r[5]), "is_active": bool(r[6])} for r in result]
    except Exception:
        try:
            with engine.connect() as conn:
                result = conn.execute(text("SELECT username, role, full_name, email, phone, blocked FROM users ORDER BY created_at")).fetchall()
            return [{"username": r[0], "role": r[1], "name": r[2], "email": r[3], "phone": r[4], "blocked": bool(r[5]), "is_active": True} for r in result]
        except Exception:
            return []


def delete_user(username):
    try:
        with get_connection() as conn:
            conn.execute(text("DELETE FROM daily_items WHERE user_name = :u"), {"u": username})
            conn.execute(text("DELETE FROM transactions WHERE username = :u"), {"u": username})
            conn.execute(text("DELETE FROM memory WHERE username = :u"), {"u": username})
            conn.execute(text("DELETE FROM users WHERE username = :u"), {"u": username})
        return True, f"User {username} deleted"
    except Exception as e:
        return False, str(e)


def get_all_users_for_messages(current_user):
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT username, full_name FROM users WHERE username != :u AND is_active = 1 AND blocked = 0"), {"u": current_user}).fetchall()
        return [{"username": r[0], "display": r[1] or r[0]} for r in result]
    except Exception:
        return []


def get_message_recipients(group_id, current_user):
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT DISTINCT to_user FROM user_messages
                WHERE message_group_id = :g AND to_user != :u
            """), {"g": group_id, "u": current_user}).fetchall()
        return [r[0] for r in rows]
    except Exception:
        return []


def reply_message(from_user, to_user, subject, body, parent_message_id, message_group_id):
    return send_message(from_user, to_user, subject, body, parent_message_id=parent_message_id, message_group_id=message_group_id)


def can_signup():
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM users WHERE role = 'owner'")).fetchone()
        return (result[0] if result else 0) == 0
    except Exception:
        return False


def save_transactions(df, username):
    if df is None or df.empty:
        return False, "Empty DataFrame"
    required_cols = {"date", "description", "amount", "category"}
    if not required_cols.issubset(df.columns):
        return False, f"Missing columns: {required_cols - set(df.columns)}"
    try:
        df = df.copy()
        df["username"] = username
        df["amount"] = df["amount"].astype(float)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        if "payment_method" not in df.columns:
            df["payment_method"] = None
        if "reference" not in df.columns:
            df["reference"] = None
        if "month" not in df.columns:
            df["month"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m")
        with get_connection() as conn:
            df.to_sql("transactions", conn, if_exists="append", index=False, method="multi")
        return True, f"{len(df)} transactions saved"
    except Exception as e:
        logger.exception("Failed to save transactions")
        return False, str(e)


def load_transactions(username):
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text("SELECT date, description, amount, category FROM transactions WHERE username = :u ORDER BY date DESC"), conn, params={"u": username})
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            if hasattr(df["date"].dt, 'tz') and df["date"].dt.tz is not None:
                df["date"] = df["date"].dt.tz_localize(None)
        return df
    except Exception:
        return pd.DataFrame()


def save_trends(username, trends_dict):
    if not trends_dict:
        return False, "No trends to save"
    try:
        with get_connection() as conn:
            for trend_type, value in trends_dict.items():
                if isinstance(value, pd.DataFrame):
                    payload = value.to_json()
                elif isinstance(value, pd.Series):
                    payload = value.to_json()
                elif isinstance(value, dict):
                    payload = json.dumps(value)
                else:
                    payload = str(value)
                conn.execute(text("INSERT INTO trends (username, trend_type, trend_data) VALUES (:u, :t, :d)"), {"u": username, "t": trend_type, "d": payload[:10000]})
        return True, "Trends saved"
    except Exception as e:
        return False, str(e)


def save_anomalies(username, anomalies_df):
    if anomalies_df is None or anomalies_df.empty:
        return False, "No anomalies to save"
    try:
        rows = []
        for _, row in anomalies_df.iterrows():
            date_val = row.get("date")
            if hasattr(date_val, 'strftime'):
                date_val = date_val.strftime("%Y-%m-%d")
            amount_val = row.get("amount", 0)
            if isinstance(amount_val, Decimal):
                amount_val = float(amount_val)
            rows.append({
                "username": username,
                "date": date_val,
                "description": str(row.get("description", ""))[:500],
                "amount": float(amount_val),
                "category": str(row.get("category", "unknown"))[:100],
                "anomaly_type": str(row.get("anomaly", "unknown"))[:100],
                "explanation": str(row.get("explanation", ""))[:1000],
            })
        with get_connection() as conn:
            batch_size = 100
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i+batch_size]
                conn.execute(text("""
                    INSERT INTO anomalies (username, date, description, amount, category, anomaly_type, explanation)
                    VALUES (:username, :date, :description, :amount, :category, :anomaly_type, :explanation)
                """), batch)
        return True, f"{len(rows)} anomalies saved"
    except Exception as e:
        return False, str(e)


def add_product(user_name, name, sku, category, buying_price, selling_price, current_stock=0, min_stock_level=0, unit="pcs", supplier_id=None):
    try:
        with get_connection() as conn:
            existing = conn.execute(text("SELECT id FROM products WHERE sku = :s AND user_name = :u"), {"s": sku, "u": user_name}).fetchone()
            if existing:
                return False, "Product with this SKU already exists"
            conn.execute(text("""
                INSERT INTO products (user_name, name, sku, category, buying_price, selling_price, current_stock, min_stock_level, unit, supplier_id)
                VALUES (:u, :n, :s, :c, :bp, :sp, :cs, :ms, :un, :si)
            """), {"u": user_name, "n": name, "s": sku, "c": category, "bp": buying_price, "sp": selling_price, "cs": current_stock, "ms": min_stock_level, "un": unit, "si": supplier_id})
        return True, "Product added"
    except Exception as e:
        return False, str(e)


def get_all_products(user_name):
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text("""
                SELECT p.*, s.name as supplier_name
                FROM products p
                LEFT JOIN suppliers s ON p.supplier_id = s.id
                WHERE p.user_name = :u
                ORDER BY p.name
            """), conn, params={"u": user_name})
        return df
    except Exception:
        return pd.DataFrame()


def update_stock(user_name, product_id, quantity, movement_type, reference="", notes="", adjust_mode="replace"):
    pid = int(product_id)
    try:
        with get_connection() as conn:
            product = conn.execute(text("SELECT current_stock, min_stock_level FROM products WHERE id = :id AND user_name = :u"), {"id": pid, "u": user_name}).fetchone()
            if not product:
                return False, "Product not found"
            current_stock = product[0]
            min_stock = product[1] if len(product) > 1 else 0
            if movement_type == "sale":
                new_stock = current_stock - quantity
            elif movement_type == "purchase":
                new_stock = current_stock + quantity
            elif movement_type == "adjustment":
                if adjust_mode == "add":
                    new_stock = current_stock + quantity
                elif adjust_mode == "subtract":
                    new_stock = current_stock - quantity
                else:
                    new_stock = quantity
            else:
                new_stock = current_stock
            conn.execute(text("UPDATE products SET current_stock = :ns WHERE id = :id AND user_name = :u"), {"ns": new_stock, "id": pid, "u": user_name})
            conn.execute(text("""
                INSERT INTO stock_movements (user_name, product_id, date, quantity, movement_type, reference, notes)
                VALUES (:u, :pid, :d, :q, :mt, :r, :n)
            """), {"u": user_name, "pid": pid, "d": datetime.now().strftime("%Y-%m-%d"), "q": quantity, "mt": movement_type, "r": reference, "n": notes})
            if new_stock <= min_stock:
                return True, f"Stock updated. WARNING: Stock below minimum level ({new_stock} <= {min_stock})"
        return True, "Stock updated"
    except Exception as e:
        return False, str(e)


def get_low_stock_products(user_name):
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text("""
                SELECT name, sku, current_stock, min_stock_level, selling_price
                FROM products
                WHERE user_name = :u AND current_stock <= min_stock_level AND current_stock > 0
                ORDER BY current_stock ASC
            """), conn, params={"u": user_name})
        return df
    except Exception:
        return pd.DataFrame()


def add_customer(user_name, name, phone, email="", address="", credit_limit=0, approved_by=None, notes=""):
    try:
        with get_connection() as conn:
            credit_status = "active" if credit_limit > 0 else "no_credit"
            conn.execute(text("""
                INSERT INTO customers (user_name, name, phone, email, address, credit_limit, credit_status, approved_by, approved_at, notes)
                VALUES (:u, :n, :p, :e, :a, :cl, :cs, :ab, :aat, :nt)
            """), {"u": user_name, "n": name, "p": phone, "e": email, "a": address, "cl": credit_limit, 
                   "cs": credit_status, "ab": approved_by, "aat": datetime.now().isoformat(), "nt": notes})
        return True, "Customer added"
    except Exception as e:
        return False, str(e)


def get_all_customers(user_name):
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text("""
                SELECT id, name, phone, email, credit_limit, current_balance, credit_status, 
                       approved_by, approved_at, last_credit_review, notes
                FROM customers WHERE user_name = :u ORDER BY name
            """), conn, params={"u": user_name})
        return df
    except Exception:
        return pd.DataFrame()


def update_customer_credit_limit(user_name, customer_id, new_credit_limit, approved_by=None, notes=""):
    """Update a customer's credit limit with tracking."""
    try:
        with get_connection() as conn:
            # Get current credit status
            current = conn.execute(text("SELECT credit_limit, current_balance FROM customers WHERE id = :cid AND user_name = :u"), 
                                  {"cid": customer_id, "u": user_name}).fetchone()
            if not current:
                return False, "Customer not found"
            
            old_limit = current[0] or 0
            current_balance = current[1] or 0
            
            # Determine new credit status
            if new_credit_limit <= 0:
                credit_status = "no_credit"
            elif current_balance > new_credit_limit:
                credit_status = "over_limit"
            else:
                credit_status = "active"
            
            conn.execute(text("""
                UPDATE customers 
                SET credit_limit = :cl, credit_status = :cs, last_credit_review = :lcr, notes = :nt
                WHERE id = :cid AND user_name = :u
            """), {"cl": new_credit_limit, "cs": credit_status, "lcr": datetime.now().strftime("%Y-%m-%d"), 
                   "nt": notes, "cid": customer_id, "u": user_name})
            
            # If approved_by is provided, also update approved_by field
            if approved_by:
                conn.execute(text("UPDATE customers SET approved_by = :ab, approved_at = :aat WHERE id = :cid AND user_name = :u"),
                           {"ab": approved_by, "aat": datetime.now().isoformat(), "cid": customer_id, "u": user_name})
            
            action = "increased" if new_credit_limit > old_limit else "decreased" if new_credit_limit < old_limit else "unchanged"
        return True, f"Credit limit {action} to KES {new_credit_limit:,.2f}. Status: {credit_status}"
    except Exception as e:
        return False, str(e)


def update_customer_credit_status(user_name, customer_id, new_status, notes=""):
    """Manually update a customer's credit status (suspend, activate, etc.)."""
    valid_statuses = ["active", "suspended", "no_credit", "over_limit", "blacklisted"]
    if new_status not in valid_statuses:
        return False, f"Invalid status. Must be one of: {valid_statuses}"
    try:
        with get_connection() as conn:
            conn.execute(text("""
                UPDATE customers SET credit_status = :cs, notes = :nt
                WHERE id = :cid AND user_name = :u
            """), {"cs": new_status, "nt": notes, "cid": customer_id, "u": user_name})
        return True, f"Credit status updated to: {new_status}"
    except Exception as e:
        return False, str(e)


def get_customer_credit_history(user_name, customer_id):
    """Get credit history for a customer including all credit sales and payments."""
    try:
        with engine.connect() as conn:
            credit_sales = pd.read_sql(text("""
                SELECT cs.id, cs.date, cs.amount, cs.description, cs.due_date, cs.paid_amount, cs.status
                FROM credit_sales cs
                WHERE cs.user_name = :u AND cs.customer_id = :cid
                ORDER BY cs.date DESC
            """), conn, params={"u": user_name, "cid": customer_id})
            
            payments = pd.read_sql(text("""
                SELECT cp.id, cp.payment_date, cp.amount, cp.payment_method, cp.reference
                FROM credit_payments cp
                WHERE cp.user_name = :u 
                  AND cp.credit_sale_id IN (SELECT id FROM credit_sales WHERE customer_id = :cid)
                ORDER BY cp.payment_date DESC
            """), conn, params={"u": user_name, "cid": customer_id})
            
        return credit_sales, payments
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


def get_credit_status_summary(user_name):
    """Get a summary of credit status across all customers."""
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text("""
                SELECT 
                    credit_status,
                    COUNT(*) as customer_count,
                    SUM(current_balance) as total_outstanding,
                    SUM(credit_limit) as total_credit_limit,
                    AVG(credit_limit) as avg_credit_limit
                FROM customers 
                WHERE user_name = :u
                GROUP BY credit_status
            """), conn, params={"u": user_name})
        return df
    except Exception:
        return pd.DataFrame()


def update_credit_statuses(user_name):
    """Automatically update credit statuses based on current balances and due dates."""
    try:
        with get_connection() as conn:
            today = datetime.now().strftime("%Y-%m-%d")
            
            # Update customers who are over their credit limit
            conn.execute(text("""
                UPDATE customers 
                SET credit_status = 'over_limit'
                WHERE user_name = :u AND current_balance > credit_limit AND credit_limit > 0
            """), {"u": user_name})
            
            # Update customers who have overdue credit sales
            conn.execute(text("""
                UPDATE customers 
                SET credit_status = 'overdue'
                WHERE user_name = :u AND id IN (
                    SELECT customer_id FROM credit_sales 
                    WHERE user_name = :u AND due_date < :today AND status = 'pending'
                ) AND credit_status != 'over_limit'
            """), {"u": user_name, "today": today})
            
            # Update customers with no balance and credit limit
            conn.execute(text("""
                UPDATE customers 
                SET credit_status = 'active'
                WHERE user_name = :u AND current_balance <= credit_limit AND credit_limit > 0
                  AND id NOT IN (
                    SELECT customer_id FROM credit_sales 
                    WHERE user_name = :u AND due_date < :today AND status = 'pending'
                  )
            """), {"u": user_name, "today": today})
            
            # Update customers with no credit limit
            conn.execute(text("""
                UPDATE customers 
                SET credit_status = 'no_credit'
                WHERE user_name = :u AND credit_limit <= 0
            """), {"u": user_name})
            
        return True, "Credit statuses updated"
    except Exception as e:
        return False, str(e)


def add_credit_sale(user_name, customer_id, amount, description, due_date):
    try:
        with get_connection() as conn:
            conn.execute(text("""
                INSERT INTO credit_sales (user_name, customer_id, date, amount, description, due_date)
                VALUES (:u, :cid, :d, :a, :desc, :dd)
            """), {"u": user_name, "cid": customer_id, "d": datetime.now().strftime("%Y-%m-%d"), "a": amount, "desc": description, "dd": due_date})
            conn.execute(text("UPDATE customers SET current_balance = current_balance + :a WHERE id = :cid"), {"a": amount, "cid": customer_id})
        return True, "Credit sale recorded"
    except Exception as e:
        return False, str(e)


def update_customer(user_name, customer_id, name, phone, email="", credit_limit=0):
    try:
        with get_connection() as conn:
            conn.execute(text("""
                UPDATE customers SET name = :n, phone = :p, email = :e, credit_limit = :cl
                WHERE id = :cid AND user_name = :u
            """), {"n": name, "p": phone, "e": email, "cl": credit_limit, "cid": customer_id, "u": user_name})
        return True, "Customer updated"
    except Exception as e:
        return False, str(e)


def delete_customer(user_name, customer_id):
    try:
        with get_connection() as conn:
            conn.execute(text("DELETE FROM invoices WHERE customer_id = :cid AND user_name = :u"), {"cid": customer_id, "u": user_name})
            conn.execute(text("DELETE FROM credit_sales WHERE customer_id = :cid AND user_name = :u"), {"cid": customer_id, "u": user_name})
            conn.execute(text("DELETE FROM customers WHERE id = :cid AND user_name = :u"), {"cid": customer_id, "u": user_name})
        return True, "Customer deleted"
    except Exception as e:
        return False, str(e)


def record_credit_payment(user_name, credit_sale_id, customer_name_or_id, amount, payment_method="cash", reference=""):
    try:
        with get_connection() as conn:
            customer_id = int(customer_name_or_id)
            customer = conn.execute(text("SELECT current_balance FROM customers WHERE id = :cid AND user_name = :u"), {"cid": customer_id, "u": user_name}).fetchone()
            if not customer:
                return False, "Customer not found"
            current_balance = customer[0] or 0
            if current_balance <= 0:
                invoice_total = conn.execute(text("""
                    SELECT COALESCE(SUM(total_amount - paid_amount), 0) FROM invoices
                    WHERE user_name = :u AND customer_id = :cid AND status IN ('sent', 'overdue')
                """), {"u": user_name, "cid": customer_id}).fetchone()[0]
                if invoice_total <= 0:
                    return False, "Customer has no outstanding debt"
                current_balance = float(invoice_total) or 0
            if amount > current_balance:
                return False, f"Payment amount KES {amount:,.2f} exceeds outstanding debt KES {current_balance:,.2f}"
            conn.execute(text("""
                INSERT INTO credit_payments (user_name, credit_sale_id, payment_date, amount, payment_method, reference)
                VALUES (:u, :cid, :d, :a, :pm, :r)
            """), {"u": user_name, "cid": credit_sale_id, "d": datetime.now().strftime("%Y-%m-%d"), "a": amount, "pm": payment_method, "r": reference})
            new_balance = current_balance - amount
            conn.execute(text("UPDATE customers SET current_balance = :nb WHERE id = :cid"), {"nb": new_balance, "cid": customer_id})
            stmt = text("UPDATE invoices SET paid_amount = paid_amount + :a, payment_status = CASE WHEN paid_amount + :a >= total_amount THEN 'paid' ELSE 'partial' END WHERE user_name = :u AND customer_id = :cid AND status IN ('sent', 'overdue') ORDER BY created_at ASC LIMIT 1")
            conn.execute(stmt, {"a": amount, "u": user_name, "cid": customer_id})
        return True, f"Payment recorded. New balance: KES {new_balance:,.2f}"
    except Exception as e:
        return False, str(e)


def get_credit_payment_history(user_name, customer_id):
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text("""
                SELECT cp.id, cp.payment_date, cp.amount, cp.payment_method, cp.reference
                FROM credit_payments cp
                WHERE cp.user_name = :u
                  AND (cp.credit_sale_id IN (SELECT id FROM credit_sales WHERE user_name = :u AND customer_id = :cid) OR cp.reference IS NOT NULL)
                ORDER BY cp.payment_date DESC
            """), conn, params={"u": user_name, "cid": customer_id})
        return df
    except Exception:
        return pd.DataFrame()


def get_debtors_list(user_name):
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text("""
                SELECT c.id, c.name, c.phone, c.current_balance, c.credit_limit,
                       COUNT(cs.id) as unpaid_invoices
                FROM customers c
                LEFT JOIN credit_sales cs ON c.id = cs.customer_id AND cs.status = 'pending'
                WHERE c.user_name = :u AND (c.current_balance > 0 OR EXISTS(
                    SELECT 1 FROM invoices i WHERE i.user_name = :u AND i.customer_id = c.id AND i.status IN ('sent', 'overdue') AND i.paid_amount < i.total_amount
                ))
                GROUP BY c.id
                ORDER BY c.current_balance DESC
            """), conn, params={"u": user_name})
        return df
    except Exception:
        return pd.DataFrame()


def add_supplier(user_name, name, phone, email="", address="", payment_terms="", avg_delivery_days=0):
    try:
        with get_connection() as conn:
            conn.execute(text("""
                INSERT INTO suppliers (user_name, name, phone, email, address, payment_terms, average_delivery_days)
                VALUES (:u, :n, :p, :e, :a, :pt, :add)
            """), {"u": user_name, "n": name, "p": phone, "e": email, "a": address, "pt": payment_terms, "add": avg_delivery_days})
        return True, "Supplier added"
    except Exception as e:
        return False, str(e)


def get_all_suppliers(user_name):
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text("SELECT id, name, phone, email, payment_terms, average_delivery_days FROM suppliers WHERE user_name = :u ORDER BY name"), conn, params={"u": user_name})
        return df
    except Exception:
        return pd.DataFrame()


def record_supplier_payment(user_name, supplier_id, amount, description):
    try:
        with get_connection() as conn:
            conn.execute(text("""
                INSERT INTO supplier_transactions (user_name, supplier_id, date, amount, transaction_type, description)
                VALUES (:u, :sid, :d, :a, 'payment', :desc)
            """), {"u": user_name, "sid": supplier_id, "d": datetime.now().strftime("%Y-%m-%d"), "a": amount, "desc": description})
        return True, "Payment recorded"
    except Exception as e:
        return False, str(e)


def get_supplier_performance(user_name):
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text("""
                SELECT s.name, s.average_delivery_days,
                       COUNT(st.id) as transaction_count,
                       SUM(CASE WHEN st.transaction_type = 'payment' THEN st.amount ELSE 0 END) as total_paid
                FROM suppliers s
                LEFT JOIN supplier_transactions st ON s.id = st.supplier_id
                WHERE s.user_name = :u
                GROUP BY s.id
                ORDER BY s.average_delivery_days ASC
            """), conn, params={"u": user_name})
        return df
    except Exception:
        return pd.DataFrame()


def submit_expense_request(user_name, requester, amount, category, description, receipt_path=""):
    try:
        with get_connection() as conn:
            conn.execute(text("""
                INSERT INTO expense_approvals (user_name, requester, amount, category, description, receipt_path, status)
                VALUES (:u, :r, :a, :c, :d, :rp, 'pending')
            """), {"u": user_name, "r": requester, "a": amount, "c": category, "d": description, "rp": receipt_path})
        return True, "Expense request submitted for approval"
    except Exception as e:
        return False, str(e)


def approve_expense(request_id, approved_by, reject_reason=None):
    try:
        with get_connection() as conn:
            if reject_reason:
                status = "rejected"
                conn.execute(text("""
                    UPDATE expense_approvals
                    SET status = :s, approved_by = :ab, rejected_reason = :rr, approved_at = :at
                    WHERE id = :id
                """), {"s": status, "ab": approved_by, "rr": reject_reason, "at": datetime.now().isoformat(), "id": request_id})
            else:
                status = "approved"
                conn.execute(text("""
                    UPDATE expense_approvals
                    SET status = :s, approved_by = :ab, approved_at = :at
                    WHERE id = :id
                """), {"s": status, "ab": approved_by, "at": datetime.now().isoformat(), "id": request_id})
                request = conn.execute(text("SELECT user_name, amount, category, description FROM expense_approvals WHERE id = :id"), {"id": request_id}).fetchone()
                if request:
                    save_daily_item(request[0], datetime.now().strftime("%Y-%m-%d"), request[3], request[1], request[2], "expense")
        return True, f"Expense {status}"
    except Exception as e:
        return False, str(e)


def get_expenses_by_status(user_name, status):
    try:
        with engine.connect() as conn:
            if status == "all":
                if user_name:
                    df = pd.read_sql(text("""
                        SELECT id, requester, amount, category, description, receipt_path, status, approved_by, approved_at, rejected_reason, created_at
                        FROM expense_approvals
                        WHERE user_name = :u
                        ORDER BY created_at DESC
                    """), conn, params={"u": user_name})
                else:
                    df = pd.read_sql(text("""
                        SELECT id, requester, amount, category, description, receipt_path, status, approved_by, approved_at, rejected_reason, created_at
                        FROM expense_approvals
                        ORDER BY created_at DESC
                    """), conn)
            else:
                if user_name:
                    df = pd.read_sql(text("""
                        SELECT id, requester, amount, category, description, receipt_path, status, approved_by, approved_at, rejected_reason, created_at
                        FROM expense_approvals
                        WHERE user_name = :u AND status = :s
                        ORDER BY created_at DESC
                    """), conn, params={"u": user_name, "s": status})
                else:
                    df = pd.read_sql(text("""
                        SELECT id, requester, amount, category, description, receipt_path, status, approved_by, approved_at, rejected_reason, created_at
                        FROM expense_approvals
                        WHERE status = :s
                        ORDER BY created_at DESC
                    """), conn, params={"s": status})
        return df
    except Exception:
        return pd.DataFrame()


def get_pending_expenses(user_name=None):
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text("""
                SELECT id, requester, amount, category, description, receipt_path, created_at
                FROM expense_approvals
                WHERE status = 'pending'
                ORDER BY created_at ASC
            """), conn)
        return df
    except Exception:
        return pd.DataFrame()


def set_budget_enhanced(user_name, category, amount, period, year, month=None):
    try:
        with get_connection() as conn:
            conn.execute(text("DELETE FROM budgets WHERE user_name = :u AND category = :c AND period = :p AND year = :y AND month = :m"), {"u": user_name, "c": category, "p": period, "y": year, "m": month or 0})
            conn.execute(text("INSERT INTO budgets (user_name, category, amount, period, year, month) VALUES (:u, :c, :a, :p, :y, :m)"), {"u": user_name, "c": category, "a": amount, "p": period, "y": year, "m": month or 0})
        return True, "Budget saved"
    except Exception as e:
        return False, str(e)


def get_budget_status(user_name, year, month):
    try:
        with engine.connect() as conn:
            budgets_df = pd.read_sql(text("SELECT category, amount FROM budgets WHERE user_name = :u AND period = 'monthly' AND year = :y AND (month = :m OR month = 0)"), conn, params={"u": user_name, "y": year, "m": month})
            spending_df = pd.read_sql(text("""
                SELECT category, SUM(amount) as spent
                FROM daily_items
                WHERE user_name = :u AND item_type = 'expense'
                AND strftime('%Y', date) = :y AND strftime('%m', date) = :m2
                GROUP BY category
            """), conn, params={"u": user_name, "y": str(year), "m2": f"{month:02d}"})
            if budgets_df.empty:
                return pd.DataFrame()
            result = budgets_df.merge(spending_df, on="category", how="left")
            result["spent"] = result["spent"].fillna(0)
            result["remaining"] = result["amount"] - result["spent"]
            result["percent"] = (result["spent"] / result["amount"] * 100).round(2)
            result["status"] = result.apply(lambda x: "⚠️ Overspent" if x["spent"] > x["amount"] else ("⚠️ Near Limit" if x["percent"] >= 80 else "✅ OK"), axis=1)
        return result
    except Exception:
        return pd.DataFrame()


def get_yoy_comparison(user_name):
    try:
        with engine.connect() as conn:
            current_year = datetime.now().year
            last_year = current_year - 1
            current_df = pd.read_sql(text("""
                SELECT strftime('%m', date) as month, SUM(amount) as total
                FROM daily_items
                WHERE user_name = :u AND strftime('%Y', date) = :y AND item_type = 'expense'
                GROUP BY month
            """), conn, params={"u": user_name, "y": str(current_year)})
            last_df = pd.read_sql(text("""
                SELECT strftime('%m', date) as month, SUM(amount) as total
                FROM daily_items
                WHERE user_name = :u AND strftime('%Y', date) = :y AND item_type = 'expense'
                GROUP BY month
            """), conn, params={"u": user_name, "y": str(last_year)})
            if current_df.empty and last_df.empty:
                return pd.DataFrame()
            result = current_df.merge(last_df, on="month", how="outer", suffixes=("_current", "_last"))
            result["total_current"] = result["total_current"].fillna(0)
            result["total_last"] = result["total_last"].fillna(0)
            result["change"] = result["total_current"] - result["total_last"]
            result["change_percent"] = (result["change"] / result["total_last"] * 100).fillna(0).round(2)
        return result
    except Exception:
        return pd.DataFrame()


def export_accountant_report(user_name, start_date, end_date):
    try:
        with engine.connect() as conn:
            transactions = pd.read_sql(text("""
                SELECT date, description, amount, category, item_type
                FROM daily_items
                WHERE user_name = :u AND date BETWEEN :s AND :e
                ORDER BY date
            """), conn, params={"u": user_name, "s": start_date, "e": end_date})
            debtors = pd.read_sql(text("""
                SELECT c.name, cs.date, cs.amount, cs.due_date, cs.paid_amount, cs.status
                FROM credit_sales cs
                JOIN customers c ON cs.customer_id = c.id
                WHERE cs.user_name = :u AND cs.date BETWEEN :s AND :e
            """), conn, params={"u": user_name, "s": start_date, "e": end_date})
            suppliers = pd.read_sql(text("""
                SELECT s.name, st.date, st.amount, st.description
                FROM supplier_transactions st
                JOIN suppliers s ON st.supplier_id = s.id
                WHERE st.user_name = :u AND st.date BETWEEN :s AND :e
            """), conn, params={"u": user_name, "s": start_date, "e": end_date})
            total_income = transactions[transactions["item_type"] == "sale"]["amount"].sum() if not transactions.empty else 0
            total_expenses = transactions[transactions["item_type"] == "expense"]["amount"].sum() if not transactions.empty else 0
            outstanding_debt = debtors[debtors["status"] == "pending"]["amount"].sum() if not debtors.empty else 0
            summary = pd.DataFrame([{
                "Report Period": f"{start_date} to {end_date}",
                "Total Income": total_income,
                "Total Expenses": total_expenses,
                "Net Profit": total_income - total_expenses,
                "Outstanding Debtors": outstanding_debt
            }])
            category_summary = transactions.groupby("category")["amount"].sum().reset_index() if not transactions.empty else pd.DataFrame()
        return {"transactions": transactions, "debtors": debtors, "suppliers": suppliers, "summary": summary, "category_breakdown": category_summary}
    except Exception as e:
        logger.exception(f"Export failed: {e}")
        return None


def save_daily_item(user_name, date_str, description, amount, category, item_type):
    try:
        amount_float = float(amount) if isinstance(amount, Decimal) else float(amount)
        with get_connection() as conn:
            conn.execute(text("""
                INSERT INTO daily_items (user_name, date, description, amount, category, item_type)
                VALUES (:u, :d, :desc, :amt, :cat, :typ)
            """), {"u": user_name, "d": date_str, "desc": description[:500], "amt": amount_float, "cat": category[:100], "typ": item_type})
        return True, "Daily item saved"
    except Exception as e:
        return False, str(e)


def get_daily_items(user_name, start_date, end_date):
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text("""
                SELECT date, description, amount, category, item_type
                FROM daily_items
                WHERE user_name = :u AND date BETWEEN :start AND :end
                ORDER BY date DESC, created_at DESC
            """), conn, params={"u": user_name, "start": start_date, "end": end_date})
        return df
    except Exception:
        return pd.DataFrame()


def save_message(username, role, message):
    try:
        with get_connection() as conn:
            conn.execute(text("INSERT INTO memory (username, role, message) VALUES (:u, :r, :m)"), {"u": username, "r": role, "m": message[:5000]})
        return True
    except Exception:
        return False


def get_memory(username, limit=50):
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT role, message, created_at FROM memory WHERE username = :u ORDER BY created_at ASC LIMIT :l"), {"u": username, "l": limit}).fetchall()
        return [{"role": r[0], "message": r[1], "timestamp": str(r[2]) if r[2] else None} for r in rows]
    except Exception:
        return []


def update_user_details(username, full_name=None, email=None, phone=None, notes=None):
    updates = []
    params = {"u": username}
    if full_name is not None:
        updates.append("full_name = :fn")
        params["fn"] = full_name
    if email is not None:
        updates.append("email = :e")
        params["e"] = email
    if phone is not None:
        updates.append("phone = :p")
        params["p"] = phone
    if notes is not None:
        updates.append("notes = :n")
        params["n"] = notes
    if not updates:
        return True, "No changes"
    try:
        with get_connection() as conn:
            try:
                conn.execute(text("ALTER TABLE users ADD COLUMN notes TEXT"))
            except Exception:
                pass
            sql = f"UPDATE users SET {', '.join(updates)} WHERE username = :u"
            conn.execute(text(sql), params)
        return True, "User updated"
    except Exception as e:
        return False, str(e)


def get_user_notes(username):
    try:
        with engine.connect() as conn:
            try:
                result = conn.execute(text("SELECT notes FROM users WHERE username = :u"), {"u": username}).fetchone()
            except Exception:
                try:
                    conn.execute(text("ALTER TABLE users ADD COLUMN notes TEXT"))
                    result = conn.execute(text("SELECT notes FROM users WHERE username = :u"), {"u": username}).fetchone()
                except Exception:
                    return ""
        return result[0] if result and result[0] else ""
    except Exception:
        return ""


import uuid as _uuid


def send_message(from_user, to_user, subject, body, parent_message_id=None, message_group_id=None):
    recipients = to_user if isinstance(to_user, list) else [to_user]
    group_id = message_group_id or _uuid.uuid4().hex
    ts = datetime.now().isoformat()
    try:
        with get_connection() as conn:
            for r in recipients:
                conn.execute(text("""
                    INSERT INTO user_messages (from_user, to_user, subject, body, created_at, parent_message_id, message_group_id)
                    VALUES (:f, :t, :s, :b, :ts, :p, :g)
                """), {"f": from_user, "t": r, "s": subject, "b": body, "ts": ts, "p": parent_message_id, "g": group_id})
        return True, f"Message sent to {len(recipients)} recipient(s)", group_id
    except Exception as e:
        return False, str(e), None


def get_inbox(username):
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text("""
                SELECT id, from_user, subject, body, is_read, created_at, message_group_id
                FROM user_messages
                WHERE to_user = :u
                ORDER BY created_at DESC
            """), conn, params={"u": username})
        return df
    except Exception:
        return pd.DataFrame()


def get_sent(username):
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text("""
                SELECT id, to_user, subject, body, is_read, created_at
                FROM user_messages
                WHERE from_user = :u
                ORDER BY created_at DESC
            """), conn, params={"u": username})
        return df
    except Exception:
        return pd.DataFrame()


def mark_message_read(message_id):
    try:
        with get_connection() as conn:
            conn.execute(text("UPDATE user_messages SET is_read = 1 WHERE id = :id"), {"id": message_id})
        return True, "Message sent"
    except Exception as e:
        return False, str(e)


def get_all_users_for_messages(current_user):
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM users WHERE role = 'owner'")).fetchone()
        return (result[0] if result else 0) == 0
    except Exception:
        return False


def create_invoice(user_name, customer_id, invoice_number, invoice_date, due_date, subtotal, tax_amount, total_amount, notes=""):
    try:
        with get_connection() as conn:
            conn.execute(text("""
                INSERT INTO invoices (user_name, customer_id, invoice_number, invoice_date, due_date, subtotal, tax_amount, total_amount, status, payment_status, notes)
                VALUES (:u, :cid, :num, :idate, :ddate, :sub, :tax, :tot, 'sent', 'unpaid', :notes)
            """), {"u": user_name, "cid": customer_id, "num": invoice_number, "idate": invoice_date, "ddate": due_date, "sub": subtotal, "tax": tax_amount, "tot": total_amount, "notes": notes})
            conn.execute(text("UPDATE customers SET current_balance = current_balance + :a WHERE id = :cid"), {"a": total_amount, "cid": customer_id})
        return True, "Invoice created"
    except Exception as e:
        return False, str(e)


def get_invoices(user_name, status=None):
    try:
        with engine.connect() as conn:
            if status:
                df = pd.read_sql(text("SELECT * FROM invoices WHERE user_name = :u AND status = :s ORDER BY invoice_date DESC"), conn, params={"u": user_name, "s": status})
            else:
                df = pd.read_sql(text("SELECT * FROM invoices WHERE user_name = :u ORDER BY invoice_date DESC"), conn, params={"u": user_name})
        return df
    except Exception:
        return pd.DataFrame()


def record_invoice_payment(invoice_id, amount, payment_method):
    try:
        with get_connection() as conn:
            invoice = conn.execute(text("SELECT user_name, customer_id, total_amount, paid_amount FROM invoices WHERE id = :id"), {"id": invoice_id}).fetchone()
            if invoice:
                customer_id = invoice[1]
                remaining = max(0, float(invoice[2]) - float(invoice[3]))
                actual_amount = min(amount, remaining) if remaining > 0 else 0
                conn.execute(text("""
                    UPDATE invoices SET paid_amount = paid_amount + :amt, payment_status = CASE WHEN paid_amount + :amt >= total_amount THEN 'paid' ELSE 'partial' END
                    WHERE id = :id
                """), {"amt": actual_amount, "id": invoice_id})
                if actual_amount > 0:
                    conn.execute(text("UPDATE customers SET current_balance = current_balance - :a WHERE id = :cid"), {"a": actual_amount, "cid": customer_id})
        return True, "Payment recorded"
    except Exception as e:
        return False, str(e)


def add_cash_flow(user_name, date, description, amount, flow_type, category, reference=""):
    try:
        with get_connection() as conn:
            conn.execute(text("""
                INSERT INTO cash_flow (user_name, date, description, amount, flow_type, category, reference)
                VALUES (:u, :d, :desc, :amt, :ft, :cat, :ref)
            """), {"u": user_name, "d": date, "desc": description, "amt": amount, "ft": flow_type, "cat": category, "ref": reference})
        return True, "Cash flow recorded"
    except Exception as e:
        return False, str(e)


def get_cash_flow(user_name, start_date, end_date):
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text("""
                SELECT date, description, amount, flow_type, category, reference
                FROM cash_flow WHERE user_name = :u AND date BETWEEN :s AND :e
                ORDER BY date DESC
            """), conn, params={"u": user_name, "s": start_date, "e": end_date})
        return df
    except Exception:
        return pd.DataFrame()


def get_cash_flow_summary(user_name, year, month):
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text("""
                SELECT flow_type, SUM(amount) as total
                FROM cash_flow WHERE user_name = :u AND strftime('%Y', date) = :y AND strftime('%m', date) = :m
                GROUP BY flow_type
            """), conn, params={"u": user_name, "y": str(year), "m": f"{month:02d}"})
        return df
    except Exception:
        return pd.DataFrame()


def add_payroll(user_name, employee_name, employee_id, department, basic_salary, allowances, deductions, net_pay, payment_date, payment_method, notes=""):
    try:
        with get_connection() as conn:
            conn.execute(text("""
                INSERT INTO payroll (user_name, employee_name, employee_id, department, basic_salary, allowances, deductions, net_pay, payment_date, payment_method, notes)
                VALUES (:u, :ename, :eid, :dept, :basic, :allow, :deduct, :net, :pdate, :pmethod, :notes)
            """), {"u": user_name, "ename": employee_name, "eid": employee_id, "dept": department, "basic": basic_salary, "allow": allowances, "deduct": deductions, "net": net_pay, "pdate": payment_date, "pmethod": payment_method, "notes": notes})
        return True, "Payroll recorded"
    except Exception as e:
        return False, str(e)


def get_payroll(user_name, year=None, month=None):
    try:
        with engine.connect() as conn:
            if year and month:
                df = pd.read_sql(text("""
                    SELECT * FROM payroll WHERE user_name = :u AND strftime('%Y', payment_date) = :y AND strftime('%m', payment_date) = :m
                    ORDER BY payment_date DESC
                """), conn, params={"u": user_name, "y": str(year), "m": f"{month:02d}"})
            else:
                df = pd.read_sql(text("SELECT * FROM payroll WHERE user_name = :u ORDER BY payment_date DESC"), conn, params={"u": user_name})
        return df
    except Exception:
        return pd.DataFrame()


def add_tax_record(user_name, tax_type, period, amount, due_date, status="pending", receipt_path=""):
    try:
        with get_connection() as conn:
            conn.execute(text("""
                INSERT INTO tax_records (user_name, tax_type, period, amount, status, due_date, receipt_path)
                VALUES (:u, :tt, :p, :amt, :s, :ddate, :rpath)
            """), {"u": user_name, "tt": tax_type, "p": period, "amt": amount, "s": status, "ddate": due_date, "rpath": receipt_path})
        return True, "Tax record added"
    except Exception as e:
        return False, str(e)


def get_tax_records(user_name, status=None):
    try:
        with engine.connect() as conn:
            if status:
                df = pd.read_sql(text("SELECT * FROM tax_records WHERE user_name = :u AND status = :s ORDER BY due_date ASC"), conn, params={"u": user_name, "s": status})
            else:
                df = pd.read_sql(text("SELECT * FROM tax_records WHERE user_name = :u ORDER BY due_date ASC"), conn, params={"u": user_name})
        return df
    except Exception:
        return pd.DataFrame()


def init_creditors_db():
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS creditors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_name TEXT NOT NULL,
                    name TEXT NOT NULL,
                    phone TEXT,
                    email TEXT,
                    description TEXT DEFAULT '',
                    amount_owed REAL DEFAULT 0,
                    due_date DATE,
                    status TEXT DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_name) REFERENCES users(username)
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS creditor_payments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_name TEXT NOT NULL,
                    creditor_id INTEGER,
                    payment_date DATE,
                    amount REAL,
                    payment_method TEXT,
                    reference TEXT,
                    notes TEXT DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_name) REFERENCES users(username),
                    FOREIGN KEY(creditor_id) REFERENCES creditors(id)
                )
            """))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_creditors_user ON creditors(user_name)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_creditor_payments_user ON creditor_payments(user_name)"))
    except Exception:
        pass


init_creditors_db()


def add_creditor(user_name, name, phone="", email="", description="", amount_owed=0, due_date=None):
    try:
        with get_connection() as conn:
            conn.execute(text("""
                INSERT INTO creditors (user_name, name, phone, email, description, amount_owed, due_date)
                VALUES (:u, :n, :p, :e, :d, :a, :dd)
            """), {"u": user_name, "n": name, "p": phone, "e": email, "d": description, "a": amount_owed, "dd": due_date})
        return True, "Creditor added"
    except Exception as e:
        return False, str(e)


def update_creditor(user_name, creditor_id, name, phone, email, description, amount_owed, due_date, status):
    try:
        with get_connection() as conn:
            conn.execute(text("""
                UPDATE creditors SET name=:n, phone=:p, email=:e, description=:d, amount_owed=:a, due_date=:dd, status=:s
                WHERE id=:cid AND user_name=:u
            """), {"n": name, "p": phone, "e": email, "d": description, "a": amount_owed, "dd": due_date, "s": status, "cid": creditor_id, "u": user_name})
        return True, "Creditor updated"
    except Exception as e:
        return False, str(e)


def delete_creditor(user_name, creditor_id):
    try:
        with get_connection() as conn:
            conn.execute(text("DELETE FROM creditors WHERE id=:cid AND user_name=:u"), {"cid": creditor_id, "u": user_name})
        return True, "Creditor deleted"
    except Exception as e:
        return False, str(e)


def get_creditors_by_user(user_name, status=None):
    try:
        with engine.connect() as conn:
            if status:
                df = pd.read_sql(text("""
                    SELECT id, name, phone, email, description, amount_owed, due_date, status, created_at
                    FROM creditors WHERE user_name=:u AND status=:s ORDER BY created_at DESC
                """), conn, params={"u": user_name, "s": status})
            else:
                df = pd.read_sql(text("""
                    SELECT id, name, phone, email, description, amount_owed, due_date, status, created_at
                    FROM creditors WHERE user_name=:u ORDER BY created_at DESC
                """), conn, params={"u": user_name})
        return df
    except Exception:
        return pd.DataFrame()


def get_total_payable(user_name):
    try:
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT COALESCE(SUM(amount_owed),0) FROM creditors WHERE user_name=:u
            """), {"u": user_name}).fetchone()
        return float(row[0]) if row else 0.0
    except Exception:
        return 0.0


def record_creditor_payment(user_name, creditor_id, amount, payment_method="cash", reference="", notes=""):
    try:
        with get_connection() as conn:
            conn.execute(text("""
                INSERT INTO creditor_payments (user_name, creditor_id, payment_date, amount, payment_method, reference, notes)
                VALUES (:u, :cid, :d, :a, :pm, :r, :n)
            """), {"u": user_name, "cid": creditor_id, "d": datetime.now().strftime("%Y-%m-%d"), "a": amount, "pm": payment_method, "r": reference, "n": notes})
            owed = conn.execute(text("SELECT amount_owed FROM creditors WHERE id=:cid AND user_name=:u"), {"cid": creditor_id, "u": user_name}).fetchone()
            if owed:
                new_owed = max(0, owed[0] - amount)
                status = "paid" if new_owed <= 0 else "partial" if new_owed < owed[0] else "pending"
                conn.execute(text("UPDATE creditors SET amount_owed=:o, status=:s WHERE id=:cid"), {"o": new_owed, "s": status, "cid": creditor_id})
        return True, "Payment recorded"
    except Exception as e:
        return False, str(e)


def get_creditor_payment_history(user_name, creditor_id):
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text("""
                SELECT id, payment_date, amount, payment_method, reference, notes, created_at
                FROM creditor_payments
                WHERE user_name=:u AND creditor_id=:cid
                ORDER BY payment_date DESC
            """), conn, params={"u": user_name, "cid": creditor_id})
        return df
    except Exception:
        return pd.DataFrame()