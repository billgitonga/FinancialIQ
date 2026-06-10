# dashboard.py

import os
import time
from datetime import date, timedelta, datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from threading import Timer

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from app.pipeline import run_pipeline
from app.auth import authenticate, create_user, is_user_blocked
from app.database import (
    init_db,
    get_all_products,
    update_stock,
    save_daily_item,
    get_daily_items,
    add_customer,
    update_customer,
    delete_customer,
    get_all_customers,
    get_debtors_list,
    add_supplier,
    get_all_suppliers,
    get_supplier_performance,
    submit_expense_request,
    get_expenses_by_status,
    get_pending_expenses,
    approve_expense,
    set_budget_enhanced,
    get_budget_status,
    get_all_users,
    activate_user,
    reject_user,
    block_user,
    unblock_user,
    delete_user,
    get_user_notes,
    update_user_details,
    is_user_active,
    get_user_role,
    mark_message_read,
    get_inbox,
    get_sent,
    send_message,
    get_all_users_for_messages,
    get_message_recipients,
    reply_message,
    load_transactions,
    get_low_stock_products,
    get_yoy_comparison,
    export_accountant_report,
    get_pending_users,
    can_signup,
    create_invoice,
    get_invoices,
    record_invoice_payment,
    record_credit_payment,
    get_credit_payment_history,
    add_credit_sale,
    add_product,
    add_cash_flow,
    get_cash_flow,
    get_cash_flow_summary,
    add_payroll,
    get_payroll,
    add_tax_record,
    get_tax_records,
    add_creditor,
    get_creditors_by_user,
    get_total_payable,
    record_creditor_payment,
    get_creditor_payment_history,
    update_customer_credit_limit,
    update_customer_credit_status,
    get_customer_credit_history,
    get_credit_status_summary,
    update_credit_statuses,
)
from app.health import calculate_health_score, health_insights
from app.predictor import predict_next_spending
from app.chatbot import generate_response
from app.reporting import ReportGenerator
from app.categorizer import categorize_item_description, get_categories
from app.ocr import extract_receipt, process_receipt_lines
from app.anomaly import detect_anomalies

# Initialize database
init_db()

st.set_page_config(page_title="FinanceIQ - Complete SME Manager", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@media (max-width: 768px) {
    .stMetric { min-width: 120px; }
    .stButton > button { font-size: 0.8rem; padding: 0.4rem; }
    .stTabs [data-baseweb="tab"] { font-size: 0.75rem; padding: 0.5rem; }
    .stDataFrame { overflow-x: auto; }
}
.stButton > button { width: 100%; }
</style>
""", unsafe_allow_html=True)

def _notify(key, title, message):
    st.session_state.setdefault("notifications", [])
    st.session_state["notifications"] = [n for n in st.session_state["notifications"] if n.get("key") != key]
    st.session_state["notifications"].append({"key": key, "title": title, "message": message, "ts": datetime.now().strftime("%H:%M")})


def _dismiss_notification(key):
    st.session_state.setdefault("notifications", [])
    st.session_state["notifications"] = [n for n in st.session_state["notifications"] if n.get("key") != key]


# Session state defaults
DEFAULT_SESSION_STATE = {
    "logged_in": False,
    "user": None,
    "user_role": "cashier",
    "csv_data": None,
    "file_uploaded": False,
    "chat_history": [],
    "chat_open": False,
    "messages_open": False,
    "selected_chat_user": None,
    "messaging_refresh": False,
    "feedback_mode": False,
    "research_mode": False,
    "streaming": False,
    "last_bot_msg": "",
    "refresh_trigger": False,
    "selected_product": None,
    "selected_customer": None,
    "selected_supplier": None,
    "show_stock_sale": False,
    "notifications": [],
    "reset_entry_form": False,
    "reset_inventory_form": False,
    "reset_customer_form": False,
    "reset_supplier_form": False,
    "reset_invoice_form": False,
    "reset_cashflow_form": False,
    "reset_payroll_form": False,
    "reset_tax_form": False,
    "reset_adjust_form": False,
}
for key, value in DEFAULT_SESSION_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value

if "report_generator" not in st.session_state:
    st.session_state.report_generator = ReportGenerator()


def _render_notifications():
    st.session_state.setdefault("notifications", [])
    if not st.session_state["notifications"]:
        return

    for n in list(st.session_state["notifications"]):
        with st.container(border=True):
            left, right = st.columns([5, 1])
            with left:
                st.markdown(f"**{n['title']}**")
                st.caption(n["message"])
            with right:
                if st.button("✕", key=f"dismiss_{n['key']}"):
                    _dismiss_notification(n["key"])
                    st.rerun()


def _apply_form_resets():
    st.session_state.setdefault("reset_entry_form", False)
    st.session_state.setdefault("reset_inventory_form", False)
    st.session_state.setdefault("reset_customer_form", False)
    st.session_state.setdefault("reset_supplier_form", False)
    st.session_state.setdefault("reset_invoice_form", False)
    st.session_state.setdefault("reset_cashflow_form", False)
    st.session_state.setdefault("reset_payroll_form", False)
    st.session_state.setdefault("reset_tax_form", False)
    st.session_state.setdefault("reset_adjust_form", False)

    if st.session_state.get("reset_entry_form"):
        st.session_state.entry_desc = ""
        st.session_state.entry_amount = 0.0
        st.session_state.manual_total = 0.0
        st.session_state.ocr_correct_text = ""
        st.session_state.reset_entry_form = False

    if st.session_state.get("reset_inventory_form"):
        for k in ["prod_name", "prod_sku", "prod_unit", "buying_price", "selling_price", "initial_stock", "min_stock"]:
            st.session_state.pop(k, None)
        st.session_state.reset_inventory_form = False

    if st.session_state.get("reset_adjust_form"):
        for k in ["adjust_product", "adj_mode", "adjust_qty"]:
            st.session_state.pop(k, None)
        st.session_state.reset_adjust_form = False

    if st.session_state.get("reset_customer_form"):
        for k in ["cust_name", "cust_phone", "cust_email", "cust_credit_limit"]:
            st.session_state.pop(k, None)
        st.session_state.reset_customer_form = False

    if st.session_state.get("reset_supplier_form"):
        for k in ["sup_name", "sup_phone", "sup_email", "sup_avg_delivery"]:
            st.session_state.pop(k, None)
        st.session_state.reset_supplier_form = False

    if st.session_state.get("reset_invoice_form"):
        for k in ["inv_num", "inv_subtotal", "inv_tax_rate", "inv_notes"]:
            st.session_state.pop(k, None)
        st.session_state.reset_invoice_form = False

    if st.session_state.get("reset_cashflow_form"):
        for k in ["cf_desc", "cf_amount", "cf_ref"]:
            st.session_state.pop(k, None)
        st.session_state.reset_cashflow_form = False

    if st.session_state.get("reset_payroll_form"):
        for k in ["emp_name", "emp_id", "emp_basic", "emp_allow", "emp_deduct"]:
            st.session_state.pop(k, None)
        st.session_state.reset_payroll_form = False

    if st.session_state.get("reset_tax_form"):
        for k in ["tax_period", "tax_amount", "tax_receipt"]:
            st.session_state.pop(k, None)
        st.session_state.reset_tax_form = False


_apply_form_resets()
_render_notifications()


def get_combined_transaction_data(username):
    all_data = []
    if st.session_state.csv_data is not None and not st.session_state.csv_data.empty:
        all_data.append(st.session_state.csv_data)
    try:
        end_date = date.today().strftime("%Y-%m-%d")
        start_date = (date.today() - timedelta(days=365)).strftime("%Y-%m-%d")
        daily_df = get_daily_items(username, start_date, end_date)
        if daily_df is not None and not daily_df.empty:
            all_data.append(daily_df)
    except Exception:
        pass
    if not all_data:
        return pd.DataFrame()
    combined = pd.concat(all_data, ignore_index=True)
    return combined.drop_duplicates(subset=["date", "description", "amount"])


ROLE_LEVEL = {"cashier": 1, "accountant": 2, "manager": 3, "owner": 4}

TAB_ROLES = {
    "📝 Daily Entry": "cashier",
    "📦 Inventory": "manager",
    "👥 Debtors/Credit": "accountant",
    "🏭 Suppliers": "manager",
    "✅ Expense Approval": "cashier",
    "📊 Reports & Analytics": "cashier",
    "💰 Budgets": "manager",
    "👥 User Management": "owner",
    "🧾 Invoices": "accountant",
    "💵 Cash Flow": "manager",
    "👔 Payroll": "manager",
    "🏛️ Tax": "accountant",
}

def check_permission(required_role):
    current_level = ROLE_LEVEL.get(st.session_state.user_role, 1)
    required_level = ROLE_LEVEL.get(required_role, 1)
    return current_level >= required_level


# ---- Authentication Sidebar ----
st.sidebar.title("🔐 Account")

if not st.session_state.logged_in:
    owner_exists = not can_signup()
    mode = st.sidebar.radio("Mode", ["Login", "Sign Up"], key="auth_mode")
    username = st.sidebar.text_input("Username", key="auth_username")
    password = st.sidebar.text_input("Password", type="password", key="auth_password")

    if mode == "Sign Up":
        phone = st.sidebar.text_input("Phone Number (compulsory)", key="signup_phone")
        full_name = st.sidebar.text_input("Full Name", key="signup_fullname")
        email = st.sidebar.text_input("Email", key="signup_email")
        if owner_exists:
            st.sidebar.warning("⚠️ An owner already exists. New accounts require owner approval before they become active.")
        if st.sidebar.button("Create Account", key="signup_btn"):
            if not phone or not phone.strip():
                st.sidebar.error("Phone number is required.")
            elif not username or not password:
                st.sidebar.error("Username and password required.")
            else:
                role = "cashier"
                ok, msg = create_user(username, password, role, full_name, email, phone)
                if ok:
                    st.sidebar.success("Signup successful. Your account is pending owner approval.")
                else:
                    st.sidebar.error(msg)
    else:
        if st.sidebar.button("Login", key="login_btn"):
            if not username or not password:
                st.sidebar.error("Enter username and password")
            elif authenticate(username, password):
                st.session_state.logged_in = True
                st.session_state.user = username
                st.session_state.user_role = get_user_role(username)
                st.session_state.csv_data = load_transactions(username)
                st.rerun()
            else:
                st.sidebar.error("Invalid credentials")
    st.title("💰 FinanceIQ - Complete SME Manager")
    st.info("Login or sign up to continue.")
    st.stop()

if st.sidebar.button("🚪 Logout", key="logout_btn"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# ---- Main UI after login ----
st.sidebar.markdown(f"### 👤 {st.session_state.user}")
st.sidebar.caption(f"Role: **{st.session_state.user_role.upper()}**")

pending_count = 0
if st.session_state.user_role == "owner":
    pending_users = get_pending_users()
    pending_count = len(pending_users)
    if pending_count > 0:
        st.sidebar.warning(f"⚠️ **{pending_count}** pending user(s) need approval")
        with st.sidebar.expander("📋 Review Pending Users"):
            for u in pending_users:
                st.write(f"**{u.get('name', u['username'])}** (@{u['username']})")
                st.caption(f"Role: {u['role']} | {u.get('email', 'N/A')} | {u.get('phone', 'N/A')}")
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("✅ Approve", key=f"approve_{u['username']}"):
                        ok, msg = activate_user(u['username'])
                        if ok:
                            st.sidebar.success(f"Approved {u['username']}")
                            st.rerun()
                        else:
                            st.sidebar.error(msg)
                with col_b:
                    if st.button("❌ Reject", key=f"reject_{u['username']}"):
                        ok, msg = reject_user(u['username'])
                        if ok:
                            st.sidebar.success(f"Rejected and removed {u['username']}")
                            st.rerun()
                        else:
                            st.sidebar.error(msg)

st.sidebar.markdown("---")

st.title("💰 FinanceIQ - Complete SME Manager")
st.markdown(f"Welcome back, **{st.session_state.user}**! Track inventory, manage debtors, control expenses, and grow your business.")

combined_data = get_combined_transaction_data(st.session_state.user)

# ---- KPI Row ----
if not combined_data.empty:
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    expenses = combined_data[combined_data["item_type"] == "expense"]["amount"].sum() if "item_type" in combined_data else combined_data["amount"].sum()
    sales = combined_data[combined_data["item_type"] == "sale"]["amount"].sum() if "item_type" in combined_data else 0
    with col1:
        st.metric("Total Sales (MTD)", f"KES {sales:,.0f}")
    with col2:
        st.metric("Total Expenses", f"KES {expenses:,.0f}")
    with col3:
        st.metric("Net Profit", f"KES {sales - expenses:,.0f}", delta="profit" if sales - expenses > 0 else "loss")
    with col4:
        debtors_df = get_debtors_list(st.session_state.user)
        total_debt = debtors_df["current_balance"].sum() if not debtors_df.empty else 0
        st.metric("Outstanding Debt", f"KES {total_debt:,.0f}")
    with col5:
        low_stock = get_low_stock_products(st.session_state.user)
        st.metric("Low Stock Items", len(low_stock))
    with col6:
        try:
            score = calculate_health_score(combined_data)
            st.metric("Health Score", f"{score:.0f}/100" if score else "N/A")
        except:
            st.metric("Health Score", "N/A")
    st.divider()
else:
    st.info("📭 Add daily transactions to see your business KPIs")


if st.sidebar.button("💬 Messages" + (f" ({pending_count})" if pending_count > 0 else ""), key="toggle_messages"):
    st.session_state.messages_open = not st.session_state.messages_open
    if st.session_state.messages_open:
        st.session_state.chat_open = False
if st.sidebar.button("🤖 AI Assistant", key="toggle_chat"):
    st.session_state.chat_open = not st.session_state.chat_open
    if st.session_state.chat_open:
        st.session_state.messages_open = False
st.sidebar.markdown("---")


def tab_daily_entry():
    st.subheader("Quick Transaction Entry")
    col_q1, col_q2, col_q3, col_q4, col_q5, col_q6 = st.columns(6)
    with col_q1:
        if st.button("🍔 Food", width="stretch", key="quick_food"):
            st.session_state.entry_type = "expense"
            st.session_state.entry_desc = "Food"
            st.rerun()
    with col_q2:
        if st.button("⛽ Fuel", width="stretch", key="quick_fuel"):
            st.session_state.entry_type = "expense"
            st.session_state.entry_desc = "Fuel"
            st.rerun()
    with col_q3:
        if st.button("🚖 Transport", width="stretch", key="quick_transport"):
            st.session_state.entry_type = "expense"
            st.session_state.entry_desc = "Transport"
            st.rerun()
    with col_q4:
        if st.button("🏠 Rent", width="stretch", key="quick_rent"):
            st.session_state.entry_type = "expense"
            st.session_state.entry_desc = "Rent"
            st.rerun()
    with col_q5:
        if st.button("💰 Sale", width="stretch", key="quick_sale"):
            st.session_state.entry_type = "sale"
            st.session_state.entry_desc = "Sale"
            st.rerun()
    with col_q6:
        if st.button("📦 Stock Sale", width="stretch", key="quick_stock_sale"):
            st.session_state.show_stock_sale = True
            st.session_state.entry_type = "sale"
    
    col1, col2 = st.columns(2)
    with col1:
        entry_date = st.date_input("Date", value=date.today(), key="entry_date")
        entry_type = st.radio("Type", ["Expense", "Sale"], index=0 if st.session_state.get("entry_type", "expense") == "expense" else 1, key="entry_type")
        description = st.text_input("Description", value=st.session_state.get("entry_desc", ""), key="entry_desc")
        amount = st.number_input("Amount (KES)", min_value=0.0, step=10.0, value=0.0, key="entry_amount")
        payment_method = st.selectbox("Payment Method", ["cash", "mpesa", "bank", "credit"], key="entry_payment")
        if st.button("➕ Add Entry", type="primary", key="add_entry"):
            if description and amount > 0:
                category = categorize_item_description(description)
                success, msg = save_daily_item(st.session_state.user, entry_date.strftime("%Y-%m-%d"), description, amount, category, entry_type.lower())
                if success:
                    st.success(f"✅ Added: {description} for KES {amount:.2f}")
                    st.session_state.refresh_trigger = True
                    st.session_state.reset_entry_form = True
                    time.sleep(0.3)
                    st.rerun()
                else:
                    st.error(f"Failed: {msg}")
            else:
                st.warning("Enter description and amount")
    
    with col2:
        available_cats = get_categories()
        suggested_cat = categorize_item_description(description) if description else "other"
        category_input = st.selectbox("Category", available_cats, index=available_cats.index(suggested_cat) if suggested_cat in available_cats else 0, key="entry_category")
        if st.session_state.get("show_stock_sale", False):
            st.subheader("Sell from Inventory")
            products_df = get_all_products(st.session_state.user)
            if not products_df.empty:
                product_names = products_df["name"].tolist()
                selected_product = st.selectbox("Select Product", product_names, key="stock_product")
                product_row = products_df[products_df["name"] == selected_product]
                if product_row.empty:
                    st.error("Selected product not found in inventory")
                else:
                    product_row = product_row.iloc[0]
                    quantity = st.number_input("Quantity", min_value=0.0, step=1.0, value=1.0, key="stock_qty")
                    total_amount = quantity * product_row["selling_price"]
                    st.info(f"Total: KES {total_amount:.2f}")
                    if st.button("Record Stock Sale", key="record_stock_sale"):
                        success, msg = update_stock(st.session_state.user, product_row["id"], quantity, "sale", f"Sale to customer", f"Quantity: {quantity}")
                        if success:
                            save_ok, save_msg = save_daily_item(st.session_state.user, date.today().strftime("%Y-%m-%d"), f"Sale: {selected_product} x{quantity}", total_amount, product_row["category"], "sale")
                            if save_ok:
                                st.success(f"Recorded sale of {quantity} x {selected_product}")
                            else:
                                st.error(f"Stock updated but daily item failed: {save_msg}")
                            st.session_state.show_stock_sale = False
                            st.session_state.reset_entry_form = True
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error(msg)
            else:
                st.warning("No products in inventory. Add products in Inventory tab.")
    
    with st.expander("📸 Upload Receipt (OCR)"):
        receipt_file = st.file_uploader("Choose receipt image (JPG/PNG) or PDF", type=["jpg", "png", "pdf"], key="receipt_upload")
        if receipt_file is not None:
            temp_path = f"/tmp/{receipt_file.name}"
            with open(temp_path, "wb") as f:
                f.write(receipt_file.getbuffer())
            with st.spinner("Extracting text from receipt..."):
                try:
                    receipt_data = extract_receipt(temp_path)
                    total = receipt_data.get("total", 0)
                    merchant = receipt_data.get("merchant", "Unknown")
                    raw_text = receipt_data.get("raw_text", "")
                    st.subheader("Extracted Information")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Merchant", merchant)
                    with col_b:
                        st.metric("Total Amount (KES)", f"{total:.2f}" if total else "Not found")
                    if receipt_data.get("items"):
                        st.write("**Line Items**")
                        for it in receipt_data.get("items", []):
                            st.write(f"- {it.get('description')} — Qty: {it.get('quantity')} — Unit: KES {it.get('unit_price') or 0:.2f} — Total: KES {it.get('total'):.2f}")
                    with st.expander("Raw OCR Text (click to edit)"):
                        corrected_text = st.text_area("Edit if needed", raw_text, height=200, key="ocr_correct_text")
                        if st.button("Re‑extract from corrected text", key="re_extract_ocr"):
                            lines = [{"text": line} for line in corrected_text.split("\n") if line.strip()]
                            if lines:
                                receipt_data = process_receipt_lines(lines)
                                total = receipt_data.get("total", 0)
                                merchant = receipt_data.get("merchant", "Unknown")
                                st.success(f"Updated total: {total:.2f}")
                                st.rerun()
                    manual_total = st.number_input("Or enter total manually (KES)", value=float(total) if total else 0.0, step=10.0, key="manual_total")
                    if st.button("Add Receipt Entry", key="add_receipt"):
                        if manual_total > 0:
                            receipt_date = st.date_input("Receipt Date", value=entry_date, key="receipt_date")
                            receipt_category = st.selectbox("Category", available_cats, key="receipt_category")
                            success, msg = save_daily_item(st.session_state.user, receipt_date.strftime("%Y-%m-%d"), f"Receipt: {merchant}", manual_total, receipt_category, "expense")
                            if success:
                                st.success(f"Added expense: {merchant} for KES {manual_total:.2f}")
                                st.session_state.reset_entry_form = True
                                st.rerun()
                            else:
                                st.error(msg)
                        else:
                            st.warning("Enter a valid amount")
                except Exception as e:
                    st.error(f"OCR failed: {e}")
    
    st.subheader("📋 Recent Transactions")
    week_ago = (date.today() - timedelta(days=7)).strftime("%Y-%m-%d")
    recent_df = get_daily_items(st.session_state.user, week_ago, date.today().strftime("%Y-%m-%d"))
    if recent_df is not None and not recent_df.empty:
        display_df = recent_df.copy()
        display_df["amount"] = display_df["amount"].apply(lambda x: f"KES {x:.2f}")
        st.dataframe(display_df, width="stretch")
    else:
        st.info("No recent transactions")
    
    
def tab_inventory():
    st.subheader("📦 Inventory / Products")
    if not check_permission("manager"):
        st.warning("Only managers and owners can manage inventory")
    else:
        inv_tab1, inv_tab2, inv_tab3 = st.tabs(["Products", "Add Product", "Stock Alerts"])
        with inv_tab1:
            products_df = get_all_products(st.session_state.user)
            if not products_df.empty:
                total_stock_value = (products_df["current_stock"] * products_df["buying_price"]).sum()
                st.metric("Total Inventory Value", f"KES {total_stock_value:,.2f}")
                st.dataframe(products_df[["name", "sku", "category", "current_stock", "min_stock_level", "selling_price"]], width="stretch")
                st.subheader("Adjust Stock")
                col_a, col_b = st.columns(2)
                with col_a:
                    prod_to_adjust = st.selectbox("Select Product", products_df["name"].tolist(), key="adjust_product")
                    prod_row = products_df[products_df["name"] == prod_to_adjust].iloc[0]
                    st.caption(f"Current stock: **{prod_row['current_stock']}**")
                with col_b:
                    adj_mode = st.selectbox("Mode", ["Add", "Subtract", "Replace"], key="adj_mode")
                    adjustment_qty = st.number_input("Quantity", min_value=0.0, step=1.0, value=1.0, key="adjust_qty")
                if st.button("Update Stock", key="update_stock_btn"):
                    mode = "add" if adj_mode == "Add" else "subtract" if adj_mode == "Subtract" else "replace"
                    success, msg = update_stock(st.session_state.user, prod_row["id"], adjustment_qty, "adjustment", adjust_mode=mode)
                    if success:
                        _notify("stock_updated", "Stock updated", msg)
                        st.session_state.reset_adjust_form = True
                        st.session_state.reset_inventory_form = True
                        st.rerun()
                    else:
                        st.error(msg)
            else:
                st.info("No products added yet")
        with inv_tab2:
            col1, col2 = st.columns(2)
            with col1:
                prod_name = st.text_input("Product Name", key="prod_name")
                prod_sku = st.text_input("SKU (Unique Code)", key="prod_sku")
                prod_category = st.selectbox("Category", get_categories(), key="prod_category")
                prod_unit = st.selectbox("Unit", ["pcs", "kg", "liters", "meters", "boxes", "packs"], key="prod_unit")
            with col2:
                buying_price = st.number_input("Buying Price (KES)", min_value=0.0, step=1.0, key="buying_price")
                selling_price = st.number_input("Selling Price (KES)", min_value=0.0, step=1.0, key="selling_price")
                initial_stock = st.number_input("Initial Stock", min_value=0.0, step=1.0, value=0.0, key="initial_stock")
                min_stock = st.number_input("Min Stock Alert Level", min_value=0.0, step=1.0, value=10.0, key="min_stock")
                supplier_id = None
            if st.button("➕ Add Product", type="primary", key="add_product_btn"):
                if prod_name and prod_sku:
                    success, msg = add_product(st.session_state.user, prod_name, prod_sku, prod_category, buying_price, selling_price, initial_stock, min_stock, prod_unit, supplier_id)
                    if success:
                        st.success(msg)
                        st.session_state.pop("prod_name", None)
                        st.session_state.pop("prod_sku", None)
                        st.session_state.pop("buying_price", None)
                        st.session_state.pop("selling_price", None)
                        st.session_state.pop("initial_stock", None)
                        st.session_state.pop("min_stock", None)
                        st.rerun()
                    else:
                        st.error(msg)
                else:
                    st.warning("Product name and SKU required")
        with inv_tab3:
            low_stock = get_low_stock_products(st.session_state.user)
            if not low_stock.empty:
                st.warning("⚠️ Products below minimum stock level:")
                for _, row in low_stock.iterrows():
                    st.write(f"**{row['name']}** - Current: {row['current_stock']} | Min: {row['min_stock_level']}")
            else:
                st.success("✅ All products above minimum stock levels")
    
def tab_debtors_credit():
    st.subheader("👥 Customer Credit & Payables")
    if not check_permission("accountant"):
        st.warning("Only accountants, managers, and owners can manage credit")
        return
    
    if "credit_active_tab" not in st.session_state:
        st.session_state.credit_active_tab = "Debtors"
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🟦 Debtors", use_container_width=True, type="primary" if st.session_state.credit_active_tab == "Debtors" else "secondary"):
            st.session_state.credit_active_tab = "Debtors"
            st.rerun()
    with col2:
        if st.button("🟥 Creditors", use_container_width=True, type="primary" if st.session_state.credit_active_tab == "Creditors" else "secondary"):
            st.session_state.credit_active_tab = "Creditors"
            st.rerun()
    st.divider()
    
    if st.session_state.credit_active_tab == "Debtors":
        _render_debtors_tab()
    elif st.session_state.credit_active_tab == "Creditors":
        _render_creditors_tab()


def _render_debtors_tab():
    st.subheader("Customers / Debtors - Credit Management")
    
    # Update credit statuses automatically
    update_credit_statuses(st.session_state.user)
    
    # Credit Status Summary
    credit_summary = get_credit_status_summary(st.session_state.user)
    if not credit_summary.empty:
        with st.expander("📊 Credit Status Overview"):
            cols = st.columns(min(len(credit_summary), 5))
            for idx, (_, row) in enumerate(credit_summary.iterrows()):
                status_emoji = {
                    "active": "🟢", "suspended": "🟡", "no_credit": "⚪",
                    "over_limit": "🔴", "overdue": "🔴", "blacklisted": "⚫"
                }.get(row["credit_status"], "⚪")
                with cols[idx % 5]:
                    st.metric(
                        f"{status_emoji} {row['credit_status'].title()}",
                        f"{row['customer_count']} customers",
                        f"KES {row['total_outstanding']:,.0f} outstanding"
                    )
    
    # Add Customer Form
    with st.expander("➕ Add New Customer"):
        c1, c2 = st.columns(2)
        with c1:
            cust_name = st.text_input("Full Name", key="cust_name")
            cust_phone = st.text_input("Phone Number", key="cust_phone")
        with c2:
            cust_email = st.text_input("Email", key="cust_email")
            credit_limit = st.number_input("Credit Limit (KES)", min_value=0.0, step=1000.0, value=0.0, key="cust_credit_limit")
            cust_notes = st.text_area("Notes", key="cust_notes", height=50)
        
        # Opening balance for existing debt
        with st.expander("📋 Opening Balance (Optional)"):
            outstanding_amount = st.number_input("Opening Outstanding Amount (KES)", min_value=0.0, step=100.0, value=0.0, key="cust_outstanding_amount")
            outstanding_due = st.date_input("Opening Due Date", value=date.today() + timedelta(days=30), key="cust_outstanding_due")
        
        if st.button("Add Customer", key="add_customer_btn", type="primary"):
            if cust_name and cust_phone:
                ok, msg = add_customer(st.session_state.user, cust_name, cust_phone, cust_email, "", credit_limit, st.session_state.user, cust_notes)
                if ok:
                    customers_df = get_all_customers(st.session_state.user)
                    new_customer = customers_df[
                        (customers_df["name"] == cust_name) &
                        (customers_df["phone"] == cust_phone)
                    ]
                    if outstanding_amount > 0 and not new_customer.empty:
                        customer_id = new_customer.iloc[-1]["id"]
                        credit_ok, credit_msg = add_credit_sale(
                            st.session_state.user,
                            customer_id,
                            outstanding_amount,
                            "Opening outstanding credit",
                            outstanding_due.strftime("%Y-%m-%d"),
                        )
                        if not credit_ok:
                            st.warning(f"Customer added, but opening credit was not recorded: {credit_msg}")
                    st.success(f"✅ {msg}")
                    for k in ["cust_name", "cust_phone", "cust_email", "cust_credit_limit", "cust_outstanding_amount", "cust_outstanding_due", "cust_notes"]:
                        st.session_state.pop(k, None)
                    st.rerun()
                else:
                    st.error(msg)
            else:
                st.warning("Name and phone are required")

    # Search and filter
    col_search, col_filter = st.columns([3, 1])
    with col_search:
        search_debtors = st.text_input("🔍 Search customers...", key="search_debtors")
    with col_filter:
        status_filter = st.selectbox("Filter by Status", ["all", "active", "no_credit", "over_limit", "overdue", "suspended", "blacklisted"], key="status_filter")
    
    debtors_df = get_all_customers(st.session_state.user)
    
    # Apply filters
    if search_debtors and not debtors_df.empty:
        debtors_df = debtors_df[
            debtors_df["name"].str.contains(search_debtors, case=False, na=False) |
            debtors_df["phone"].str.contains(search_debtors, case=False, na=False) |
            debtors_df["email"].str.contains(search_debtors, case=False, na=False)
        ]
    if status_filter != "all" and not debtors_df.empty:
        debtors_df = debtors_df[debtors_df["credit_status"] == status_filter]
    
    if not debtors_df.empty:
        total_receivable = debtors_df["current_balance"].sum()
        total_credit_limit = debtors_df["credit_limit"].sum()
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Total Receivable", f"KES {total_receivable:,.2f}")
        with col_m2:
            st.metric("Total Credit Limit", f"KES {total_credit_limit:,.2f}")
        with col_m3:
            utilization = (total_receivable / total_credit_limit * 100) if total_credit_limit > 0 else 0
            st.metric("Credit Utilization", f"{utilization:.1f}%")
        
        st.caption(f"**{len(debtors_df)} customer(s)**")
        
        for _, row in debtors_df.iterrows():
            # Determine status emoji and color
            status_info = {
                "active": ("🟢", "green", "Credit active"),
                "suspended": ("🟡", "orange", "Credit suspended"),
                "no_credit": ("⚪", "gray", "No credit facility"),
                "over_limit": ("🔴", "red", "Over credit limit!"),
                "overdue": ("🔴", "red", "Has overdue payments!"),
                "blacklisted": ("⚫", "black", "Blacklisted")
            }
            status_emoji, status_color, status_desc = status_info.get(row.get("credit_status", "no_credit"), ("⚪", "gray", "Unknown"))
            
            with st.container(border=True):
                # Header row
                col_h1, col_h2, col_h3, col_h4 = st.columns([3, 2, 2, 1])
                with col_h1:
                    st.markdown(f"**{row['name']}** {status_emoji}")
                    st.caption(f"📞 {row['phone']} | {row['email'] or 'No email'}")
                    if row.get('approved_by'):
                        st.caption(f"Credit approved by: {row['approved_by']}")
                with col_h2:
                    st.markdown(f"**Outstanding:** KES {row['current_balance']:,.2f}")
                    st.caption(f"Credit Limit: KES {row['credit_limit']:,.2f}")
                with col_h3:
                    avail = max(0, float(row['credit_limit']) - float(row['current_balance']))
                    st.markdown(f"**Available:** KES {avail:,.2f}")
                    st.caption(status_desc)
                with col_h4:
                    # Quick actions
                    if st.button("⚙️", key=f"actions_{row['id']}", help="Credit Actions"):
                        st.session_state[f"show_credit_actions_{row['id']}"] = not st.session_state.get(f"show_credit_actions_{row['id']}", False)
                
                # Credit actions (shown when gear icon clicked)
                if st.session_state.get(f"show_credit_actions_{row['id']}", False):
                    with st.expander("⚙️ Credit Management Actions", expanded=True):
                        # Update credit limit
                        new_limit = st.number_input("New Credit Limit (KES)", 
                                                   value=float(row['credit_limit']), 
                                                   min_value=0.0, 
                                                   step=1000.0,
                                                   key=f"new_limit_{row['id']}")
                        if st.button("Update Credit Limit", key=f"update_limit_{row['id']}"):
                            ok, msg = update_customer_credit_limit(st.session_state.user, row['id'], new_limit, st.session_state.user)
                            if ok:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)
                        
                        # Change credit status
                        new_status = st.selectbox("Change Credit Status", 
                                                 ["active", "suspended", "no_credit", "blacklisted"],
                                                 index=["active", "suspended", "no_credit", "blacklisted"].index(row.get("credit_status", "no_credit")) if row.get("credit_status") in ["active", "suspended", "no_credit", "blacklisted"] else 0,
                                                 key=f"new_status_{row['id']}")
                        status_notes = st.text_area("Reason for status change", key=f"status_notes_{row['id']}")
                        if st.button("Update Status", key=f"update_status_{row['id']}"):
                            ok, msg = update_customer_credit_status(st.session_state.user, row['id'], new_status, status_notes)
                            if ok:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)
                
                # Payment section
                st.divider()
                st.markdown("**💰 Record Payment**")
                col_p1, col_p2, col_p3, col_p4 = st.columns([2, 2, 2, 1])
                with col_p1:
                    payment_amount = st.number_input("Amount (KES)", min_value=0.0, step=100.0, key=f"debt_pay_{row['id']}")
                with col_p2:
                    payment_method = st.selectbox("Method", ["cash", "mpesa", "bank"], key=f"debt_method_{row['id']}")
                with col_p3:
                    payment_ref = st.text_input("Reference", key=f"debt_ref_{row['id']}")
                with col_p4:
                    if st.button("Record Payment", key=f"debt_btn_{row['id']}", type="primary"):
                        if payment_amount > 0:
                            ok, msg = record_credit_payment(st.session_state.user, 0, row["id"], payment_amount, payment_method, payment_ref)
                            if ok:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)
                        else:
                            st.warning("Enter payment amount")
                
                # Credit history and invoices
                with st.expander("📄 Credit Sales & Invoices"):
                    # Credit sales
                    credit_sales_df, payments_df = get_customer_credit_history(st.session_state.user, row['id'])
                    if not credit_sales_df.empty:
                        st.markdown("**Credit Sales:**")
                        st.dataframe(credit_sales_df[['date', 'amount', 'description', 'due_date', 'paid_amount', 'status']], width="stretch")
                    else:
                        st.info("No credit sales recorded")
                    
                    # Invoices
                    inv_df = get_invoices(st.session_state.user)
                    if not inv_df.empty:
                        cust_inv = inv_df[inv_df["customer_id"] == row["id"]]
                        if not cust_inv.empty:
                            st.markdown("**Invoices:**")
                            st.dataframe(cust_inv[["invoice_number", "invoice_date", "due_date", "total_amount", "paid_amount", "status"]], width="stretch")
                        else:
                            st.info("No invoices")
                    else:
                        st.info("No invoices")
                
                with st.expander("📜 Payment History"):
                    hist_df = get_credit_payment_history(st.session_state.user, row["id"])
                    if not hist_df.empty:
                        st.dataframe(hist_df, width="stretch")
                        st.caption(f"Total paid: KES {hist_df['amount'].sum():,.2f}")
                    else:
                        st.info("No payments recorded yet")
    else:
        st.info("No customers added yet")
def _render_creditors_tab():
    st.subheader("Who the business owes (Creditors)")
    with st.expander("➕ Add Creditor"):
        c1, c2 = st.columns(2)
        with c1:
            cred_name = st.text_input("Creditor Name", key="cred_name")
            cred_phone = st.text_input("Phone", key="cred_phone")
            cred_email = st.text_input("Email", key="cred_email")
        with c2:
            cred_desc = st.text_input("Description (e.g. Supplier Loan, advance)", key="cred_desc")
            cred_amt = st.number_input("Amount Owed (KES)", min_value=0.0, step=100.0, key="cred_amt")
            cred_due = st.date_input("Due Date", value=date.today() + timedelta(days=30), key="cred_due")
        if st.button("Add Creditor", key="add_creditor_btn"):
            if cred_name and cred_amt > 0:
                ok, msg = add_creditor(st.session_state.user, cred_name, cred_phone, cred_email, cred_desc, cred_amt, cred_due.strftime("%Y-%m-%d"))
                if ok:
                    for k in ["cred_name", "cred_phone", "cred_email", "cred_desc", "cred_amt"]:
                        st.session_state.pop(k, None)
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
            else:
                st.warning("Creditor name and amount owed are required")

    search_creditors = st.text_input("🔍 Search creditors...", key="search_creditors")
    creditors_df = get_creditors_by_user(st.session_state.user)
    if search_creditors and not creditors_df.empty:
        creditors_df = creditors_df[
            creditors_df["name"].str.contains(search_creditors, case=False, na=False) |
            creditors_df["phone"].str.contains(search_creditors, case=False, na=False) |
            creditors_df["email"].str.contains(search_creditors, case=False, na=False) |
            creditors_df["description"].str.contains(search_creditors, case=False, na=False)
        ]
    total_payable = get_total_payable(st.session_state.user)
    st.metric("Total Payable", f"KES {total_payable:,.2f}")
    if not creditors_df.empty:
        st.caption(f"**{len(creditors_df)} creditor(s)**")
        for _, row in creditors_df.iterrows():
            with st.container(border=True):
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.markdown(f"**{row['name']}**")
                    st.caption(f"{row['description']}")
                with col_b:
                    st.markdown(f"**KES {row['amount_owed']:,.2f}**")
                    st.caption(f"Due: {row['due_date']}")
                pay_amount = st.number_input("Payment Amount (KES)", min_value=0.0, step=100.0, key=f"cred_pay_{row['id']}")
                pay_method = st.selectbox("Method", ["cash", "mpesa", "bank", "cheque"], key=f"cred_method_{row['id']}")
                pay_ref = st.text_input("Reference", key=f"cred_ref_{row['id']}")
                if st.button("Record Payment", key=f"cred_btn_{row['id']}"):
                    if pay_amount > 0:
                        ok, msg = record_creditor_payment(st.session_state.user, row["id"], pay_amount, pay_method, pay_ref, "")
                        if ok:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)
                with st.expander("📜 Payment History"):
                    hist = get_creditor_payment_history(st.session_state.user, row["id"])
                    if not hist.empty:
                        st.dataframe(hist, width="stretch")
                        st.caption(f"Total paid: KES {hist['amount'].sum():,.2f}")
                    else:
                        st.info("No payments yet")
    else:
        st.info("No creditors yet")
def tab_suppliers():
    st.subheader("🏭 Supplier Management")
    if not check_permission("manager"):
        st.warning("Only managers and owners can manage suppliers")
    else:
        sup_tab1, sup_tab2, sup_tab3 = st.tabs(["Suppliers", "Add Supplier", "Performance"])
        with sup_tab1:
            suppliers_df = get_all_suppliers(st.session_state.user)
            if not suppliers_df.empty:
                st.dataframe(suppliers_df, width="stretch")
            else:
                st.info("No suppliers added")
        with sup_tab2:
            sup_name = st.text_input("Supplier Name", key="sup_name")
            sup_phone = st.text_input("Phone", key="sup_phone")
            sup_email = st.text_input("Email", key="sup_email")
            payment_terms = st.selectbox("Payment Terms", ["cash_on_delivery", "net_30", "net_60", "net_90"], key="sup_payment_terms")
            avg_delivery = st.number_input("Avg Delivery Days", min_value=0, value=7, key="sup_avg_delivery")
            if st.button("Add Supplier", key="add_supplier_btn"):
                if sup_name and sup_phone:
                    success, msg = add_supplier(st.session_state.user, sup_name, sup_phone, sup_email, "", payment_terms, avg_delivery)
                    if success:
                        st.success(msg)
                        st.session_state.pop("sup_name", None)
                        st.session_state.pop("sup_phone", None)
                        st.session_state.pop("sup_email", None)
                        st.session_state.pop("sup_avg_delivery", None)
                        st.rerun()
                    else:
                        st.error(msg)
        with sup_tab3:
            perf_df = get_supplier_performance(st.session_state.user)
            if not perf_df.empty:
                st.dataframe(perf_df, width="stretch")
                fig = px.bar(perf_df, x="name", y="average_delivery_days", title="Supplier Delivery Performance")
                st.plotly_chart(fig, width="stretch")
    
    
def tab_expense_approval():
    st.subheader("✅ Expense Approval Workflow")
    if st.session_state.user_role != "owner":
        app_tab1, app_tab2 = st.tabs(["Submit Request", "My Requests & Status"])
        with app_tab1:
            if check_permission("cashier"):
                req_amount = st.number_input("Expense Amount (KES)", min_value=0.0, step=100.0, key="req_amount")
                req_category = st.selectbox("Category", get_categories(), key="req_category")
                req_description = st.text_area("Description / Justification", key="req_description")
                receipt_file = st.file_uploader("Upload Receipt (optional)", type=["jpg", "png", "pdf"], key="req_receipt")
                receipt_path = ""
                if receipt_file:
                    os.makedirs("receipts", exist_ok=True)
                    receipt_path = f"receipts/{int(time.time())}_{receipt_file.name}"
                    with open(receipt_path, "wb") as f:
                        f.write(receipt_file.getbuffer())
                if st.button("Submit for Approval", key="submit_expense"):
                    if req_amount > 0 and req_description:
                        success, msg = submit_expense_request(st.session_state.user, st.session_state.user, req_amount, req_category, req_description, receipt_path)
                        if success:
                            st.success(msg)
                        else:
                            st.error(msg)
                    else:
                        st.warning("Enter amount and description")
        with app_tab2:
            st.subheader("📋 My Expense Requests")
            my_expenses = get_expenses_by_status(st.session_state.user, "all")
            if not my_expenses.empty:
                status_filter = st.selectbox("Filter by Status", ["all", "pending", "approved", "rejected"], key="my_expense_filter")
                filtered = my_expenses if status_filter == "all" else my_expenses[my_expenses["status"] == status_filter]
                if not filtered.empty:
                    for idx, row in filtered.iterrows():
                        status_color = {"pending": "🟡", "approved": "🟢", "rejected": "🔴"}.get(row["status"], "⚪")
                        with st.expander(f"{status_color} {row['description']} - KES {row['amount']:,.2f} ({row['status'].upper()})"):
                            st.write(f"**Category:** {row['category']}")
                            st.write(f"**Date:** {row['created_at']}")
                            st.write(f"**Status:** {row['status']}")
                            if row["status"] == "approved":
                                st.write(f"**Approved by:** {row['approved_by']}")
                                st.write(f"**Approved at:** {row['approved_at']}")
                            elif row["status"] == "rejected":
                                st.write(f"**Rejected by:** {row['approved_by']}")
                                st.write(f"**Reason:** {row['rejected_reason']}")
                else:
                    st.info("No requests match the selected filter")
            else:
                st.info("You haven't submitted any expense requests")
    if st.session_state.user_role == "owner":
        st.divider()
        st.subheader("👔 Approve / Reject Requests")
        approval_tab1, approval_tab2, approval_tab3 = st.tabs(["🟡 Pending", "✅ Approved", "❌ Rejected"])
        with approval_tab1:
            pending_df = get_pending_expenses()
            if not pending_df.empty:
                for idx, row in pending_df.iterrows():
                    with st.expander(f"{row['description']} - KES {row['amount']:,.2f}"):
                        st.write(f"**Requester:** {row['requester']}")
                        st.write(f"**Category:** {row['category']}")
                        st.write(f"**Date:** {row['created_at']}")
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(f"✅ Approve", key=f"approve_{row['id']}_{idx}"):
                                success, msg = approve_expense(row['id'], st.session_state.user)
                                st.success(msg)
                                st.rerun()
                        with col2:
                            reason = st.text_input("Rejection reason", key=f"reason_{row['id']}_{idx}")
                            if st.button(f"❌ Reject", key=f"reject_{row['id']}_{idx}"):
                                success, msg = approve_expense(row['id'], st.session_state.user, reason)
                                st.success(msg)
                                st.rerun()
            else:
                st.success("No pending approvals")
        with approval_tab2:
            approved_df = get_expenses_by_status(None, "approved")
            if not approved_df.empty:
                for idx, row in approved_df.iterrows():
                    with st.expander(f"✅ {row['description']} - KES {row['amount']:,.2f}"):
                        st.write(f"**Requester:** {row['requester']}")
                        st.write(f"**Approved by:** {row['approved_by']}")
                        st.write(f"**Approved at:** {row['approved_at']}")
                        st.write(f"**Date:** {row['created_at']}")
            else:
                st.info("No approved expenses yet")
        with approval_tab3:
            rejected_df = get_expenses_by_status(None, "rejected")
            if not rejected_df.empty:
                for idx, row in rejected_df.iterrows():
                    with st.expander(f"❌ {row['description']} - KES {row['amount']:,.2f}"):
                        st.write(f"**Requester:** {row['requester']}")
                        st.write(f"**Rejected by:** {row['approved_by']}")
                        st.write(f"**Reason:** {row['rejected_reason']}")
                        st.write(f"**Date:** {row['created_at']}")
            else:
                st.info("No rejected expenses yet")
    
    
def tab_reports_analytics():
    st.subheader("📊 Business Intelligence & Reports")
    report_tab1, report_tab2, report_tab3, report_tab4 = st.tabs(["Year-over-Year", "Top Products", "Accountant Export", "Financial Analysis"])
    with report_tab1:
        st.subheader("Year-over-Year Comparison")
        yoy_df = get_yoy_comparison(st.session_state.user)
        if not yoy_df.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=yoy_df["month"], y=yoy_df["total_current"], name="Current Year", marker_color="blue"))
            fig.add_trace(go.Bar(x=yoy_df["month"], y=yoy_df["total_last"], name="Last Year", marker_color="lightblue"))
            fig.update_layout(title="Monthly Spending Comparison", xaxis_title="Month", yaxis_title="Amount (KES)")
            st.plotly_chart(fig, width="stretch")
            st.subheader("Biggest Changes")
            changes = yoy_df.nlargest(3, "change")[["month", "change", "change_percent"]]
            st.dataframe(changes, width="stretch")
        else:
            st.info("Need at least 2 years of data for comparison")
    with report_tab2:
        st.subheader("Top Selling Products")
        products_df = get_all_products(st.session_state.user)
        if not products_df.empty:
            st.dataframe(products_df[["name", "current_stock", "selling_price"]].head(10), width="stretch")
            fig = px.bar(products_df.head(10), x="name", y="selling_price", title="Top Products by Price")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Add products to see top sellers")
    with report_tab3:
        st.subheader("Export for Accountant")
        start_date = st.date_input("Start Date", value=date.today() - timedelta(days=30), key="report_start")
        end_date = st.date_input("End Date", value=date.today(), key="report_end")
        if st.button("Generate Accountant Report", key="gen_report"):
            with st.spinner("Generating report..."):
                report_data = export_accountant_report(st.session_state.user, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
                if report_data:
                    st.subheader("Summary")
                    st.dataframe(report_data["summary"], width="stretch")
                    for name, df in report_data.items():
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            csv = df.to_csv(index=False)
                            st.download_button(f"Download {name}.csv", csv, f"{name}.csv", "text/csv", key=f"download_{name}")
                    st.success("Report ready for accountant")
                else:
                    st.error("No data for selected period")
    with report_tab4:
        st.subheader("Financial Health Analysis")
        if not combined_data.empty:
            score = calculate_health_score(combined_data)
            st.metric("Overall Health Score", f"{score}/100" if score else "N/A")
            insights = health_insights(combined_data)
            for insight in insights:
                st.write(f"• {insight}")
            anomalies = detect_anomalies(combined_data, lookback_days=90)
            if anomalies is not None and not anomalies.empty:
                st.warning(f"⚠️ {len(anomalies)} unusual transactions detected")
                st.dataframe(anomalies[["date", "description", "amount", "explanation"]].head(5), width="stretch")
        else:
            st.info("Add more data for analysis")
    
    
def tab_budgets():
    st.subheader("💰 Budget Management")
    if not check_permission("manager"):
        st.warning("Only managers and owners can set budgets")
    else:
        current_year = date.today().year
        current_month = date.today().month
        col1, col2 = st.columns(2)
        with col1:
            budget_year = st.number_input("Year", min_value=2020, value=current_year, key="budget_year")
        with col2:
            budget_month = st.selectbox("Month", range(1, 13), index=current_month - 1, key="budget_month")
        with st.expander("Set New Budget"):
            budget_cat = st.selectbox("Category", get_categories(), key="budget_cat")
            budget_amt = st.number_input("Budget Amount (KES)", min_value=0.0, step=1000.0, key="budget_amt")
            if st.button("Save Budget", key="save_budget"):
                success, msg = set_budget_enhanced(st.session_state.user, budget_cat, budget_amt, "monthly", budget_year, budget_month)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
        st.subheader(f"Budget Status - {budget_year}/{budget_month}")
        budget_status = get_budget_status(st.session_state.user, budget_year, budget_month)
        if not budget_status.empty:
            for _, row in budget_status.iterrows():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{row['category'].title()}**")
                    st.progress(min(row['percent'] / 100, 1.0))
                    st.caption(f"Budget: KES {row['amount']:,.2f} | Spent: KES {row['spent']:,.2f} | Remaining: KES {row['remaining']:,.2f}")
                with col2:
                    st.write(row['status'])
                st.divider()
        else:
            st.info("No budgets set for this period")
    
    
def tab_user_management():
    st.subheader("👥 User Management")
    if st.session_state.user_role != "owner":
        st.warning("Only the business owner can manage users")
    else:
        tab_users, tab_add, tab_pending = st.tabs(["👤 Manage Users", "Add New User", "📋 Pending Approvals"])
        with tab_pending:
            pending = get_pending_users()
            if pending:
                st.write(f"**{len(pending)} user(s) waiting for approval**")
                for u in pending:
                    with st.expander(f"{u.get('name', u['username'])} (@{u['username']})"):
                        st.caption(f"Role: {u['role']} | Phone: {u.get('phone', 'N/A')} | Email: {u.get('email', 'N/A')}")
                        c1, c2 = st.columns(2)
                        with c1:
                            if st.button("✅ Approve", key=f"approve_sidebar_{u['username']}"):
                                ok, msg = activate_user(u['username'])
                                if ok:
                                    st.success(f"Approved {u['username']}")
                                    st.rerun()
                                else:
                                    st.error(msg)
                        with c2:
                            if st.button("❌ Reject", key=f"reject_sidebar_{u['username']}"):
                                ok, msg = reject_user(u['username'])
                                if ok:
                                    st.success(f"Rejected {u['username']}")
                                    st.rerun()
                                else:
                                    st.error(msg)
            else:
                st.success("✅ No pending approvals")
        with tab_users:
            users = get_all_users(st.session_state.user)
            if users:
                for user in users:
                    if user["username"] == st.session_state.user and user.get("blocked"):
                        user["blocked"] = False
                    with st.container(border=True):
                        c1, c2, c3, c4 = st.columns([3, 2, 2, 3])
                        with c1:
                            st.write(f"**{user.get('name', user['username'])}** (@{user['username']})")
                            st.caption(f"Role: {user['role']}")
                        with c2:
                            st.write(f"📧 {user.get('email', 'N/A')}")
                        with c3:
                            st.write(f"📞 {user.get('phone', 'N/A')}")
                        with c4:
                            if user["username"] == st.session_state.user:
                                st.caption("⚠️ Current owner")
                            elif user.get("blocked"):
                                if st.button("✅ Unblock", key=f"unblock_{user['username']}", type="primary"):
                                    ok, msg = unblock_user(user["username"])
                                    if ok:
                                        st.success(msg)
                                        st.rerun()
                                    else:
                                        st.error(msg)
                            else:
                                bcol, dcol = st.columns(2)
                                with bcol:
                                    if st.button("🚫 Block", key=f"block_{user['username']}"):
                                        ok, msg = block_user(user["username"])
                                        if ok:
                                            st.success(msg)
                                            st.rerun()
                                        else:
                                            st.error(msg)
                                with dcol:
                                    if st.button("🗑️ Delete", key=f"delete_{user['username']}"):
                                        ok, msg = delete_user(user["username"])
                                        if ok:
                                            st.success(msg)
                                            st.rerun()
                                        else:
                                            st.error(msg)
                        if user["username"] != st.session_state.user:
                            with st.expander("✏️ Edit Details & Notes"):
                                ed_name = st.text_input("Full Name", value=user.get("name") or "", key=f"edit_name_{user['username']}")
                                ed_email = st.text_input("Email", value=user.get("email") or "", key=f"edit_email_{user['username']}")
                                ed_phone = st.text_input("Phone", value=user.get("phone") or "", key=f"edit_phone_{user['username']}")
                                ed_notes = st.text_area("Owner Notes", value=get_user_notes(user["username"]), key=f"edit_notes_{user['username']}")
                                if st.button("💾 Save Changes", key=f"save_edit_{user['username']}"):
                                    ok, msg = update_user_details(user["username"], ed_name, ed_email, ed_phone, ed_notes)
                                    if ok:
                                        st.success(msg)
                                        st.rerun()
                                    else:
                                        st.error(msg)
            else:
                st.info("No users found")
        with tab_add:
            st.subheader("Add New User")
            new_user = st.text_input("Username", key="new_user")
            new_pass = st.text_input("Password", type="password", key="new_pass")
            new_role = st.selectbox("Role", ["owner", "manager", "accountant", "cashier"], key="new_role")
            new_name = st.text_input("Full Name", key="new_name")
            new_email = st.text_input("Email", key="new_email")
            new_phone = st.text_input("Phone Number (compulsory)", key="new_phone")
            if st.button("Create User", type="primary", key="create_user_btn"):
                if not new_phone or not new_phone.strip():
                    st.warning("Phone number is required when creating a new user.")
                elif new_user and new_pass:
                    ok, msg = create_user(new_user, new_pass, new_role, new_name, new_email, new_phone)
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
                else:
                    st.warning("Username and password required")
    
    
def tab_invoices():
    st.subheader("🧾 Invoices")
    if not check_permission("accountant"):
        st.warning("Only accountants, managers, and owners can manage invoices")
    else:
        inv_tab1, inv_tab2 = st.tabs(["Create Invoice", "Invoice History"])
        with inv_tab1:
            customers_df = get_all_customers(st.session_state.user)
            if not customers_df.empty:
                col1, col2 = st.columns(2)
                with col1:
                    cust = st.selectbox("Customer", customers_df["name"].tolist(), key="inv_customer")
                    cust_row = customers_df[customers_df["name"] == cust].iloc[0]
                    invoice_num = st.text_input("Invoice Number", key="inv_num")
                    invoice_date = st.date_input("Invoice Date", value=date.today(), key="inv_date")
                    due_date = st.date_input("Due Date", value=date.today() + timedelta(days=30), key="inv_due")
                with col2:
                    subtotal = st.number_input("Subtotal (KES)", min_value=0.0, step=100.0, key="inv_subtotal")
                    tax_rate = st.number_input("Tax Rate (%)", min_value=0.0, max_value=100.0, value=16.0, key="inv_tax_rate")
                    tax_amount = subtotal * (tax_rate / 100)
                    total = subtotal + tax_amount
                    st.metric("Total Amount", f"KES {total:,.2f}", delta=f"Tax: KES {tax_amount:,.2f}")
                notes = st.text_area("Notes", key="inv_notes")
                if st.button("Create Invoice", type="primary", key="create_invoice"):
                    if invoice_num and subtotal > 0:
                        ok, msg = create_invoice(st.session_state.user, cust_row["id"], invoice_num, invoice_date.strftime("%Y-%m-%d"), due_date.strftime("%Y-%m-%d"), subtotal, tax_amount, total, notes)
                        if ok:
                            st.success(msg)
                            st.session_state.pop("inv_num", None)
                            st.session_state.pop("inv_subtotal", None)
                            st.session_state.pop("inv_tax_rate", None)
                            st.session_state.pop("inv_notes", None)
                            st.rerun()
                        else:
                            st.error(msg)
                    else:
                        st.warning("Invoice number and subtotal required")
            else:
                st.info("Add customers first in Debtors/Credit tab")
        with inv_tab2:
            invoices_df = get_invoices(st.session_state.user)
            if not invoices_df.empty:
                status_filter = st.selectbox("Filter", ["all", "draft", "sent", "paid", "overdue"], key="inv_filter")
                filtered = invoices_df if status_filter == "all" else invoices_df[invoices_df["status"] == status_filter]
                if not filtered.empty:
                    st.dataframe(filtered, width="stretch")
                else:
                    st.info("No invoices match filter")
            else:
                st.info("No invoices yet")
    
    
def tab_cash_flow():
    st.subheader("💵 Cash Flow Management")
    if not check_permission("manager"):
        st.warning("Only managers and owners can manage cash flow")
    else:
        cf_tab1, cf_tab2 = st.tabs(["Record Cash Flow", "Cash Flow Statement"])
        with cf_tab1:
            col1, col2 = st.columns(2)
            with col1:
                cf_date = st.date_input("Date", value=date.today(), key="cf_date")
                cf_type = st.selectbox("Type", ["income", "expense", "transfer"], key="cf_type")
                cf_category = st.selectbox("Category", ["sales", "purchases", "salaries", "rent", "utilities", "tax", "loan", "other"], key="cf_category")
            with col2:
                cf_amount = st.number_input("Amount (KES)", min_value=0.0, step=100.0, key="cf_amount")
                cf_desc = st.text_input("Description", key="cf_desc")
                cf_ref = st.text_input("Reference (optional)", key="cf_ref")
            if st.button("Record Cash Flow", type="primary", key="add_cf"):
                if cf_amount > 0 and cf_desc:
                    ok, msg = add_cash_flow(st.session_state.user, cf_date.strftime("%Y-%m-%d"), cf_desc, cf_amount, cf_type, cf_category, cf_ref)
                    if ok:
                        st.success(msg)
                        st.session_state.pop("cf_desc", None)
                        st.session_state.pop("cf_amount", None)
                        st.session_state.pop("cf_ref", None)
                        st.rerun()
                    else:
                        st.error(msg)
            start_date = st.date_input("Start Date", value=date.today() - timedelta(days=30), key="cf_start")
            end_date = st.date_input("End Date", value=date.today(), key="cf_end")
            cf_df = get_cash_flow(st.session_state.user, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            if not cf_df.empty:
                summary = get_cash_flow_summary(st.session_state.user, start_date.year, start_date.month)
                if not summary.empty:
                    for _, row in summary.iterrows():
                        st.metric(f"Total {row['flow_type'].title()}", f"KES {row['total']:,.2f}")
                st.dataframe(cf_df, width="stretch")
            else:
                st.info("No cash flow records for selected period")
    
    
def tab_payroll():
    st.subheader("👔 Payroll Management")
    if not check_permission("manager"):
        st.warning("Only managers and owners can manage payroll")
    else:
        payroll_tab1, payroll_tab2 = st.tabs(["Add Payroll", "Payroll History"])
        with payroll_tab1:
            col1, col2 = st.columns(2)
            with col1:
                emp_name = st.text_input("Employee Name", key="emp_name")
                emp_id = st.text_input("Employee ID", key="emp_id")
                department = st.selectbox("Department", ["sales", "admin", "finance", "operations", "it", "other"], key="emp_dept")
                basic = st.number_input("Basic Salary (KES)", min_value=0.0, step=100.0, key="emp_basic")
            with col2:
                allowances = st.number_input("Allowances (KES)", min_value=0.0, step=100.0, value=0.0, key="emp_allow")
                deductions = st.number_input("Deductions (KES)", min_value=0.0, step=100.0, value=0.0, key="emp_deduct")
                net = basic + allowances - deductions
                st.metric("Net Pay", f"KES {net:,.2f}")
                pay_date = st.date_input("Payment Date", value=date.today(), key="pay_date")
                pay_method = st.selectbox("Payment Method", ["bank", "mpesa", "cash", "cheque"], key="pay_method")
            if st.button("Record Payroll", type="primary", key="add_payroll"):
                if emp_name and basic > 0:
                    ok, msg = add_payroll(st.session_state.user, emp_name, emp_id, department, basic, allowances, deductions, net, pay_date.strftime("%Y-%m-%d"), pay_method)
                    if ok:
                        st.success(f"✅ Payroll recorded for {emp_name}")
                        st.session_state.pop("emp_name", None)
                        st.session_state.pop("emp_id", None)
                        st.session_state.pop("emp_basic", None)
                        st.session_state.pop("emp_allow", None)
                        st.session_state.pop("emp_deduct", None)
                        st.rerun()
                    else:
                        st.error(msg)
                else:
                    st.warning("Employee name and basic salary required")
        with payroll_tab2:
            payroll_df = get_payroll(st.session_state.user)
            if not payroll_df.empty:
                st.dataframe(payroll_df, width="stretch")
            else:
                st.info("No payroll records yet")
    
    
def tab_tax():
    st.subheader("🏛️ Tax Management")
    if not check_permission("accountant"):
        st.warning("Only accountants, managers, and owners can manage taxes")
    else:
        tax_tab1, tax_tab2 = st.tabs(["Record Tax", "Tax History"])
        with tax_tab1:
            col1, col2 = st.columns(2)
            with col1:
                tax_type = st.selectbox("Tax Type", ["income_tax", "vat", "paye", "withholding_tax", "customs", "other"], key="tax_type")
                tax_period = st.text_input("Period (e.g. 2026-05)", key="tax_period")
                tax_amount = st.number_input("Amount (KES)", min_value=0.0, step=100.0, key="tax_amount")
                due_date = st.date_input("Due Date", value=date.today() + timedelta(days=30), key="tax_due")
            with col2:
                tax_status = st.selectbox("Status", ["pending", "filed", "paid", "overdue"], key="tax_status")
                receipt_path = st.text_input("Receipt/Reference (optional)", key="tax_receipt")
            if st.button("Save Tax Record", type="primary", key="add_tax"):
                if tax_period and tax_amount > 0:
                    ok, msg = add_tax_record(st.session_state.user, tax_type, tax_period, tax_amount, due_date.strftime("%Y-%m-%d"), tax_status, receipt_path)
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
                else:
                    st.warning("Period and amount required")
        with tax_tab2:
            tax_df = get_tax_records(st.session_state.user)
            if not tax_df.empty:
                st.dataframe(tax_df, width="stretch")
            else:
                st.info("No tax records yet")
    

TAB_FUNCTIONS = {
    "📝 Daily Entry": tab_daily_entry,
    "📦 Inventory": tab_inventory,
    "👥 Debtors/Credit": tab_debtors_credit,
    "🏭 Suppliers": tab_suppliers,
    "✅ Expense Approval": tab_expense_approval,
    "📊 Reports & Analytics": tab_reports_analytics,
    "💰 Budgets": tab_budgets,
    "👥 User Management": tab_user_management,
    "🧾 Invoices": tab_invoices,
    "💵 Cash Flow": tab_cash_flow,
    "👔 Payroll": tab_payroll,
    "🏛️ Tax": tab_tax,
}
    
role = st.session_state.get("user_role", "cashier")
visible_tabs = [tab for tab, required_role in TAB_ROLES.items() if ROLE_LEVEL.get(role, 1) >= ROLE_LEVEL.get(required_role, 1)]
main_tabs = st.tabs(visible_tabs)
    
for i, tab_name in enumerate(visible_tabs):
    with main_tabs[i]:
        TAB_FUNCTIONS[tab_name]()
    

@st.dialog("💬 Messages")
def messages_dialog():
    if "msg_active_tab" not in st.session_state:
        st.session_state.msg_active_tab = "Inbox"

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📥 Inbox", use_container_width=True, type="primary" if st.session_state.msg_active_tab == "Inbox" else "secondary"):
            st.session_state.msg_active_tab = "Inbox"
            st.rerun()
    with col2:
        if st.button("📤 Sent", use_container_width=True, type="primary" if st.session_state.msg_active_tab == "Sent" else "secondary"):
            st.session_state.msg_active_tab = "Sent"
            st.rerun()
    with col3:
        if st.button("✉️ Send", use_container_width=True, type="primary" if st.session_state.msg_active_tab == "Send" else "secondary"):
            st.session_state.msg_active_tab = "Send"
            st.rerun()
    st.divider()

    if st.session_state.msg_active_tab == "Inbox":
        render_inbox_tab()
    elif st.session_state.msg_active_tab == "Sent":
        render_sent_tab()
    elif st.session_state.msg_active_tab == "Send":
        render_send_tab()

    if st.button("Close", key="close_msg_dialog"):
        st.session_state.messages_open = False
        st.session_state.reply_to = ""
        st.session_state.reply_subject = ""
        st.session_state.reply_group_id = None
        st.session_state.msg_active_tab = "Inbox"
        st.rerun()


def render_inbox_tab():
    st.subheader("Inbox")
    inbox_df = get_inbox(st.session_state.user)
    search_inbox = st.text_input("🔍 Search messages...", key="search_inbox")
    if inbox_df is not None and not inbox_df.empty:
        if search_inbox:
            inbox_df = inbox_df[
                inbox_df["subject"].str.contains(search_inbox, case=False, na=False) |
                inbox_df["body"].str.contains(search_inbox, case=False, na=False) |
                inbox_df["from_user"].str.contains(search_inbox, case=False, na=False)
            ]
        if not inbox_df.empty:
            st.caption(f"**{len(inbox_df)} message(s)**")
            for _, row in inbox_df.iterrows():
                msg_icon = "🟢" if not row["is_read"] else "⚪"
                with st.container(border=True):
                    st.caption(f"{msg_icon} From: **{row['from_user']}** | {row['created_at']}")
                    st.write(f"**{row['subject']}**")
                    group_id = row.get("message_group_id")
                    if pd.notna(group_id) and group_id:
                        tagged = get_message_recipients(group_id, st.session_state.user)
                        if tagged:
                            st.caption(f"🏷️ Also sent to: {', '.join(tagged)}")
                    with st.expander("View message"):
                        st.write(f">{row['body']}")
                        col_r, col_m = st.columns(2)
                        with col_r:
                            if st.button("↩️ Reply", key=f"reply_{row['id']}"):
                                st.session_state.reply_to = row["from_user"]
                                st.session_state.reply_subject = f"Re: {row['subject']}"
                                st.session_state.reply_group_id = group_id if pd.notna(group_id) else None
                                st.session_state.msg_active_tab = "Send"
                                st.rerun()
                        with col_m:
                            if not row["is_read"]:
                                if st.button("Mark as read", key=f"read_{row['id']}"):
                                    mark_message_read(row["id"])
        else:
            st.info("No messages match your search")
    else:
        st.info("No messages")


def render_sent_tab():
    st.subheader("Sent Messages")
    sent_df = get_sent(st.session_state.user)
    if sent_df is not None and not sent_df.empty:
        st.caption(f"**{len(sent_df)} sent message(s)**")
        for _, row in sent_df.iterrows():
            with st.container(border=True):
                st.caption(f"To: **{row['to_user']}** | {row['created_at']}")
                st.write(f"**{row['subject']}**")
                with st.expander("View message"):
                    st.write(f">{row['body']}")
    else:
        st.info("No sent messages")


def render_send_tab():
    st.subheader("📤 Send New Message")
    users_list = get_all_users_for_messages(st.session_state.user)
    if users_list:
        reply_to = st.session_state.get("reply_to", "")
        reply_subject = st.session_state.get("reply_subject", "")
        reply_group_id = st.session_state.get("reply_group_id", None)
        try:
            reply_idx = next((i for i, u in enumerate(users_list) if u["username"] == reply_to), None)
        except Exception:
            reply_idx = None
        selected_displays = st.multiselect("To (select one or more)", [u["display"] for u in users_list], default=[users_list[reply_idx]["display"]] if reply_idx is not None else [], key="msg_to_users")
        subject = st.text_input("Subject", value=reply_subject, key="msg_subject")
        body = st.text_area("Message", key="msg_body")
        recipients = [next((u["username"] for u in users_list if u["display"] == d), d) for d in selected_displays]
        if st.button("Send", key="send_msg", type="primary"):
            if subject and body and recipients:
                ok, msg, _ = send_message(st.session_state.user, recipients, subject, body, message_group_id=reply_group_id)
                if ok:
                    st.session_state.reply_to = ""
                    st.session_state.reply_subject = ""
                    st.session_state.reply_group_id = None
                    st.session_state.msg_active_tab = "Inbox"
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
            else:
                st.warning("Subject, message, and at least one recipient required")
    else:
        st.info("No users available to message")


if st.session_state.messages_open:
    messages_dialog()


# ============================================================
# FIXED AI ASSISTANT DIALOG - WITH TIMEOUT AND FALLBACK
# ============================================================
@st.dialog("🤖 AI Financial Assistant")
def chat_dialog():
    # Display conversation history
    for role, msg in st.session_state.chat_history[-20:]:
        with st.chat_message(role):
            st.write(msg)

    query = st.chat_input("Ask about your business...")

    if query:
        st.session_state.chat_history.append(("user", query))
        current_data = get_combined_transaction_data(st.session_state.user)

        with st.spinner("Analyzing..."):
            try:
                # Use a short timeout (10 seconds)
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        generate_response,
                        query,
                        current_data,
                        st.session_state.user,
                        st.session_state.chat_history,
                        st.session_state.feedback_mode,
                    )
                    response = future.result(timeout=30)
            except FutureTimeoutError:
                print("⚠️ AI response timed out after 10 seconds")
                response = "⏳ The assistant took too long. Please try again or check your Ollama server."
            except Exception as e:
                print(f"❌ AI error: {e}")
                response = f"⚠️ Error: {str(e)}"

        # Guarantee a non-empty string
        if not response or not isinstance(response, str):
            response = "I couldn't process that request. Please try again."

        st.session_state.chat_history.append(("assistant", response))
        st.rerun()

    if st.button("Close Assistant", key="close_chat_dialog"):
        st.session_state.chat_open = False
        st.rerun()


if st.session_state.chat_open:
    chat_dialog()