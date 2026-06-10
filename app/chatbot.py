# app/chatbot.py

import pandas as pd
from typing import Optional, Dict, Any

from app.financial_ai import full_analysis, financial_advice, calculate_metrics
from app.memory import add_message
from app.retriever import retrieve_context, format_transactions
from app.llm import ask_ollama, is_ollama_reachable, FORCE_OFFLINE
from app.intent import classify_intent
from app.anomaly import detect_anomalies
from app.predictor import predict_next_spending
from app.agent import run_agent
from app.categorizer import save_feedback


def safe_dataframe(df):
    if df is None:
        return pd.DataFrame()
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    return df.copy()


def format_memory(context):
    if not context:
        return ""
    lines = []
    for item in context[-10:]:
        role = item.get("role", "user")
        msg = item.get("message", "")
        lines.append(f"{role}: {msg}")
    return "\n".join(lines)


def handle_feedback(query, history=None):
    q = query.lower()
    positive = ["yes", "correct", "right"]
    negative = ["no", "wrong", "incorrect"]
    if any(word in q for word in positive):
        return "👍 Thanks! The category has been confirmed."
    if any(word in q for word in negative):
        categories = ["food", "transport", "shopping", "bills", "entertainment", "healthcare", "education", "subscriptions", "income", "other"]
        corrected = next((c for c in categories if c in q), None)
        if corrected is None:
            return "❓ Please specify the correct category.\n\nExample: `wrong category should be food`"
        last_user_msg = ""
        if history:
            for role, msg in reversed(history):
                if role == "user":
                    last_user_msg = msg
                    break
        if last_user_msg:
            try:
                save_feedback(last_user_msg, corrected)
            except Exception:
                pass
        return f"✅ Updated category to '{corrected}'. I'll use this feedback to improve future predictions."
    return "❓ Feedback mode is enabled.\nReply with:\n- 'yes' if the category is correct\n- 'wrong food' to correct it"


def format_anomalies(anomalies_df, limit=5):
    if anomalies_df.empty:
        return "✅ No anomalies detected."
    lines = []
    for _, row in anomalies_df.head(limit).iterrows():
        amount = row.get("amount", 0)
        category = row.get("category", "unknown")
        explanation = row.get("explanation", "Unusual transaction")
        lines.append(f"• {category} → {amount:.2f}\n  {explanation}")
    extra = ""
    if len(anomalies_df) > limit:
        extra = f"\n\n...and {len(anomalies_df) - limit} more anomalies."
    return "⚠️ Detected unusual transactions:\n\n" + "\n\n".join(lines) + extra


def generate_response(query, df, user=None, history=None, enable_feedback=False):
    df = safe_dataframe(df)
    if history is None:
        history = []
    query = str(query).strip()
    if not query:
        return "Please type a message."

    if enable_feedback:
        return handle_feedback(query, history)

    try:
        intent = classify_intent(query)
    except Exception:
        intent = "general"

    # Intent-specific handlers
    if intent == "forecast":
        if df.empty:
            return "⚠️ No transaction data available for forecasting."
        try:
            pred = predict_next_spending(df)
            return f"📈 Predicted next spending: {pred:.2f}"
        except Exception as e:
            return f"⚠️ Forecast error: {str(e)}"

    if intent == "anomaly_check":
        if df.empty:
            return "⚠️ No transaction data available."
        try:
            anomalies = detect_anomalies(df, lookback_days=30)
            return format_anomalies(anomalies)
        except Exception as e:
            return f"⚠️ Anomaly detection failed: {str(e)}"

    if intent == "advice":
        if df.empty:
            return "⚠️ No financial data available."
        try:
            analysis = full_analysis(df)
            advice_list = analysis.get("optimization", {}).get("recommendations", [])
            if not advice_list:
                advice_list = financial_advice(df, analysis)
            if not advice_list:
                return "No financial advice available yet."
            return "💡 Financial Advice:\n\n" + "\n".join(f"• {a}" for a in advice_list)
        except Exception as e:
            return f"⚠️ Advice generation failed: {str(e)}"

    if intent == "report":
        if df.empty:
            return "⚠️ No transactions found."
        try:
            text = format_transactions(df)
            return f"📊 Recent Transactions:\n\n{text}"
        except Exception as e:
            return f"⚠️ Report generation failed: {str(e)}"

    # Try rule-based agent for budget commands
    if intent in ("set_budget", "check_budget", "budget_summary"):
        try:
            agent_response = run_agent(query, user, df)
            if agent_response and isinstance(agent_response, str) and not agent_response.startswith("❌"):
                if user:
                    try:
                        add_message(user, "user", query)
                        add_message(user, "assistant", agent_response)
                    except Exception:
                        pass
                return agent_response
        except Exception:
            pass

    # Prepare context for LLM
    try:
        retrieved_df = retrieve_context(df, query, top_k=3, spending_top_k=2, recent_top_k=2)
        retrieved_text = format_transactions(retrieved_df)
    except Exception:
        retrieved_text = "No matching transactions found."

    memory_text = ""
    if user:
        try:
            from app.memory import get_context
            memory_context = get_context(user)
            memory_text = format_memory(memory_context)
        except Exception:
            pass

    metrics = calculate_metrics(df) if not df.empty else {}
    total = metrics.get("total", 0)
    top_cat = metrics.get("top_category", "unknown")
    summary = f"Total spending: {total:.2f}. Top category: {top_cat}."

    system_prompt = f"""You are FinanceIQ AI Assistant.

Financial summary: {summary}
Recent transactions: {retrieved_text[:300]}
Conversation: {memory_text[:200]}
Answer concisely based on the data above.
"""

    # Try LLM (Ollama) if reachable
    if not FORCE_OFFLINE and is_ollama_reachable():
        try:
            llm_answer = ask_ollama(query, system=system_prompt, timeout=25)
            if llm_answer:
                if user:
                    try:
                        add_message(user, "user", query)
                        add_message(user, "assistant", llm_answer)
                    except Exception:
                        pass
                return llm_answer
        except Exception:
            pass

    # Simple fallback for when LLM unavailable
    if metrics.get("total", 0) > 0:
        return f"🤖 FinanceIQ Summary\n\n• Total: {total:.2f}\n• Top: {top_cat}\n\n{retrieved_text}"
    return "🤖 No data available. Add transactions in Daily Entry tab."