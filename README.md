# FinanceIQ

FinanceIQ is an all-in-one SME financial management dashboard. It helps small and medium businesses track daily transactions, manage inventory, monitor debtors/credit, control expenses, generate reports, and interact with an AI financial assistant.

## Features

- **Daily Entry** — quick transaction entry, stock sales, OCR receipt upload, recent transactions view
- **Inventory** — product management, stock adjustments, low-stock alerts
- **Debtors/Credit** — customer credit management, payment recording, debtor reminders
- **Suppliers** — supplier management, payment terms, delivery performance tracking
- **Expense Approval** — submit, review, and approve/reject expense requests (role-based workflow)
- **Reports & Analytics** — year-over-year comparison, top products, accountant export, financial health analysis
- **Budgets** — set and monitor category budgets with progress bars and alerts
- **User Management** — owner-only user creation, approval, blocking, and role assignment
- **In-App Messaging** — send/receive messages between users, multi-recipient support, read/unread tracking
- **AI Financial Assistant** — natural-language queries for budgets, spending, forecasts, and advice
- **OCR Receipts** — upload receipt images/PDFs to auto-extract totals and line items

## Role-Based Access

| Role | Access |
|------|--------|
| Cashier | Daily Entry, Expense Approval, Reports |
| Accountant | Daily Entry, Debtors/Credit, Expense Approval, Reports, Budgets |
| Manager | Daily Entry, Inventory, Debtors/Credit, Suppliers, Expense Approval, Reports, Budgets |
| Owner | All tabs + User Management |

## Tech Stack

- **Frontend:** Streamlit
- **Database:** SQLite (default) / PostgreSQL (optional)
- **AI/LLM:** Ollama (Llama 3 / TinyLlama) with fallback rule-based agent
- **OCR:** Tesseract / pytesseract
- **ML:** scikit-learn (IsolationForest for anomaly detection, SentenceTransformers for intent classification)

## Prerequisites

- Python 3.10+
- pip
- (Optional) PostgreSQL
- (Optional) Ollama for AI chat

## Installation

```bash
git clone https://github.com/billgitonga/FinancialIQ.git
cd FinancialIQ
chmod +x setup.sh
./setup.sh
```

The setup script will:
1. Update system packages
2. Install system dependencies (Python, Tesseract, Poppler, PostgreSQL)
3. Create a Python virtual environment
4. Install Python dependencies from `requirements.txt`
5. Create `.env` configuration and project directories
6. Optionally initialize PostgreSQL

## Configuration

Environment variables are stored in `.env`. Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite:///finance.db` | Database connection string |
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama API endpoint |
| `OLLAMA_MODEL` | `llama3` | LLM model name |
| `OLLAMA_TIMEOUT` | `60` | Request timeout in seconds |
| `OLLAMA_ENABLE_CACHE` | `true` | Enable LLM response caching |
| `FINANCEIQ_FORCE_OFFLINE` | `false` | Disable LLM calls |
| `FINANCEIQ_BUDGET_FILE` | `budget_store.json` | Budget persistence file |

## Running the App

```bash
source venv/bin/activate
streamlit run dashboard.py
```

Then open `http://localhost:8501` in your browser.

## Running Ollama (AI Chat)

```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve &
ollama pull llama3
```

If Ollama is unavailable, the AI assistant falls back to rule-based responses.

## Project Structure

```
FinancialIQ/
├── dashboard.py          # Main Streamlit app
├── setup.sh              # Arch Linux setup script
├── start.sh              # Quick-start launcher
├── requirements.txt      # Python dependencies
├── .env                  # Environment configuration
├── finance.db            # SQLite database (created on first run)
├── budget_store.json     # Budget data
├── app/                  # Application modules
│   ├── auth.py           # Authentication
│   ├── database.py       # Database models and queries
│   ├── agent.py          # Rule-based financial agent
│   ├── chatbot.py        # AI chatbot orchestration
│   ├── llm.py            # Ollama API client
│   ├── intent.py         # Intent classification
│   ├── budget.py         # Budget management
│   ├── anomaly.py        # Anomaly detection
│   ├── health.py         # Business health scoring
│   ├── predictor.py      # Spending prediction
│   ├── reporting.py      # Report generation
│   ├── categorizer.py    # Transaction categorization
│   ├── ocr.py            # Receipt OCR pipeline
│   ├── extractor.py      # Receipt field extraction
│   ├── retriever.py      # Transaction search
│   ├── memory.py         # Conversation memory
│   ├── financial_ai.py   # AI analysis helpers
│   ├── ingestion.py      # Data loading
│   ├── trends.py         # Trend analysis
│   └── pipeline.py       # Main data pipeline
├── logs/                 # Log files
├── receipts/             # Uploaded receipts
└── ocr_cache/            # OCR result cache
```

## License

Proprietary — All rights reserved.
