#!/bin/bash

# ============================================
# FinanceIQ - Quick Start Script
# ============================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run ./setup.sh first."
    exit 1
fi

source venv/bin/activate

echo "=========================================="
echo "  Starting FinanceIQ"
echo "=========================================="
echo ""
echo "Open http://localhost:8501 in your browser"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run dashboard.py --server.port 8501 --server.headless true
