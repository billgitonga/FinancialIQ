#!/bin/bash
# Terminate all FinancialIQ processes

echo "Stopping FinancialIQ services..."
pkill -f 'streamlit run dashboard.py' 2>/dev/null
pkill -f 'streamlit' 2>/dev/null
pkill -f 'FINANCI' 2>/dev/null
echo "Done. All processes terminated."