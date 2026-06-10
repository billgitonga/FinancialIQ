#!/bin/bash

# ============================================
# FinanceIQ - Complete Setup Script for Arch Linux
# ============================================

set -e  # Exit on any error

echo "=========================================="
echo "  FinanceIQ Setup for Arch Linux"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# -------------------------------------------
# 1. System Update
# -------------------------------------------
echo -e "${YELLOW}[1/8] Updating system packages...${NC}"
sudo pacman -Syu --noconfirm

# -------------------------------------------
# 2. Fix Locale (IMPORTANT for PostgreSQL)
# -------------------------------------------
echo -e "${YELLOW}[2/8] Fixing locale settings...${NC}"
sudo sed -i 's/^#en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen
sudo locale-gen
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
echo -e "${GREEN}Locale fixed!${NC}"

# Make locale permanent
if ! grep -q "LANG=en_US.UTF-8" ~/.bashrc; then
    echo 'export LANG=en_US.UTF-8' >> ~/.bashrc
    echo 'export LC_ALL=en_US.UTF-8' >> ~/.bashrc
fi

# -------------------------------------------
# 3. Install System Dependencies
# -------------------------------------------
echo -e "${YELLOW}[3/8] Installing system dependencies...${NC}"
sudo pacman -S --noconfirm \
    python \
    python-pip \
    python-virtualenv \
    tesseract \
    tesseract-data-eng \
    poppler \
    postgresql \
    git \
    base-devel

# -------------------------------------------
# 4. Setup PostgreSQL
# -------------------------------------------
echo -e "${YELLOW}[4/8] Setting up PostgreSQL...${NC}"
echo "Do you want to setup PostgreSQL? (y/n)"
read -r setup_pg

if [ "$setup_pg" = "y" ] || [ "$setup_pg" = "Y" ]; then
    # Initialize PostgreSQL if not already done
    if [ ! -d "/var/lib/postgres/data" ]; then
        echo "Initializing PostgreSQL database..."
        sudo -u postgres initdb -D /var/lib/postgres/data
    fi
    
    # Start and enable PostgreSQL
    sudo systemctl enable postgresql
    sudo systemctl start postgresql
    
    # Wait for PostgreSQL to start
    sleep 2
    
    # Create database and user
    echo "Creating PostgreSQL database and user..."
    sudo -u postgres psql -c "CREATE USER financeiq WITH PASSWORD 'financeiq123';" 2>/dev/null || echo "User may already exist"
    sudo -u postgres psql -c "CREATE DATABASE financeiq OWNER financeiq;" 2>/dev/null || echo "Database may already exist"
    sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE financeiq TO financeiq;" 2>/dev/null
    
    echo -e "${GREEN}PostgreSQL setup complete!${NC}"
    echo "Database: financeiq"
    echo "User: financeiq"
    echo "Password: financeiq123"
    echo ""
else
    echo -e "${YELLOW}Skipping PostgreSQL setup. Will use SQLite.${NC}"
fi

# -------------------------------------------
# 5. Create Project Directory
# -------------------------------------------
echo -e "${YELLOW}[5/8] Creating project directory...${NC}"
PROJECT_DIR="$HOME/Projects/Programming/Python/FinancialIQ"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# -------------------------------------------
# 6. Setup Python Virtual Environment
# -------------------------------------------
echo -e "${YELLOW}[6/8] Setting up Python virtual environment...${NC}"
python -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# -------------------------------------------
# 7. Create requirements.txt and Install
# -------------------------------------------
echo -e "${YELLOW}[7/8] Installing Python packages...${NC}"
cat > requirements.txt << 'EOF'
joblib>=1.4.2
numpy>=1.26.4
pandas>=2.2.2
pdf2image>=1.17.0
Pillow>=10.4.0
pytesseract>=0.3.13
python-dotenv>=1.0.1
bcrypt>=4.1.3
scikit-learn>=1.5.1
SQLAlchemy>=2.0.31
streamlit>=1.37.1
requests>=2.32.3
opencv-python>=4.10.0
psycopg2-binary>=2.9.9
openpyxl>=3.1.2
EOF

# Install Python dependencies
pip install -r requirements.txt

# -------------------------------------------
# 8. Create .env file and Project Structure
# -------------------------------------------
echo -e "${YELLOW}[8/8] Creating project configuration...${NC}"

if [ "$setup_pg" = "y" ] || [ "$setup_pg" = "Y" ]; then
    cat > .env << 'EOF'
# FinanceIQ Environment Configuration

# PostgreSQL connection
DATABASE_URL=postgresql://financeiq:financeiq123@localhost:5432/financeiq

# Ollama configuration (optional - for open-source LLM)
OLLAMA_URL=http://localhost:11434/api/generate
OLLAMA_MODEL=llama3

# App settings
DEBUG=False
SECRET_KEY=financeiq-secret-key-change-in-production-2024
EOF
else
    cat > .env << 'EOF'
# FinanceIQ Environment Configuration

# SQLite connection (no PostgreSQL needed)
DATABASE_URL=sqlite:///finance.db

# Ollama configuration (optional - for open-source LLM)
OLLAMA_URL=http://localhost:11434/api/generate
OLLAMA_MODEL=llama3

# App settings
DEBUG=False
SECRET_KEY=financeiq-secret-key-change-in-production-2024
EOF
fi

echo -e "${GREEN}.env file created!${NC}"

# Create necessary directories
echo "Creating project directories..."
mkdir -p app data models memory_store cache_store

# Create sample CSV file
echo "Creating sample data file..."
cat > data/sample.csv << 'EOF'
date,description,amount
2024-01-05,Rent payment,-1200.00
2024-01-07,Uber ride,-25.50
2024-01-10,Restaurant dinner,-45.00
2024-01-12,Electricity bill,-89.00
2024-01-15,Netflix subscription,-15.99
2024-01-18,Grocery shopping,-120.50
2024-01-20,Amazon purchase,-67.99
2024-01-22,Coffee shop,-4.50
2024-01-25,Phone bill,-75.00
2024-01-28,Movie tickets,-24.00
2024-02-01,Rent payment,-1200.00
2024-02-03,Bus fare,-3.00
2024-02-08,Lunch at KFC,-12.50
2024-02-12,Internet bill,-65.00
2024-02-15,Shopping mall,-150.00
2024-02-18,Spotify premium,-9.99
2024-02-20,Fuel,-55.00
2024-02-22,Water bill,-35.00
2024-02-25,Pizza delivery,-22.00
2024-02-28,Game purchase,-59.99
EOF

echo -e "${GREEN}Sample CSV created!${NC}"

# -------------------------------------------
# Done!
# -------------------------------------------
echo ""
echo "=========================================="
echo -e "${GREEN}  Setup Complete! 🎉${NC}"
echo "=========================================="
echo ""
echo "Project location: $PROJECT_DIR"
echo ""
echo "To get started:"
echo "  cd $PROJECT_DIR"
echo "  source venv/bin/activate"
echo "  streamlit run dashboard.py"
echo ""
echo "Optional - Install Ollama for AI chat:"
echo "  curl -fsSL https://ollama.ai/install.sh | sh"
echo "  ollama serve &"
echo "  ollama pull llama3"
echo ""
echo "Then open http://localhost:8501 in your browser"
echo "=========================================="