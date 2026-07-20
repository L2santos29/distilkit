#!/bin/bash
# DistilKit GUI Launcher
# Opens the web-based GUI (FastAPI + Tailwind CSS) in your browser.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "⚡ DistilKit — Starting Web GUI..."
echo ""

# ─── Virtual environment ──────────────────────────────────────
VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# ─── Dependencies ─────────────────────────────────────────────
echo "📦 Installing dependencies..."
pip install --quiet --upgrade pip
pip install -q -r requirements.txt

# ─── Launch ───────────────────────────────────────────────────
echo ""
echo "🚀 Opening http://localhost:7860"
echo "   Press Ctrl+C to stop."
echo ""

python3 -m src.webapp
