#!/bin/bash
# DistilKit quick start script
set -e

echo "⚡ DistilKit — Knowledge Distillation Framework"
echo "==============================================="
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "❌ Python 3 is required"
    exit 1
fi

# Check venv
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install -q -r requirements.txt

echo ""
echo "🚀 Running basic distillation example..."
echo ""

python examples/basic_classifier.py
