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

echo "📦 Installing dependencies..."
pip install --quiet --upgrade pip
pip install -q -r requirements.txt

echo ""
echo "🚀 Running distillation via CLI..."
echo ""

# Use the CLI with selected options
python -m src.cli train \
    --teacher resnet18 \
    --epochs 5 \
    --temperature 4.0 \
    --alpha 0.7 \
    --batch-size 64 \
    --export onnx \
    --benchmark cpu
