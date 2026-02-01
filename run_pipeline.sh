#!/bin/bash
# Quick start script to run the entire pipeline

set -e  # Exit on error

echo "=========================================="
echo "AI Models Testing Pipeline"
echo "=========================================="
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate venv
source venv/bin/activate

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found!"
    echo "Creating from template..."
    cp .env.example .env
    echo ""
    echo "❌ Please edit .env and add your API keys, then run this script again"
    exit 1
fi

echo "Step 1: Extracting questions from PDFs..."
echo "=========================================="
python scripts/1_extract_questions.py
echo ""

echo "Step 2: Testing models..."
echo "=========================================="
python scripts/2_test_models.py
echo ""

echo "Step 3: Evaluating responses..."
echo "=========================================="
python scripts/3_evaluate.py
echo ""

echo "✅ Pipeline complete!"
echo ""
echo "📊 Check results/comparison.xlsx for the full comparison"
