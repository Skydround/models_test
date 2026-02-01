# Models Testing Project - Exam Q&A Evaluation

This project extracts questions from Polish exam PDFs and tests different AI models' ability to answer them correctly.

## Project Structure

```
models_test/
├── pdfs/                      # Source PDFs
│   ├── polski_2025.pdf
│   ├── polski_2025_odp.pdf
│   ├── matematyka_2025.pdf
│   └── matematyka_2025_odp.pdf
├── data/                      # Extracted and processed data
│   ├── questions.db          # SQLite database with questions
│   └── responses/            # Model responses
├── scripts/
│   ├── 1_extract_questions.py    # Extract Q&A from PDFs
│   ├── 2_test_models.py          # Run models on questions
│   ├── 3_evaluate.py             # Evaluate responses
│   └── utils.py                  # Shared utilities
├── results/
│   └── comparison.xlsx       # Final comparison spreadsheet
├── venv/                     # Virtual environment
├── requirements.txt
└── README.md
```

## Workflow

1. **Extract Questions** - Parse PDFs using AI to structure Q&A pairs
2. **Test Models** - Run questions through multiple AI models
3. **Evaluate** - Compare responses and score them
4. **Report** - Generate comparison spreadsheet

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Create `.env` file with API keys:
```
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
```

## Usage

```bash
# Step 1: Extract questions from PDFs
python scripts/1_extract_questions.py

# Step 2: Test models
python scripts/2_test_models.py

# Step 3: Evaluate and compare
python scripts/3_evaluate.py
```
