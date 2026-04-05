# AI Models Testing - Complete Guide

## 📋 Overview

This project tests different AI models on Polish matura exam questions to compare their performance, accuracy, and cost-effectiveness.

## 🎯 What It Does

1. **Extracts** questions and answers from exam PDFs using AI
2. **Tests** multiple AI models (Claude, GPT, Gemini) on each question
3. **Evaluates** responses using LLM-as-judge
4. **Generates** Excel comparison report with statistics

## 🏗️ Architecture

```
PDF Exams → AI Parser → SQLite DB → Model Testing → Evaluation → Excel Report
```

### Models Tested

| Model | Provider | Cost (per 1M tokens) | Speed |
|-------|----------|---------------------|-------|
| Claude 3.5 Haiku | Anthropic | $0.25 / $1.25 | Fast |
| Claude 3.5 Sonnet | Anthropic | $3.00 / $15.00 | Medium |
| GPT-4o-mini | OpenAI | $0.15 / $0.60 | Fast |
| GPT-4o | OpenAI | $2.50 / $10.00 | Medium |
| Gemini 1.5 Flash | Google | $0.075 / $0.30 | Fast |

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Activate virtual environment (already created)
source venv/bin/activate

# Create .env file with your API keys
cp .env.example .env
nano .env  # Add your API keys
```

### 2. Get API Key

- **OpenRouter**: https://openrouter.ai/keys

OpenRouter provides unified access to all models (Anthropic Claude, OpenAI GPT, Google Gemini, and many more) through a single API.

### 3. Run Pipeline

**Option A: Run everything at once**
```bash
./run_pipeline.sh
```

**Option B: Run step by step**
```bash
# Step 1: Extract questions (uses OpenAI GPT-4o-mini via OpenRouter, ~$1-3)
python scripts/1_extract_questions.py

# Step 2: Test models (cost varies by models selected, ~$6-10)
python scripts/2_test_models.py

# Step 3: Evaluate and generate report (~$2-3)
python scripts/3_evaluate.py
```

## 💰 Cost Estimates

For ~60 questions (30 per exam × 2 exams):

| Step | Description | Estimated Cost |
|------|-------------|----------------|
| 1 | Question extraction | $1-3 |
| 2 | Model testing (5 models) | $6-10 |
| 3 | Evaluation | $2-3 |
| **Total** | | **$9-16** |

## 📊 Output

### Excel Report: `results/comparison.xlsx`

**Sheet 1: Comparison**
- Side-by-side model responses
- Scores and correctness indicators
- Latency and cost per question

**Sheet 2: Summary**
- Overall accuracy per model
- Average latency
- Total and per-question costs
- Performance rankings

**Sheet 3: Raw Data**
- Complete dataset for custom analysis

### Database: `data/questions.db`

SQLite database with three tables:
- `questions` - Extracted Q&A pairs
- `model_responses` - All model responses
- `evaluations` - Scoring and evaluation data

## 🔧 Customization

### Add More Models

Edit `scripts/2_test_models.py` and add to `MODELS` list:

```python
{
    'name': 'your-model-name',
    'model_id': 'provider/model-id',  # e.g., 'anthropic/claude-3-opus'
    'input_cost_per_1m': 0.0,
    'output_cost_per_1m': 0.0
}
```

See available models at: https://openrouter.ai/models

### Add More Exams

1. Place PDFs in `pdfs/` directory.
2. Use the normalized naming scheme below.
3. Run `python scripts/1_extract_questions.py`.

Required naming:

```text
przedmiot_rok.pdf
przedmiot_rok_odp.pdf

przedmiot_rok_roz.pdf
przedmiot_rok_roz_odp.pdf
```

Optional transcript files:

```text
przedmiot_rok_transkrypcja.pdf
przedmiot_rok_roz_transkrypcja.pdf
```

Examples:

```text
polski_2025.pdf
polski_2025_odp.pdf

angielski_2025_roz.pdf
angielski_2025_roz_odp.pdf
angielski_2025_roz_transkrypcja.pdf
```

Step 1 automatically discovers all complete sets where exam PDF and answer PDF are both present.

### Change Evaluation Criteria

Edit `evaluate_response_with_llm()` in `scripts/3_evaluate.py` to customize the evaluation prompt.

## 📁 File Structure

```
models_test/
├── pdfs/                          # Input PDFs
│   ├── polski_2025.pdf
│   ├── polski_2025_odp.pdf
│   ├── matematyka_2025.pdf
│   └── matematyka_2025_odp.pdf
├── data/                          # Generated data
│   ├── questions.db               # Main database
│   ├── polski_2025_raw.json       # Raw extracted text
│   ├── polski_2025_questions.json # Structured questions
│   ├── angielski_2025_roz_raw.json
│   └── angielski_2025_roz_questions.json
│   └── responses/                 # Model responses (future)
├── results/                       # Final outputs
│   ├── comparison.xlsx            # Main report
│   └── comparison.html            # Grouped HTML report
├── scripts/                       # Pipeline scripts
│   ├── utils.py                   # Shared utilities
│   ├── 1_extract_questions.py     # Step 1
│   ├── 2_test_models.py           # Step 2
│   └── 3_evaluate.py              # Step 3
├── venv/                          # Virtual environment
├── .env                           # API keys (not in git)
├── .env.example                   # Template
├── requirements.txt               # Dependencies
├── run_pipeline.sh                # Quick run script
├── README.md                      # Project overview
└── GUIDE.md                       # This file
```

## 🐛 Troubleshooting

### "No questions found in database"
Run step 1 first: `python scripts/1_extract_questions.py`

### "API key not found"
Check your `.env` file has `OPENROUTER_API_KEY` set correctly

### "Rate limit exceeded"
Add delays in `scripts/2_test_models.py` (increase `time.sleep()` value)

### "PDF extraction failed"
Check PDF files are valid with: `file pdfs/*.pdf`

## 📈 Next Steps

1. **Analyze Results**: Open `results/comparison.xlsx`
2. **Identify Best Model**: Check Summary sheet for highest accuracy
3. **Cost Analysis**: Compare accuracy vs. cost trade-offs
4. **Iterate**: Add more exams or models as needed

## 🤝 Contributing

To extend this project:
1. Add more evaluation metrics in step 3
2. Implement human-in-the-loop validation
3. Add visualization (charts, graphs)
4. Create web interface for results

## 📝 Notes

- All responses are evaluated at temperature=0 for consistency
- Evaluation uses Claude 3.5 Haiku via OpenRouter (cheap and reliable)
- Database stores everything for future re-analysis
- Excel report can be customized in `scripts/3_evaluate.py`
- OpenRouter adds a small markup (~10-20%) to base model costs
