# 🎯 AI Models Testing Framework - Summary

## ✅ What We Built

A complete automated pipeline to test and compare different AI models on Polish matura exam questions.

## 📦 Project Structure

```
models_test/
│
├── 📄 Documentation
│   ├── README.md          - Project overview
│   ├── GUIDE.md           - Complete usage guide
│   └── .env.example       - API keys template
│
├── 📂 PDFs (Input)
│   ├── polski_2025.pdf           - Polish exam (24 pages, ~25 questions)
│   ├── polski_2025_odp.pdf       - Polish answers (18 pages)
│   ├── matematyka_2025.pdf       - Math exam (large file)
│   └── matematyka_2025_odp.pdf   - Math answers (40 pages)
│
├── 🔧 Scripts (Pipeline)
│   ├── utils.py                  - Database & PDF utilities
│   ├── 1_extract_questions.py    - AI-powered PDF parsing
│   ├── 2_test_models.py          - Multi-model testing
│   └── 3_evaluate.py             - LLM-as-judge evaluation
│
├── 🚀 Quick Start
│   └── run_pipeline.sh           - One-command execution
│
├── 📊 Output (Generated)
│   ├── data/
│   │   ├── questions.db          - SQLite database
│   │   ├── *_raw.json            - Raw PDF extractions
│   │   └── *_questions.json      - Structured questions
│   └── results/
│       └── comparison.xlsx       - Final Excel report
│
└── 🐍 Environment
    ├── venv/                     - Virtual environment (ready)
    └── requirements.txt          - All dependencies installed
```

## 🔄 Pipeline Flow

```
┌─────────────────┐
│  PDF Files      │
│  (4 PDFs)       │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Step 1: Extract Questions      │
│  • Uses Claude 3.5 Haiku        │
│  • Parses messy PDFs            │
│  • Matches Q&A pairs            │
│  • Cost: ~$1-3                  │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  SQLite Database                │
│  • questions                    │
│  • model_responses              │
│  • evaluations                  │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Step 2: Test Models            │
│  • Claude 3.5 Haiku & Sonnet    │
│  • GPT-4o & GPT-4o-mini         │
│  • Gemini 1.5 Flash             │
│  • Records: response, latency,  │
│    tokens, cost                 │
│  • Cost: ~$6-10                 │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Step 3: Evaluate               │
│  • LLM-as-judge (Claude Haiku)  │
│  • Scores each response         │
│  • Generates Excel report       │
│  • Cost: ~$2-3                  │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Excel Report                   │
│  • Comparison sheet             │
│  • Summary statistics           │
│  • Raw data                     │
└─────────────────────────────────┘
```

## 🎮 How to Use

### First Time Setup

```bash
# 1. You're already in the right directory with venv activated
cd /home/hope/maturex/models_test
source venv/bin/activate

# 2. Create .env file with your API keys
cp .env.example .env
nano .env  # Add your keys

# 3. Run the pipeline
./run_pipeline.sh
```

### Step-by-Step Execution

```bash
# Activate environment
source venv/bin/activate

# Step 1: Extract questions from PDFs
python scripts/1_extract_questions.py

# Step 2: Test all models
python scripts/2_test_models.py

# Step 3: Evaluate and generate report
python scripts/3_evaluate.py

# Check results
open results/comparison.xlsx  # or use LibreOffice
```

## 📊 Expected Output

### Console Output
- Progress bars for each step
- Cost tracking
- Question counts
- Model performance summaries

### Files Generated
1. **data/questions.db** - SQLite database with all data
2. **data/*_raw.json** - Raw PDF text for debugging
3. **data/*_questions.json** - Structured questions
4. **results/comparison.xlsx** - Main deliverable

### Excel Report Contents

**Comparison Sheet:**
| Exam | Q# | Question | Correct Answer | Model A Response | Model A Score | Model B Response | ... |
|------|----|-----------|--------------------|------------------|---------------|------------------|-----|

**Summary Sheet:**
| Model | Accuracy | Avg Score | Avg Latency | Total Cost | Cost/Question |
|-------|----------|-----------|-------------|------------|---------------|

## 💰 Budget Breakdown

For ~60 questions (30 per exam × 2 exams):

| Component | Cost |
|-----------|------|
| Question extraction (Claude Haiku) | $1-3 |
| Testing 5 models × 60 questions | $6-10 |
| Evaluation (LLM-as-judge) | $2-3 |
| **Total** | **$9-16** |

Well within your $20 budget! 🎉

## 🔑 Key Features

✅ **Automated PDF Parsing** - AI handles messy exam formats
✅ **Multi-Model Testing** - Compare 5 different AI models
✅ **Cost Tracking** - Know exactly what you're spending
✅ **Performance Metrics** - Accuracy, latency, cost per question
✅ **LLM-as-Judge** - Automated evaluation
✅ **Excel Export** - Easy to analyze and share
✅ **SQLite Database** - All data stored for re-analysis
✅ **Modular Design** - Easy to add more models or exams

## 🎯 Next Steps

1. **Get API Keys** from Anthropic, OpenAI, and Google
2. **Add them to .env** file
3. **Run the pipeline**: `./run_pipeline.sh`
4. **Analyze results** in Excel
5. **Iterate** - add more exams or models as needed

## 🐛 Troubleshooting

**Issue**: "No API key found"
- **Fix**: Create `.env` file and add your keys

**Issue**: "No questions in database"
- **Fix**: Run step 1 first: `python scripts/1_extract_questions.py`

**Issue**: "Rate limit error"
- **Fix**: Increase sleep time in `scripts/2_test_models.py`

**Issue**: "PDF parsing failed"
- **Fix**: Check PDFs are valid: `file pdfs/*.pdf`

## 📚 Documentation

- **README.md** - Quick project overview
- **GUIDE.md** - Detailed usage guide (recommended read!)
- **This file** - Quick reference summary

## 🎓 What Makes This Approach Good?

1. **AI-Powered Extraction** - No manual parsing needed
2. **Comprehensive Testing** - Tests multiple models at once
3. **Objective Evaluation** - LLM-as-judge removes bias
4. **Cost Efficient** - Uses cheap models where possible
5. **Reproducible** - Database stores everything
6. **Extensible** - Easy to add more models/exams
7. **Well Documented** - Clear guides and comments

## 🚀 Ready to Start?

```bash
source venv/bin/activate
cp .env.example .env
# Edit .env with your API keys
./run_pipeline.sh
```

Good luck! 🎉
