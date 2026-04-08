#!/usr/bin/env python3
"""
Step 2: Test different AI models on the extracted questions

This script:
1. Loads questions from database
2. Runs each question through multiple AI models
3. Records responses, latency, and costs
"""

import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))
from utils import QuestionDatabase

load_dotenv()

# Initialize OpenRouter client (using OpenAI-compatible interface)
import httpx
client = OpenAI(
    http_client=httpx.Client(timeout=httpx.Timeout(90.0, connect=10.0)),  # hard socket-level timeout
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv('OPENROUTER_API_KEY'),
    default_headers={
        "HTTP-Referer": "https://github.com/models_test",
        "X-Title": "Models Test - Model Evaluation"
    }
)


# Model configurations for OpenRouter
# Format: provider/model-name
# Sorted by cost efficiency (cheapest first)
MODELS = [
    {
        'name': 'step-3.5-flash-free',
        'model_id': 'stepfun/step-3.5-flash:free',
        'input_cost_per_1m': 0.00,
        'output_cost_per_1m': 0.00
    },
    {
        'name': 'deepseek-v3.2',
        'model_id': 'deepseek/deepseek-v3.2',
        'input_cost_per_1m': 0.28,
        'output_cost_per_1m': 0.88
    },
    {
        'name': 'qwen3.5-35b',
        'model_id': 'qwen/qwen3.5-35b-a3b',
        'input_cost_per_1m': 0.10,
        'output_cost_per_1m': 0.30
    },
    {
        'name': 'glm-4.7-nitro',
        'model_id': 'z-ai/glm-4.7:nitro',
        'input_cost_per_1m': 2.25,
        'output_cost_per_1m': 2.75
    },
    {
        'name': 'gemini-3-flash',
        'model_id': 'google/gemini-3-flash-preview',
        'input_cost_per_1m': 0.50,
        'output_cost_per_1m': 3.00
    },
    {
        # hunter-alpha was renamed – now available as mimo-v2-pro
        'name': 'mimo-v2-pro',
        'model_id': 'xiaomi/mimo-v2-pro',
        'input_cost_per_1m': 0.90,
        'output_cost_per_1m': 0.90
    },
]


def normalize_optional_text(value: str | None) -> str | None:
    """Convert placeholder strings to missing values."""
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized or normalized.lower() in {'null', 'none', 'brak'}:
        return None
    return normalized


def build_question_prompt(question: dict) -> str:
    """Build a subject-aware prompt for a single exam task."""
    context_text = normalize_optional_text(question.get('context_text'))
    prompt_parts = [
        "Odpowiedz na następujące pytanie z egzaminu maturalnego.",
        "Odpowiedz zwięźle, precyzyjnie i tylko na podstawie podanego polecenia oraz kontekstu.",
        f"Egzamin: {question.get('exam_name', 'nieznany')}",
        f"Typ zadania: {question.get('question_type', 'unknown')}",
        f"Maksymalna liczba punktów: {question.get('max_points', 'nieznana')}",
    ]

    if context_text:
        prompt_parts.extend([
            "",
            "Kontekst zadania:",
            context_text,
        ])

    prompt_parts.extend([
        "",
        "Pytanie:",
        question['question_text'],
    ])

    return "\n".join(prompt_parts)


def ask_model(model_config: dict, question: dict) -> dict:
    """Ask any model a question via OpenRouter"""
    prompt = build_question_prompt(question)
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        start_time = time.time()
        try:
            response = client.chat.completions.create(
                model=model_config['model_id'],
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=8000,
            )
            break  # success
        except Exception as e:
            wait = 2 ** attempt  # 2s, 4s, 8s
            if attempt < max_retries:
                print(f"\n   ⚠️  Attempt {attempt}/{max_retries} failed ({type(e).__name__}), retrying in {wait}s…")
                time.sleep(wait)
            else:
                raise
    
    latency_ms = int((time.time() - start_time) * 1000)
    
    answer = response.choices[0].message.content
    # Reasoning models (e.g. stepfun, glm) may return content=None when all
    # budget was consumed by thinking tokens.  Fall back to the reasoning text.
    if answer is None:
        msg = response.choices[0].message
        reasoning = getattr(msg, 'reasoning', None)
        if not reasoning and hasattr(msg, 'reasoning_details'):
            parts = [d.get('text', '') for d in (msg.reasoning_details or []) if d.get('text')]
            reasoning = '\n'.join(parts)
        answer = reasoning or '(no response generated)'

    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    total_tokens = response.usage.total_tokens
    
    cost = (input_tokens * model_config['input_cost_per_1m'] + 
            output_tokens * model_config['output_cost_per_1m']) / 1_000_000
    
    return {
        'response': answer,
        'latency_ms': latency_ms,
        'tokens_used': total_tokens,
        'cost_usd': cost
    }


def test_model_on_question(model_config: dict, question: dict) -> dict:
    """Test a single model on a single question"""
    
    try:
        return ask_model(model_config, question)
    
    except Exception as e:
        print(f"\n❌ Error with {model_config['name']}: {e}")
        return {
            'response': f"ERROR: {str(e)}",
            'latency_ms': 0,
            'tokens_used': 0,
            'cost_usd': 0
        }


def main():
    """Main testing workflow"""
    
    print("="*70)
    print("STEP 2: TEST MODELS ON QUESTIONS")
    print("="*70)
    
    # Check API key
    if not os.getenv('OPENROUTER_API_KEY'):
        print("\n❌ ERROR: OPENROUTER_API_KEY not found in environment")
        print("Get your API key from: https://openrouter.ai/keys")
        return
    
    # Load questions from database
    db = QuestionDatabase()
    questions = db.get_all_questions()
    
    if not questions:
        print("\n❌ No questions found in database!")
        print("Run scripts/1_extract_questions.py first")
        return
    
    print(f"\n✅ Loaded {len(questions)} questions from database")
    
    # All models are available through OpenRouter
    available_models = MODELS
    
    print(f"\n🤖 Testing {len(available_models)} models via OpenRouter:")
    for model in available_models:
        print(f"   - {model['name']} ({model['model_id']})")
    
    # Build per-question work: {question -> [models to test]}
    work = {}
    for question in questions:
        pending_models = [
            model for model in available_models
            if not db.response_exists(question['id'], model['name'])
        ]
        if pending_models:
            work[question['id']] = (question, pending_models)

    total_tests = sum(len(m) for _, m in work.values())
    total_cost = 0
    completed = 0
    lock = threading.Lock()

    print(f"\n📊 Total tests to run: {total_tests}")
    print(f"   ({len(questions)} questions × {len(available_models)} models, {len(work)} questions have pending work)")

    if not work:
        print("\n✅ All question/model pairs already have saved responses")
        db.close()
        return

    # How many model calls to run in parallel (one per model, capped at 6)
    MAX_WORKERS = int(os.getenv('TEST_WORKERS', '6'))

    with tqdm(total=total_tests, desc="Testing", unit="req") as pbar:
        for question, models_to_run in work.values():
            current_qid = db.get_current_question_id(
                question['id'], question['exam_name'], question['question_number']
            )
            if current_qid is None:
                pbar.update(len(models_to_run))
                continue

            # Fire all pending models for this question in parallel
            futures = {}
            with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(models_to_run))) as pool:
                for model in models_to_run:
                    if not db.response_exists(current_qid, model['name']):
                        futures[pool.submit(test_model_on_question, model, question)] = model

            # Collect results and write to DB sequentially (SQLite is not thread-safe for writes)
            for future, model in futures.items():
                pbar.set_description(f"{model['name']} Q{question['question_number']}")
                try:
                    result = future.result()
                except Exception as e:
                    result = {'response': f'ERROR: {e}', 'latency_ms': 0, 'tokens_used': 0, 'cost_usd': 0}

                # Re-check qid in case of concurrent re-extraction
                current_qid = db.get_current_question_id(
                    question['id'], question['exam_name'], question['question_number']
                )
                if current_qid is None or db.response_exists(current_qid, model['name']):
                    pbar.update(1)
                    continue

                db.add_response(
                    question_id=current_qid,
                    model_name=model['name'],
                    response=result['response'],
                    latency_ms=result['latency_ms'],
                    tokens_used=result['tokens_used'],
                    cost_usd=result['cost_usd'],
                )
                with lock:
                    total_cost += result['cost_usd']
                pbar.update(1)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"TESTING COMPLETE")
    print(f"{'='*70}")
    print(f"Total tests run: {total_tests}")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"\nNext step: Run scripts/3_evaluate.py")
    
    db.close()


if __name__ == "__main__":
    main()
