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
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))
from utils import QuestionDatabase

load_dotenv()

# Initialize OpenRouter client (using OpenAI-compatible interface)
client = OpenAI(
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
        'name': 'gpt-4o-mini',
        'model_id': 'openai/gpt-4o-mini',
        'input_cost_per_1m': 0.15,
        'output_cost_per_1m': 0.60
    },
    {
        'name': 'grok-4-fast',
        'model_id': 'x-ai/grok-4-fast',
        'input_cost_per_1m': 0.20,
        'output_cost_per_1m': 0.50
    },
    {
        'name': 'deepseek-v3',
        'model_id': 'deepseek/deepseek-chat',
        'input_cost_per_1m': 0.28,
        'output_cost_per_1m': 0.40
    },
    {
        'name': 'gemini-2.5-flash',
        'model_id': 'google/gemini-2.5-flash',
        'input_cost_per_1m': 0.30,
        'output_cost_per_1m': 2.50
    },
    {
        'name': 'gemini-3-flash',
        'model_id': 'google/gemini-3-flash-preview',
        'input_cost_per_1m': 0.50,
        'output_cost_per_1m': 3.00
    },
    {
        'name': 'claude-haiku-4.5',
        'model_id': 'anthropic/claude-haiku-4.5',
        'input_cost_per_1m': 1.00,
        'output_cost_per_1m': 5.00
    }
]


def ask_model(model_config: dict, question_text: str) -> dict:
    """Ask any model a question via OpenRouter"""
    start_time = time.time()
    
    response = client.chat.completions.create(
        model=model_config['model_id'],
        messages=[{
            "role": "user",
            "content": f"Odpowiedz na następujące pytanie z egzaminu maturalnego. Odpowiedz zwięźle i precyzyjnie.\n\nPytanie:\n{question_text}"
        }],
        temperature=0,
        max_tokens=2000
    )
    
    latency_ms = int((time.time() - start_time) * 1000)
    
    answer = response.choices[0].message.content
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
        return ask_model(model_config, question['question_text'])
    
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
    
    # Test each model on each question
    total_tests = len(questions) * len(available_models)
    total_cost = 0
    
    print(f"\n📊 Total tests to run: {total_tests}")
    print(f"   ({len(questions)} questions × {len(available_models)} models)")
    
    with tqdm(total=total_tests, desc="Testing") as pbar:
        for question in questions:
            for model in available_models:
                pbar.set_description(f"{model['name']} Q{question['question_number']}")
                
                result = test_model_on_question(model, question)
                
                # Save to database
                db.add_response(
                    question_id=question['id'],
                    model_name=model['name'],
                    response=result['response'],
                    latency_ms=result['latency_ms'],
                    tokens_used=result['tokens_used'],
                    cost_usd=result['cost_usd']
                )
                
                total_cost += result['cost_usd']
                pbar.update(1)
                
                # Small delay to avoid rate limits
                time.sleep(0.5)
    
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
