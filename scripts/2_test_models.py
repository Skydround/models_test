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
from anthropic import Anthropic
from openai import OpenAI
import google.generativeai as genai
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))
from utils import QuestionDatabase

load_dotenv()

# Initialize clients
# Initialize clients
anthropic_client = None
if os.getenv('ANTHROPIC_API_KEY'):
    anthropic_client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

openai_client = None
if os.getenv('OPENAI_API_KEY'):
    openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

if os.getenv('GOOGLE_API_KEY'):
    genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))


# Model configurations
MODELS = [
    {
        'name': 'claude-3.5-haiku',
        'provider': 'anthropic',
        'model_id': 'claude-3-5-haiku-20241022',
        'input_cost_per_1m': 0.25,
        'output_cost_per_1m': 1.25
    },
    {
        'name': 'claude-3.5-sonnet',
        'provider': 'anthropic',
        'model_id': 'claude-3-5-sonnet-20241022',
        'input_cost_per_1m': 3.0,
        'output_cost_per_1m': 15.0
    },
    {
        'name': 'gpt-4o-mini',
        'provider': 'openai',
        'model_id': 'gpt-4o-mini',
        'input_cost_per_1m': 0.15,
        'output_cost_per_1m': 0.60
    },
    {
        'name': 'gpt-4o',
        'provider': 'openai',
        'model_id': 'gpt-4o',
        'input_cost_per_1m': 2.50,
        'output_cost_per_1m': 10.0
    },
    {
        'name': 'gemini-1.5-flash',
        'provider': 'google',
        'model_id': 'gemini-1.5-flash',
        'input_cost_per_1m': 0.075,
        'output_cost_per_1m': 0.30
    }
]


def ask_anthropic(model_config: dict, question_text: str) -> dict:
    """Ask Claude a question"""
    start_time = time.time()
    
    response = anthropic_client.messages.create(
        model=model_config['model_id'],
        max_tokens=2000,
        temperature=0,
        messages=[{
            "role": "user",
            "content": f"Odpowiedz na następujące pytanie z egzaminu maturalnego. Odpowiedz zwięźle i precyzyjnie.\n\nPytanie:\n{question_text}"
        }]
    )
    
    latency_ms = int((time.time() - start_time) * 1000)
    
    answer = response.content[0].text
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    total_tokens = input_tokens + output_tokens
    
    cost = (input_tokens * model_config['input_cost_per_1m'] + 
            output_tokens * model_config['output_cost_per_1m']) / 1_000_000
    
    return {
        'response': answer,
        'latency_ms': latency_ms,
        'tokens_used': total_tokens,
        'cost_usd': cost
    }


def ask_openai(model_config: dict, question_text: str) -> dict:
    """Ask GPT a question"""
    start_time = time.time()
    
    response = openai_client.chat.completions.create(
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
    total_tokens = response.usage.total_tokens
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    
    cost = (input_tokens * model_config['input_cost_per_1m'] + 
            output_tokens * model_config['output_cost_per_1m']) / 1_000_000
    
    return {
        'response': answer,
        'latency_ms': latency_ms,
        'tokens_used': total_tokens,
        'cost_usd': cost
    }


def ask_google(model_config: dict, question_text: str) -> dict:
    """Ask Gemini a question"""
    start_time = time.time()
    
    model = genai.GenerativeModel(model_config['model_id'])
    response = model.generate_content(
        f"Odpowiedz na następujące pytanie z egzaminu maturalnego. Odpowiedz zwięźle i precyzyjnie.\n\nPytanie:\n{question_text}",
        generation_config=genai.types.GenerationConfig(
            temperature=0,
            max_output_tokens=2000
        )
    )
    
    latency_ms = int((time.time() - start_time) * 1000)
    
    answer = response.text
    
    # Gemini token counting is approximate
    input_tokens = response.usage_metadata.prompt_token_count
    output_tokens = response.usage_metadata.candidates_token_count
    total_tokens = input_tokens + output_tokens
    
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
        if model_config['provider'] == 'anthropic':
            return ask_anthropic(model_config, question['question_text'])
        elif model_config['provider'] == 'openai':
            return ask_openai(model_config, question['question_text'])
        elif model_config['provider'] == 'google':
            return ask_google(model_config, question['question_text'])
        else:
            raise ValueError(f"Unknown provider: {model_config['provider']}")
    
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
    
    # Check API keys
    missing_keys = []
    if not os.getenv('ANTHROPIC_API_KEY'):
        missing_keys.append('ANTHROPIC_API_KEY')
    if not os.getenv('OPENAI_API_KEY'):
        missing_keys.append('OPENAI_API_KEY')
    if not os.getenv('GOOGLE_API_KEY'):
        missing_keys.append('GOOGLE_API_KEY')
    
    if missing_keys:
        print(f"\n⚠️  Warning: Missing API keys: {', '.join(missing_keys)}")
        print("Some models will be skipped. Add keys to .env file to test all models.")
    
    # Load questions from database
    db = QuestionDatabase()
    questions = db.get_all_questions()
    
    if not questions:
        print("\n❌ No questions found in database!")
        print("Run scripts/1_extract_questions.py first")
        return
    
    print(f"\n✅ Loaded {len(questions)} questions from database")
    
    # Filter models based on available API keys
    available_models = []
    for model in MODELS:
        if model['provider'] == 'anthropic' and os.getenv('ANTHROPIC_API_KEY'):
            available_models.append(model)
        elif model['provider'] == 'openai' and os.getenv('OPENAI_API_KEY'):
            available_models.append(model)
        elif model['provider'] == 'google' and os.getenv('GOOGLE_API_KEY'):
            available_models.append(model)
    
    print(f"\n🤖 Testing {len(available_models)} models:")
    for model in available_models:
        print(f"   - {model['name']}")
    
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
