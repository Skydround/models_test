#!/usr/bin/env python3
"""
Step 3: Evaluate model responses and generate comparison report

This script:
1. Loads questions and model responses from database
2. Uses LLM-as-judge to evaluate responses
3. Generates Excel comparison spreadsheet
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import sqlite3
import json

sys.path.append(str(Path(__file__).parent))
from utils import QuestionDatabase

load_dotenv()

openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def evaluate_response_with_llm(question: dict, response: str, correct_answer: str) -> dict:
    """
    Use GPT-4o-mini as judge to evaluate a response
    """
    
    prompt = f"""You are an expert evaluator for Polish matura exam responses.

QUESTION:
{question['question_text']}

CORRECT ANSWER:
{correct_answer if correct_answer else 'Not provided'}

STUDENT RESPONSE:
{response}

TASK:
Evaluate if the student's response is correct. Consider:
1. For multiple choice: exact match required
2. For short answers: semantic equivalence is acceptable
3. For extended answers: key points must be present

Respond in JSON format:
{{
  "score": 0.0 to 1.0,
  "is_correct": true/false,
  "notes": "Brief explanation of your evaluation"
}}

Return ONLY the JSON, no other text."""

    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0
        )
        
        result_text = completion.choices[0].message.content
        result = json.loads(result_text)
        return result
        
    except Exception as e:
        print(f"\n⚠️  Evaluation error: {e}")
        return {
            'score': 0.0,
            'is_correct': False,
            'notes': f'Evaluation failed: {str(e)}'
        }


def generate_excel_report(db: QuestionDatabase, output_path: str):
    """Generate Excel comparison spreadsheet"""
    
    print("\n📊 Generating Excel report...")
    
    # Get all data from database
    conn = db.conn
    
    # Query to get all questions with responses
    query = """
    SELECT 
        q.id as question_id,
        q.exam_name,
        q.question_number,
        q.question_text,
        q.question_type,
        q.max_points,
        q.correct_answer,
        mr.model_name,
        mr.response,
        mr.latency_ms,
        mr.tokens_used,
        mr.cost_usd,
        e.score,
        e.is_correct,
        e.evaluator_notes
    FROM questions q
    LEFT JOIN model_responses mr ON q.id = mr.question_id
    LEFT JOIN evaluations e ON mr.id = e.response_id
    ORDER BY q.exam_name, q.question_number, mr.model_name
    """
    
    df = pd.read_sql_query(query, conn)
    
    if df.empty:
        print("❌ No data to export")
        return
    
    # Create pivot table for easier comparison
    pivot_data = []
    
    for question_id in df['question_id'].unique():
        q_data = df[df['question_id'] == question_id]
        
        row = {
            'Exam': q_data.iloc[0]['exam_name'],
            'Question #': q_data.iloc[0]['question_number'],
            'Question': q_data.iloc[0]['question_text'][:100] + '...',  # Truncate
            'Type': q_data.iloc[0]['question_type'],
            'Max Points': q_data.iloc[0]['max_points'],
            'Correct Answer': q_data.iloc[0]['correct_answer']
        }
        
        # Add each model's response and score
        for _, resp in q_data.iterrows():
            if pd.notna(resp['model_name']):
                model = resp['model_name']
                row[f'{model}_Response'] = resp['response']
                row[f'{model}_Score'] = resp['score']
                row[f'{model}_Correct'] = '✓' if resp['is_correct'] else '✗'
                row[f'{model}_Latency_ms'] = resp['latency_ms']
                row[f'{model}_Cost_USD'] = resp['cost_usd']
        
        pivot_data.append(row)
    
    pivot_df = pd.DataFrame(pivot_data)
    
    # Create Excel file with multiple sheets
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: Full comparison
        pivot_df.to_excel(writer, sheet_name='Comparison', index=False)
        
        # Sheet 2: Summary statistics
        summary_data = []
        for model in df['model_name'].dropna().unique():
            model_data = df[df['model_name'] == model]
            summary_data.append({
                'Model': model,
                'Total Questions': len(model_data),
                'Correct': model_data['is_correct'].sum(),
                'Accuracy': f"{model_data['is_correct'].mean() * 100:.1f}%",
                'Avg Score': f"{model_data['score'].mean():.2f}",
                'Avg Latency (ms)': f"{model_data['latency_ms'].mean():.0f}",
                'Total Cost (USD)': f"${model_data['cost_usd'].sum():.4f}",
                'Cost per Question': f"${model_data['cost_usd'].mean():.4f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Sheet 3: Raw data
        df.to_excel(writer, sheet_name='Raw Data', index=False)
    
    print(f"✅ Excel report saved to: {output_path}")


def main():
    """Main evaluation workflow"""
    
    print("="*70)
    print("STEP 3: EVALUATE RESPONSES")
    print("="*70)
    
    if not os.getenv('OPENAI_API_KEY'):
        print("\n❌ OPENAI_API_KEY required for evaluation")
        return
    
    db = QuestionDatabase()
    
    # Get all responses that need evaluation
    conn = db.conn
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT mr.id, mr.question_id, mr.model_name, mr.response,
               q.question_text, q.correct_answer, q.question_type
        FROM model_responses mr
        JOIN questions q ON mr.question_id = q.id
        WHERE mr.id NOT IN (SELECT response_id FROM evaluations)
    """)
    
    responses_to_evaluate = cursor.fetchall()
    
    if not responses_to_evaluate:
        print("\n✅ All responses already evaluated")
    else:
        print(f"\n📝 Evaluating {len(responses_to_evaluate)} responses...")
        
        total_cost = 0
        
        for resp in tqdm(responses_to_evaluate, desc="Evaluating"):
            resp_id, q_id, model_name, response, q_text, correct_ans, q_type = resp
            
            question = {
                'question_text': q_text,
                'question_type': q_type
            }
            
            # Evaluate
            eval_result = evaluate_response_with_llm(question, response, correct_ans)
            
            # Save evaluation
            cursor.execute("""
                INSERT INTO evaluations (response_id, score, is_correct, evaluator_notes)
                VALUES (?, ?, ?, ?)
            """, (
                resp_id,
                eval_result['score'],
                eval_result['is_correct'],
                eval_result['notes']
            ))
            conn.commit()
            
            # Rough cost estimate (very cheap with Haiku)
            total_cost += 0.0001
        
        print(f"\n✅ Evaluation complete. Cost: ~${total_cost:.4f}")
    
    # Generate Excel report
    output_path = "results/comparison.xlsx"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    generate_excel_report(db, output_path)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"\n📊 Results saved to: {output_path}")
    print(f"\nOpen the Excel file to see:")
    print(f"  - Comparison sheet: Side-by-side model responses")
    print(f"  - Summary sheet: Model performance statistics")
    print(f"  - Raw Data sheet: All data for further analysis")
    
    db.close()


if __name__ == "__main__":
    main()
