#!/usr/bin/env python3
"""
Step 3: Evaluate model responses and generate comparison report.

This script:
1. Loads questions and model responses from the database
2. Uses an OpenRouter model as an LLM judge
3. Generates an Excel comparison spreadsheet
"""

import os
import sys
import html
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import json

sys.path.append(str(Path(__file__).parent))
from utils import QuestionDatabase

load_dotenv()

EVALUATOR_MODEL = os.getenv('OPENROUTER_EVALUATOR_MODEL', 'openai/gpt-4o-mini')

openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv('OPENROUTER_API_KEY'),
    default_headers={
        "HTTP-Referer": "https://github.com/models_test",
        "X-Title": "Models Test - Response Evaluation"
    }
)


def normalize_optional_text(value: str | None) -> str | None:
    """Convert placeholder strings to missing values."""
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized or normalized.lower() in {'null', 'none', 'brak'}:
        return None
    return normalized


def normalize_score(raw_score: float, max_points: int | None) -> float:
    """Map model scores to the scoring granularity implied by max_points."""
    score = max(0.0, min(1.0, float(raw_score)))
    if not max_points or max_points <= 1:
        return 1.0 if score >= 0.999 else 0.0

    step = 1 / max_points
    normalized = round(score / step) * step
    return round(max(0.0, min(1.0, normalized)), 2)


def normalize_evaluation_result(result: dict, max_points: int | None) -> dict:
    """Post-process evaluator output to match the exam scoring scheme."""
    normalized = dict(result)
    score = normalize_score(normalized.get('score', 0.0), max_points)
    normalized['score'] = score
    normalized['is_correct'] = score >= 0.999
    return normalized


def evaluate_response_with_llm(question: dict, response: str, correct_answer: str) -> dict:
    """
    Use an OpenRouter model as judge to evaluate a matura response.
    """

    answer_explanation = normalize_optional_text(question.get('answer_explanation')) or 'Not provided'
    context_text = normalize_optional_text(question.get('context_text')) or 'Not provided'
    exam_name = question.get('exam_name') or 'Unknown exam'
    question_type = question.get('question_type') or 'unknown'
    max_points = question.get('max_points')
    max_points_text = max_points if max_points is not None else 'Not provided'

    prompt = f"""You are an expert evaluator for Polish matura exam responses across all subjects.

Evaluate the answer in the context of the full exam task, not only by string matching.

EXAM:
{exam_name}

QUESTION TYPE:
{question_type}

MAX POINTS:
{max_points_text}

QUESTION:
{question['question_text']}

QUESTION CONTEXT:
{context_text}

CORRECT ANSWER:
{correct_answer if correct_answer else 'Not provided'}

ANSWER EXPLANATION / MARK SCHEME:
{answer_explanation}

STUDENT RESPONSE:
{response}

TASK:
Evaluate whether the student's response deserves full credit.

SCORING RULES:
1. For multiple choice or true/false tasks: exact correctness is required.
2. For short-answer tasks: accept semantic equivalence, equivalent notation, and minor wording differences.
3. For extended or open-ended tasks: require the essential points, claims, calculations, interpretations, or conclusions expected by the mark scheme.
4. For subjects other than Polish, prioritize subject accuracy over style.
5. Ignore minor grammar, spelling, or formatting issues unless they change the meaning.
6. If max_points = 1, score must be binary: only 0.0 or 1.0.
7. If max_points > 1, score must reflect point fractions achievable in the task, i.e. multiples of 1/max_points.
8. Examples: for 2 points use 0.0, 0.5, 1.0; for 3 points use 0.0, 0.33, 0.67, 1.0; for 4 points use 0.0, 0.25, 0.5, 0.75, 1.0.

Respond in JSON format:
{{
  "score": 0.0 to 1.0,
  "is_correct": true/false,
  "notes": "Brief explanation of your evaluation"
}}

Return ONLY the JSON, no other text."""

    try:
        completion = openrouter_client.chat.completions.create(
            model=EVALUATOR_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0
        )
        
        result_text = completion.choices[0].message.content
        result = json.loads(result_text)
        usage = getattr(completion, 'usage', None)
        if usage is not None:
            result['prompt_tokens'] = getattr(usage, 'prompt_tokens', 0) or 0
            result['completion_tokens'] = getattr(usage, 'completion_tokens', 0) or 0
        return normalize_evaluation_result(result, max_points)
        
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


def generate_html_report(db: QuestionDatabase, output_path: str):
        """Generate a readable HTML comparison report."""

        print("\n🌐 Generating HTML report...")

        query = """
        SELECT
                q.id as question_id,
                q.exam_name,
                q.question_number,
                q.question_text,
                q.question_type,
                q.max_points,
                q.correct_answer,
                q.answer_explanation,
                q.context_text,
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
        ORDER BY q.exam_name, q.page_number, q.question_number, mr.model_name
        """

        df = pd.read_sql_query(query, db.conn)

        if df.empty:
                print("❌ No data to export to HTML")
                return

        def as_text(value, fallback='Brak'):
            if pd.isna(value) or value is None:
                return fallback
            text = str(value).strip()
            if text == '' or text.lower() in {'null', 'none', 'brak'}:
                return fallback
            return text

        def as_html_block(value, fallback='Brak'):
                return html.escape(as_text(value, fallback)).replace('\n', '<br>')

        summary_rows = []
        for model in df['model_name'].dropna().unique():
                model_data = df[df['model_name'] == model]
                accuracy = model_data['is_correct'].fillna(0).astype(float).mean() * 100
                avg_score = model_data['score'].fillna(0).mean()
                summary_rows.append(
                        f"""
                        <tr>
                            <td>{html.escape(model)}</td>
                            <td>{len(model_data)}</td>
                            <td>{accuracy:.1f}%</td>
                            <td>{avg_score:.2f}</td>
                            <td>${model_data['cost_usd'].fillna(0).sum():.4f}</td>
                        </tr>
                        """
                )

        question_sections = []
        for question_id in df['question_id'].unique():
                q_data = df[df['question_id'] == question_id]
                q_row = q_data.iloc[0]

                response_cards = []
                for _, resp in q_data.iterrows():
                        if pd.isna(resp['model_name']):
                                continue

                        score = 0.0 if pd.isna(resp['score']) else float(resp['score'])
                        if score >= 0.99:
                                score_class = 'score-high'
                        elif score >= 0.5:
                                score_class = 'score-mid'
                        else:
                                score_class = 'score-low'

                        is_correct = bool(resp['is_correct']) if not pd.isna(resp['is_correct']) else False
                        verdict = 'Correct' if is_correct else 'Needs review'

                        response_cards.append(
                                f"""
                                <article class="response-card">
                                    <div class="response-header">
                                        <h3>{html.escape(str(resp['model_name']))}</h3>
                                        <div class="badges">
                                            <span class="badge {score_class}">Score: {score:.2f}</span>
                                            <span class="badge {'badge-ok' if is_correct else 'badge-warn'}">{verdict}</span>
                                        </div>
                                    </div>
                                    <div class="meta">Latency: {0 if pd.isna(resp['latency_ms']) else int(resp['latency_ms'])} ms | Tokens: {0 if pd.isna(resp['tokens_used']) else int(resp['tokens_used'])} | Cost: ${0 if pd.isna(resp['cost_usd']) else float(resp['cost_usd']):.4f}</div>
                                    <div class="panel">
                                        <div class="panel-title">Model response</div>
                                        <div class="panel-body">{as_html_block(resp['response'])}</div>
                                    </div>
                                    <div class="panel">
                                        <div class="panel-title">Evaluator notes</div>
                                        <div class="panel-body">{as_html_block(resp['evaluator_notes'])}</div>
                                    </div>
                                </article>
                                """
                        )

                context_value = as_text(q_row['context_text'])
                context_block = ''
                if context_value != 'Brak':
                        context_block = f"""
                            <details class="context-block">
                                <summary>Question context</summary>
                                <div class="panel-body">{as_html_block(q_row['context_text'])}</div>
                            </details>
                        """

                question_sections.append(
                        f"""
                        <section class="question-card">
                            <div class="question-topline">
                                <span class="exam-pill">{html.escape(as_text(q_row['exam_name']))}</span>
                                <span class="question-pill">Question {html.escape(as_text(q_row['question_number']))}</span>
                                <span class="type-pill">{html.escape(as_text(q_row['question_type']))}</span>
                                <span class="points-pill">{html.escape(as_text(q_row['max_points']))} pts</span>
                            </div>
                            <h2>{html.escape(as_text(q_row['question_text']))}</h2>
                            <div class="details-grid">
                                <div class="panel">
                                    <div class="panel-title">Correct answer</div>
                                    <div class="panel-body">{as_html_block(q_row['correct_answer'])}</div>
                                </div>
                                <div class="panel">
                                    <div class="panel-title">Mark scheme / explanation</div>
                                    <div class="panel-body">{as_html_block(q_row['answer_explanation'])}</div>
                                </div>
                            </div>
                            {context_block}
                            <div class="responses-grid">
                                {''.join(response_cards)}
                            </div>
                        </section>
                        """
                )

        html_output = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Models Test Comparison</title>
    <style>
        :root {{
            --bg: #f5efe4;
            --surface: #fffdf8;
            --surface-strong: #fff8ee;
            --text: #1f1a17;
            --muted: #6f6257;
            --border: #dfcfbb;
            --accent: #b85c38;
            --accent-soft: #f3d9bf;
            --ok: #2f7d4a;
            --warn: #9d5c0d;
            --bad: #a63f3f;
            --shadow: 0 14px 40px rgba(71, 48, 26, 0.08);
        }}
        * {{ box-sizing: border-box; }}
        body {{ margin: 0; font-family: Georgia, 'Times New Roman', serif; background: radial-gradient(circle at top, #fff7ea 0%, var(--bg) 45%, #efe4d3 100%); color: var(--text); }}
        .page {{ max-width: 1500px; margin: 0 auto; padding: 32px 20px 60px; }}
        .hero {{ background: linear-gradient(135deg, rgba(184, 92, 56, 0.12), rgba(243, 217, 191, 0.75)); border: 1px solid var(--border); border-radius: 28px; padding: 28px; box-shadow: var(--shadow); }}
        h1 {{ margin: 0 0 8px; font-size: clamp(2rem, 4vw, 3.5rem); }}
        .subtitle {{ margin: 0; color: var(--muted); font-size: 1.05rem; }}
        .summary-table {{ width: 100%; border-collapse: collapse; margin-top: 24px; background: var(--surface); border-radius: 18px; overflow: hidden; box-shadow: var(--shadow); }}
        .summary-table th, .summary-table td {{ padding: 14px 16px; border-bottom: 1px solid #efe2d3; text-align: left; }}
        .summary-table th {{ background: var(--surface-strong); }}
        .question-card {{ margin-top: 28px; background: rgba(255, 253, 248, 0.88); border: 1px solid var(--border); border-radius: 28px; padding: 24px; box-shadow: var(--shadow); backdrop-filter: blur(4px); }}
        .question-topline {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 12px; }}
        .exam-pill, .question-pill, .type-pill, .points-pill, .badge {{ display: inline-flex; align-items: center; border-radius: 999px; padding: 6px 12px; font-size: 0.9rem; font-weight: 600; }}
        .exam-pill {{ background: var(--accent); color: white; }}
        .question-pill {{ background: #f0e3d1; }}
        .type-pill {{ background: #ede7dc; color: #53473f; }}
        .points-pill {{ background: #f7ead8; color: #6d4c37; }}
        .details-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin: 18px 0; }}
        .responses-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; margin-top: 18px; }}
        .response-card {{ background: var(--surface); border: 1px solid #eadbc9; border-radius: 22px; padding: 18px; box-shadow: 0 10px 30px rgba(71, 48, 26, 0.05); }}
        .response-header {{ display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; margin-bottom: 10px; }}
        .response-header h3 {{ margin: 0; font-size: 1.1rem; }}
        .badges {{ display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }}
        .badge {{ background: #ece2d3; color: #3e342e; }}
        .badge-ok {{ background: rgba(47, 125, 74, 0.14); color: var(--ok); }}
        .badge-warn {{ background: rgba(157, 92, 13, 0.14); color: var(--warn); }}
        .score-high {{ background: rgba(47, 125, 74, 0.14); color: var(--ok); }}
        .score-mid {{ background: rgba(157, 92, 13, 0.14); color: var(--warn); }}
        .score-low {{ background: rgba(166, 63, 63, 0.14); color: var(--bad); }}
        .meta {{ color: var(--muted); font-size: 0.92rem; margin-bottom: 12px; }}
        .panel {{ background: #fffaf3; border: 1px solid #efe1cf; border-radius: 16px; padding: 14px; margin-top: 12px; }}
        .panel-title {{ font-size: 0.82rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); margin-bottom: 8px; }}
        .panel-body {{ line-height: 1.6; white-space: normal; word-break: break-word; }}
        .context-block {{ margin-top: 10px; border-top: 1px solid #eadbc9; padding-top: 14px; }}
        .context-block summary {{ cursor: pointer; font-weight: 600; color: #5b4638; }}
        @media (max-width: 720px) {{
            .page {{ padding: 20px 14px 40px; }}
            .question-card, .hero {{ padding: 18px; border-radius: 22px; }}
            .response-header {{ flex-direction: column; }}
            .badges {{ justify-content: flex-start; }}
        }}
    </style>
</head>
<body>
    <main class="page">
        <section class="hero">
            <h1>Matura Models Comparison</h1>
            <p class="subtitle">Question-first report with the official answer, model responses, and evaluator verdicts in one place.</p>
        </section>

        <table class="summary-table">
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Responses</th>
                    <th>Accuracy</th>
                    <th>Avg score</th>
                    <th>Total cost</th>
                </tr>
            </thead>
            <tbody>
                {''.join(summary_rows)}
            </tbody>
        </table>

        {''.join(question_sections)}
    </main>
</body>
</html>
"""

        Path(output_path).write_text(html_output, encoding='utf-8')
        print(f"✅ HTML report saved to: {output_path}")


def main():
    """Main evaluation workflow"""
    
    print("="*70)
    print("STEP 3: EVALUATE RESPONSES")
    print("="*70)
    
    if not os.getenv('OPENROUTER_API_KEY'):
        print("\n❌ OPENROUTER_API_KEY required for evaluation")
        return

    print(f"\n🤖 Evaluator model: {EVALUATOR_MODEL}")
    
    db = QuestionDatabase()
    
    # Get all responses that need evaluation
    conn = db.conn
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT mr.id, mr.question_id, mr.model_name, mr.response,
               q.exam_name, q.question_text, q.correct_answer, q.answer_explanation,
               q.question_type, q.max_points, q.context_text
        FROM model_responses mr
        JOIN questions q ON mr.question_id = q.id
        WHERE mr.id NOT IN (SELECT response_id FROM evaluations)
    """)
    
    responses_to_evaluate = cursor.fetchall()
    
    if not responses_to_evaluate:
        print("\n✅ All responses already evaluated")
    else:
        print(f"\n📝 Evaluating {len(responses_to_evaluate)} responses...")
        total_prompt_tokens = 0
        total_completion_tokens = 0
        
        for resp in tqdm(responses_to_evaluate, desc="Evaluating"):
            (
                resp_id,
                q_id,
                model_name,
                response,
                exam_name,
                q_text,
                correct_ans,
                answer_explanation,
                q_type,
                max_points,
                context_text,
            ) = resp
            
            question = {
                'exam_name': exam_name,
                'question_text': q_text,
                'question_type': q_type,
                'answer_explanation': answer_explanation,
                'max_points': max_points,
                'context_text': context_text,
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

            total_prompt_tokens += eval_result.get('prompt_tokens', 0)
            total_completion_tokens += eval_result.get('completion_tokens', 0)

        print("\n✅ Evaluation complete")
        if total_prompt_tokens or total_completion_tokens:
            print(
                f"   Tokens used: {total_prompt_tokens} prompt + "
                f"{total_completion_tokens} completion"
            )
    
    # Generate Excel report
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    excel_output_path = output_dir / "comparison.xlsx"
    html_output_path = output_dir / "comparison.html"

    generate_excel_report(db, str(excel_output_path))
    generate_html_report(db, str(html_output_path))
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"\n📊 Results saved to: {excel_output_path}")
    print(f"🌐 HTML report saved to: {html_output_path}")
    print(f"\nOpen the generated files to see:")
    print(f"  - HTML report: Question-by-question review with model answers and evaluation")
    print(f"  - Excel comparison: Side-by-side table and summary statistics")
    
    db.close()


if __name__ == "__main__":
    main()
