
import sqlite3
import pandas as pd

conn = sqlite3.connect('data/questions.db')

# Summary stats
print("\n=== MODEL PERFORMANCE SUMMARY ===")
query = """
SELECT 
    model_name,
    COUNT(*) as total,
    SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) as correct,
    AVG(score) as avg_score,
    AVG(latency_ms) as avg_latency
FROM evaluations e
JOIN model_responses r ON e.response_id = r.id
GROUP BY model_name
"""
df = pd.read_sql(query, conn)
df['accuracy'] = (df['correct'] / df['total'] * 100).round(1)
df['avg_score'] = df['avg_score'].round(2)
df['avg_latency'] = df['avg_latency'].round(0)
print(df)

# Show a few examples
print("\n=== SAMPLE COMPARISON ===")
query_examples = """
SELECT 
    q.question_number,
    r.model_name,
    r.response,
    e.score,
    e.evaluator_notes
FROM questions q
JOIN model_responses r ON q.id = r.question_id
JOIN evaluations e ON r.id = e.response_id
WHERE q.question_number = '1'
ORDER BY r.model_name
"""
examples = pd.read_sql(query_examples, conn)
for _, row in examples.iterrows():
    print(f"\n[Model: {row['model_name']}] Score: {row['score']}")
    print(f"Response: {row['response'][:100]}...")
    print(f"Notes: {row['evaluator_notes']}")
