
import json
import sys
from pathlib import Path
from utils import QuestionDatabase, Question

def import_questions():
    print("Importing questions from data/polski_2025_questions.json...")
    
    try:
        with open('data/polski_2025_questions.json', 'r') as f:
            questions = json.load(f)
    except FileNotFoundError:
        print("Error: JSON file not found.")
        return

    db = QuestionDatabase()
    # Clear existing questions for this exam to avoid duplicates if needed, or just append
    # But usually we want clean state for testing. 
    # For now, let's just add them. The ID is auto-increment.
    
    count = 0
    for q in questions:
        question = Question(
            id=0,
            exam_name='polski_2025',
            question_number=q.get('question_number', ''),
            question_text=q.get('question_text', ''),
            question_type=q.get('question_type', 'unknown'),
            max_points=q.get('max_points', 0),
            correct_answer=q.get('correct_answer'),
            answer_explanation=q.get('answer_explanation'),
            page_number=q.get('page_number', 0),
            context_text=q.get('context_text')
        )
        db.add_question(question)
        count += 1
        
    print(f"Successfully imported {count} questions into the database.")
    db.close()

if __name__ == "__main__":
    import_questions()
