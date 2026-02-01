#!/usr/bin/env python3
"""
Step 1: Extract questions and answers from exam PDFs using AI

This script:
1. Extracts raw text from exam and answer PDFs
2. Uses Claude to parse and structure the Q&A pairs
3. Stores everything in SQLite database
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import json
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
from utils import QuestionDatabase, extract_text_from_pdf, save_json

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def parse_exam_with_ai(exam_pages: list, answer_pages: list, exam_name: str) -> list:
    """
    Use OpenAI (GPT-4o-mini) to parse exam and answer PDFs into structured Q&A pairs
    """
    
    # Combine first few pages of exam for context
    exam_text = "\n\n=== PAGE {} ===\n\n".join([
        f"{p['page_number']}\n{p['text']}" for p in exam_pages[:10]
    ])
    
    # Combine answer key pages
    answer_text = "\n\n=== PAGE {} ===\n\n".join([
        f"{p['page_number']}\n{p['text']}" for p in answer_pages[:10]
    ])
    
    prompt = f"""You are an expert at parsing Polish exam documents. I need you to extract questions and answers from a Polish matura exam.

EXAM TEXT (first 10 pages):
{exam_text}

ANSWER KEY TEXT (first 10 pages):
{answer_text}

Please extract ALL questions from the exam and match them with their correct answers. For each question, provide:

1. question_number: The task number (e.g., "1", "2.1", "3")
2. question_text: The full question text in Polish
3. question_type: One of: "multiple_choice", "short_answer", "extended"
4. max_points: Maximum points for this question (look for patterns like "0-1", "0-2", "0-4")
5. correct_answer: The correct answer from the answer key
6. answer_explanation: Any explanation provided in the answer key (if available)
7. page_number: The page number where the question appears

Return your response as a JSON array of question objects. Example format:

[
  {{
    "question_number": "1",
    "question_text": "Które z podanych zdań...",
    "question_type": "multiple_choice",
    "max_points": 1,
    "correct_answer": "B",
    "answer_explanation": "Odpowiedź B jest poprawna ponieważ...",
    "page_number": 4
  }}
]

Important:
- Extract ALL questions you can find
- Focus on questions from pages 4 onwards (skip instructions)
- Return ONLY the JSON array"""

    print(f"\n🤖 Sending to OpenAI (GPT-4o-mini) for parsing: {exam_name}")
    print(f"   Exam pages: {len(exam_pages)}, Answer pages: {len(answer_pages)}")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user", 
                "content": prompt
            }],
            response_format={"type": "json_object"},
            temperature=0
        )
        
        # Extract JSON from response
        response_text = response.choices[0].message.content
        
        # Parse JSON
        data = json.loads(response_text)
        # Handle if wrapped in a key like "questions" or just a list
        if isinstance(data, dict):
            questions = data.get('questions', list(data.values())[0])
        else:
            questions = data
            
        print(f"✅ Extracted {len(questions)} questions")
        
        # Calculate cost (approximate)
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cost = (input_tokens * 0.15 + output_tokens * 0.60) / 1_000_000
        print(f"   Cost: ${cost:.4f} ({input_tokens} in + {output_tokens} out tokens)")
        
        return questions
        
    except Exception as e:
        print(f"❌ Error parsing with AI: {e}")
        import traceback
        traceback.print_exc()
        return []


def main():
    """Main extraction workflow"""
    
    print("="*70)
    print("STEP 1: EXTRACT QUESTIONS FROM PDFs (OpenAI Mode)")
    print("="*70)
    
    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        print("\n❌ ERROR: OPENAI_API_KEY not found in environment")
        return
    
    # Initialize database
    db = QuestionDatabase()
    print("\n✅ Database initialized")
    
    # Define exams to process - ONLY POLSKI
    exams = [
        {
            'name': 'polski_2025',
            'exam_pdf': 'pdfs/polski_2025.pdf',
            'answer_pdf': 'pdfs/polski_2025_odp.pdf'
        }
        # Skipped matematyka_2025 for now
    ]
    
    total_questions = 0
    
    for exam in exams:
        print(f"\n{'='*70}")
        print(f"Processing: {exam['name']}")
        print(f"{'='*70}")
        
        # Extract text from PDFs
        print(f"\n📄 Extracting text from {exam['exam_pdf']}...")
        exam_pages = extract_text_from_pdf(exam['exam_pdf'])
        print(f"   Extracted {len(exam_pages)} pages")
        
        print(f"\n📄 Extracting text from {exam['answer_pdf']}...")
        answer_pages = extract_text_from_pdf(exam['answer_pdf'])
        print(f"   Extracted {len(answer_pages)} pages")
        
        # Save raw extracted text for debugging
        save_json({
            'exam_pages': exam_pages,
            'answer_pages': answer_pages
        }, f"data/{exam['name']}_raw.json")
        print(f"   Saved raw text to data/{exam['name']}_raw.json")
        
        # Parse with AI
        questions = parse_exam_with_ai(exam_pages, answer_pages, exam['name'])
        
        if not questions:
            print(f"⚠️  No questions extracted for {exam['name']}")
            continue
        
        # Save structured questions
        save_json(questions, f"data/{exam['name']}_questions.json")
        print(f"   Saved structured questions to data/{exam['name']}_questions.json")
        
        # Add to database
        print(f"\n💾 Adding questions to database...")
        for q in tqdm(questions, desc="Saving"):
            from utils import Question
            question = Question(
                id=0,  # Will be auto-assigned
                exam_name=exam['name'],
                question_number=q.get('question_number', ''),
                question_text=q.get('question_text', ''),
                question_type=q.get('question_type', 'unknown'),
                max_points=q.get('max_points', 0),
                correct_answer=q.get('correct_answer'),
                answer_explanation=q.get('answer_explanation'),
                page_number=q.get('page_number', 0)
            )
            db.add_question(question)
        
        total_questions += len(questions)
        print(f"✅ Added {len(questions)} questions from {exam['name']}")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"Total questions extracted: {total_questions}")
    print(f"Database: data/questions.db")
    print(f"\nNext step: Run scripts/2_test_models.py")
    
    db.close()


if __name__ == "__main__":
    main()
