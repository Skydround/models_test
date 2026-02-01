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
    
    prompt = f"""You are an expert at parsing Polish matura exam documents. 
    
OBJECTIVE: Extract questions, answers, AND their associated reading passages (context).

EXAM TEXT (first 10 pages):
{exam_text}

ANSWER KEY TEXT (first 10 pages):
{answer_text}

INSTRUCTIONS:
1. Extract ALL questions from the exam.
2. Match them with their correct answers.
3. **CRITICAL**: Extract the READING PASSAGE (context) for each question.

CONTEXT EXTRACTION RULES:
- **Explicit Context**: Look for texts labeled "Tekst 1", "Tekst 2", etc.
- **Implicit Context**: If a text is not labeled but clearly precedes a group of questions (e.g. "Przeczytaj poniższy tekst..."), that is the context.
- **Stateful Context**: Questions often don't repeat the text. If Question 2 follows Question 1, and Question 1 had "Tekst 1" as context, Question 2 likely has the SAME context unless a NEW text is introduced.
- **Reset**: Only reset the context when a NEW text, distinct section header, or "Tekst X" appears.
- **Output**: The `context_text` field should contain the FULL TEXT of the passage, or a description if it's an image/chart.

For each question, provide:
1. question_number: The task number (e.g., "1", "2.1", "3")
2. question_text: The full question text
3. question_type: "multiple_choice", "short_answer", "extended"
4. max_points: Maximum points
5. correct_answer: From answer key
6. answer_explanation: From answer key (if any)
7. page_number: Page number
8. context_text: The full reading passage/text associated with this question.

Return JSON array:
[
  {{
    "question_number": "1",
    "question_text": "...",
    "question_type": "multiple_choice",
    "max_points": 1,
    "correct_answer": "B",
    "answer_explanation": "...",
    "page_number": 4,
    "context_text": "FULL TEXT OF THE READING PASSAGE HERE..."
  }}
]

Important:
- Focus on questions from pages 4 onwards.
- Return ONLY the JSON array."""

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
        
        # --- HUMAN REVIEW STEP ---
        print(f"\n👀 HUMAN REVIEW: {exam['name']}")
        print("="*40)
        
        # Group by context to show summary
        contexts = {}
        no_context_count = 0
        for q in questions:
            ctx = q.get('context_text')
            q_num = q.get('question_number')
            if ctx:
                ctx_hash = hash(ctx) # simple grouping
                if ctx_hash not in contexts:
                    contexts[ctx_hash] = {'text': ctx, 'questions': []}
                contexts[ctx_hash]['questions'].append(q_num)
            else:
                no_context_count += 1
        
        print(f"Found {len(contexts)} unique contexts.")
        print(f"Questions without context: {no_context_count}")
        
        for i, (h, data) in enumerate(contexts.items(), 1):
            text_preview = data['text'][:100].replace('\n', ' ') + "..."
            q_list = ", ".join(data['questions'])
            print(f"\nContext {i} (Questions: {q_list}):")
            print(f"  \"{text_preview}\"")
        
        print("="*40)
        user_input = input("\n💾 Proceed to save to database? [y/N]: ").lower().strip()
        
        if user_input != 'y':
            print("❌ Skipping database insertion (JSON saved).")
            continue
            
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
                page_number=q.get('page_number', 0),
                context_text=q.get('context_text')
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
