#!/usr/bin/env python3
"""
Step 1: Extract questions and answers from exam PDFs using AI.

This script:
1. Extracts raw text from exam and answer PDFs
2. Uses an OpenRouter model to parse and structure the Q&A pairs
3. Stores everything in SQLite database
"""

import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
import json
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
from utils import QuestionDatabase, discover_exam_pdf_sets, extract_text_from_pdf, parse_exam_name, save_json

# Load environment variables
load_dotenv()

# Initialize OpenRouter client (using OpenAI-compatible interface)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv('OPENROUTER_API_KEY'),
    default_headers={
        "HTTP-Referer": "https://github.com/models_test",
        "X-Title": "Models Test - Question Extraction"
    }
)

EXTRACTION_MODEL = os.getenv('OPENROUTER_EXTRACTION_MODEL', 'openai/gpt-4o-mini')
EXAM_START_PAGE = int(os.getenv('EXAM_START_PAGE', '4'))
EXAM_BATCH_SIZE = int(os.getenv('EXAM_BATCH_SIZE', '4'))
EXAM_BATCH_OVERLAP = min(int(os.getenv('EXAM_BATCH_OVERLAP', '1')), EXAM_BATCH_SIZE - 1)
MAX_CONTEXT_CHARS = int(os.getenv('MAX_CONTEXT_CHARS', '1200'))
EXAM_FILTER = [item.strip() for item in os.getenv('EXAM_FILTER', '').split(',') if item.strip()]


def format_table_block(table) -> str | None:
    """Render a pdfplumber table into compact plain text."""
    if not table:
        return None

    rendered_rows = []
    for row in table:
        if not row:
            continue
        cleaned_cells = []
        for cell in row:
            if cell is None:
                continue
            normalized = re.sub(r'\s+', ' ', str(cell)).strip()
            if normalized:
                cleaned_cells.append(normalized)
        if cleaned_cells:
            rendered_rows.append(' | '.join(cleaned_cells))

    if not rendered_rows:
        return None
    return '\n'.join(rendered_rows)


def page_content(page: dict) -> str:
    """Build one extraction-ready text block from page text and extracted tables."""
    parts = []
    text = str(page.get('text') or '').strip()
    if text:
        parts.append(text)

    table_blocks = []
    for table in page.get('tables') or []:
        rendered = format_table_block(table)
        if rendered:
            table_blocks.append(rendered)

    if table_blocks:
        parts.append('TABLES:\n' + '\n\n'.join(table_blocks))

    return '\n\n'.join(parts).strip()


def format_pages(pages: list) -> str:
    """Render pages into a prompt-friendly string."""
    return "\n\n".join(
        [f"=== PAGE {page['page_number']} ===\n{page_content(page)}" for page in pages]
    )


def build_page_batches(exam_pages: list) -> list:
    """Split exam pages into overlapping batches starting from the question pages."""
    relevant_pages = [page for page in exam_pages if page['page_number'] >= EXAM_START_PAGE]
    if not relevant_pages:
        return []

    batches = []
    start = 0
    while start < len(relevant_pages):
        end = min(start + EXAM_BATCH_SIZE, len(relevant_pages))
        batches.append(relevant_pages[start:end])
        if end >= len(relevant_pages):
            break
        start = max(end - EXAM_BATCH_OVERLAP, start + 1)
    return batches


def parse_int_field(value, default: int = 0) -> int:
    """Parse integer-like values returned by the model."""
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    digits = re.findall(r"\d+", str(value or ""))
    if not digits:
        return default
    return max(int(part) for part in digits)


def truncate_context(context_text: str | None) -> str | None:
    """Keep context compact enough for downstream prompts and reports."""
    if not context_text:
        return None
    normalized = context_text.strip()
    if normalized.lower() in {'null', 'none', 'brak'}:
        return None
    if len(normalized) <= MAX_CONTEXT_CHARS:
        return normalized
    return normalized[:MAX_CONTEXT_CHARS].rstrip() + "..."


def normalize_text_field(value) -> str | None:
    """Convert model output variants into plain text for storage."""
    if value is None:
        return None
    if isinstance(value, list):
        parts = [normalize_text_field(item) for item in value]
        filtered = [part for part in parts if part]
        return "\n".join(filtered) if filtered else None
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)

    normalized = str(value).strip()
    if normalized.lower() in {'', 'null', 'none', 'brak'}:
        return None
    return normalized


def strip_noise_lines(text: str | None) -> str | None:
    """Remove page chrome and cut text before obvious next-section spillover."""
    if not text:
        return None

    cleaned = text.replace('\r\n', '\n').strip()
    boundary_patterns = [
        r'\nPRZENIEŚ ROZWIĄZANIA.*$',
        r'\nZadanie\s+\d+[A-Za-z]?\b.*$',
        r'\nTekst\s+\d+\.?[^\n]*$',
        r'\nStrona\s+\d+\s+z\s+\d+.*$',
        r'\n[A-Z]{3,}(?:-[A-Z0-9]+)+.*$',
    ]
    for pattern in boundary_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.DOTALL)

    kept_lines = []
    for line in cleaned.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('PRZENIEŚ ROZWIĄZANIA'):
            break
        if stripped.startswith('Strona '):
            continue
        if re.match(r'^[A-Z]{3,}(?:-[A-Z0-9]+)+', stripped):
            continue
        kept_lines.append(stripped)

    normalized = '\n'.join(kept_lines).strip()
    return normalized or None


def normalize_max_points(question_number: str, value) -> int:
    """Normalize max_points to a safe integer for one extracted question."""
    max_points = parse_int_field(value, 0)
    if max_points <= 0:
        return 1 if '.' in question_number else 0
    return max_points


def normalize_question(raw_question: dict) -> dict | None:
    """Normalize model output into the expected question schema."""
    question_number = str(raw_question.get('question_number', '')).strip()
    question_text = normalize_text_field(raw_question.get('question_text')) or ''
    question_text = strip_noise_lines(question_text) or ''
    if not question_number or not question_text:
        return None

    return {
        'question_number': question_number,
        'question_text': question_text,
        'question_type': str(raw_question.get('question_type', 'unknown')).strip() or 'unknown',
        'max_points': normalize_max_points(question_number, raw_question.get('max_points')),
        'correct_answer': normalize_text_field(raw_question.get('correct_answer')),
        'answer_explanation': strip_noise_lines(normalize_text_field(raw_question.get('answer_explanation'))),
        'page_number': parse_int_field(raw_question.get('page_number'), 0),
        'context_text': truncate_context(strip_noise_lines(normalize_text_field(raw_question.get('context_text')))),
    }


def normalize_text_for_compare(value: str | None) -> str:
    """Normalize text for rough semantic equality checks."""
    if not value:
        return ''
    collapsed = re.sub(r'\s+', ' ', str(value)).strip().lower()
    return re.sub(r'[^\wąćęłńóśźż]+', '', collapsed)


def question_marker_patterns(question_number: str) -> list[str]:
    """Return likely textual markers for a question number inside raw page text."""
    escaped = re.escape(question_number)
    return [
        rf"{escaped}\.\s*Zadanie\s+{escaped}\.",
        rf"Zadanie\s+{escaped}\.",
        rf"(?:^|\n){escaped}\.\s",
        rf"(?:^|\n){escaped}\.\n",
    ]


def find_question_block(page_text: str, question_number: str, next_question_number: str | None) -> str | None:
    """Extract the raw text block for one question from a page."""
    start_match = None
    for pattern in question_marker_patterns(question_number):
        start_match = re.search(pattern, page_text, flags=re.IGNORECASE)
        if start_match:
            break

    if not start_match:
        return None

    start_index = start_match.start()
    end_index = len(page_text)

    if next_question_number:
        for pattern in question_marker_patterns(next_question_number):
            next_match = re.search(pattern, page_text[start_match.end():], flags=re.IGNORECASE)
            if next_match:
                end_index = start_match.end() + next_match.start()
                break

    block = page_text[start_index:end_index].strip()
    return block or None


def clean_question_block(block: str | None) -> str | None:
    """Remove obvious answer lines while keeping meaningful task details."""
    if not block:
        return None

    block = strip_noise_lines(block)
    if not block:
        return None

    lines = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('Strona '):
            continue
        if stripped.startswith('PRZENIEŚ ROZWIĄZANIA'):
            break
        if re.match(r'^[A-Z]{3,}(?:-[A-Z0-9]+)+', stripped):
            continue
        if set(stripped) <= {'.', '_'}:
            continue
        lines.append(stripped)

    cleaned = '\n'.join(lines).strip()
    return cleaned or None


def block_has_useful_details(block: str | None, question_text: str) -> bool:
    """Detect whether a page block contains extra task detail beyond the stem."""
    if not block:
        return False

    block_norm = normalize_text_for_compare(block)
    question_norm = normalize_text_for_compare(question_text)
    if not block_norm or block_norm == question_norm:
        return False

    detail_signals = [
        '\nA.',
        '\nB.',
        '\nC.',
        '\nD.',
        'P F',
        'Fragment',
        'Postać',
        'Tekst 1',
        'Tekst 2',
        'Nazwa środka retorycznego',
        'Rycerz',
    ]
    return any(signal in block for signal in detail_signals)


def enrich_questions_with_page_blocks(questions: list[dict], exam_pages: list) -> list[dict]:
    """Inject page-local task details that the model may have omitted."""
    page_lookup = {page['page_number']: page_content(page) for page in exam_pages}
    questions_by_page: dict[int, list[dict]] = {}
    for question in questions:
        questions_by_page.setdefault(question['page_number'], []).append(question)

    for page_number, page_questions in questions_by_page.items():
        page_questions.sort(key=question_sort_key)
        page_text = page_lookup.get(page_number, '')
        for index, question in enumerate(page_questions):
            next_question_number = None
            if index + 1 < len(page_questions):
                next_question_number = page_questions[index + 1]['question_number']

            block = clean_question_block(
                find_question_block(page_text, question['question_number'], next_question_number)
            )
            if not block:
                continue

            current_context = question.get('context_text')
            current_context_norm = normalize_text_for_compare(current_context)
            question_text_norm = normalize_text_for_compare(question['question_text'])

            if current_context_norm == question_text_norm:
                question['context_text'] = None

            if block_has_useful_details(block, question['question_text']):
                question['context_text'] = truncate_context(block)

                if question['question_type'] == 'multiple_choice' and block not in question['question_text']:
                    question['question_text'] = block

    return questions


def merge_question(existing: dict | None, candidate: dict) -> dict:
    """Merge duplicate question records gathered from overlapping batches."""
    if existing is None:
        return candidate

    merged = dict(existing)
    for field in ['question_text', 'correct_answer', 'answer_explanation', 'context_text']:
        current_value = merged.get(field) or ''
        candidate_value = candidate.get(field) or ''
        if len(candidate_value) > len(current_value):
            merged[field] = candidate.get(field)

    if merged.get('question_type') in ('', 'unknown') and candidate.get('question_type'):
        merged['question_type'] = candidate['question_type']

    if candidate.get('max_points', 0) > merged.get('max_points', 0):
        merged['max_points'] = candidate['max_points']

    if merged.get('page_number', 0) == 0 and candidate.get('page_number', 0):
        merged['page_number'] = candidate['page_number']

    return merged


def normalize_group_max_points(questions: list[dict]) -> list[dict]:
    """Convert task-level point totals into per-subquestion points when clearly safe."""
    grouped_questions: dict[str, list[dict]] = {}
    for question in questions:
        question_number = question.get('question_number', '')
        if '.' not in question_number:
            continue
        parent_number = question_number.split('.', 1)[0]
        grouped_questions.setdefault(parent_number, []).append(question)

    for group in grouped_questions.values():
        if len(group) < 2:
            continue
        point_values = {parse_int_field(question.get('max_points'), 0) for question in group}
        if len(point_values) != 1:
            continue

        total_points = next(iter(point_values))
        if total_points > 1 and total_points == len(group):
            for question in group:
                question['max_points'] = 1

    return questions


def question_sort_key(question: dict):
    """Sort question numbers naturally, e.g. 7.2 after 7.1."""
    parts = re.findall(r"\d+|\D+", question['question_number'])
    key = []
    for part in parts:
        key.append(int(part) if part.isdigit() else part.lower())
    return key


def parse_exam_batch_with_ai(
    exam_batch: list,
    answer_pages: list,
    exam_name: str,
    batch_index: int,
    batch_count: int,
    transcript_pages: list | None = None,
) -> tuple[list, float]:
    """
    Parse one batch of exam pages into structured Q&A pairs.
    """
    exam_text = format_pages(exam_batch)
    answer_text = format_pages(answer_pages)
    transcript_text = format_pages(transcript_pages or []) if transcript_pages else 'No transcript provided.'
    batch_pages = ", ".join(str(page['page_number']) for page in exam_batch)

    prompt = f"""You are an expert at parsing Polish matura exam documents across subjects.

OBJECTIVE: Extract only the questions that appear on the provided exam pages, then match them to the answer key.

EXAM NAME:
{exam_name}

CURRENT EXAM PAGES:
{batch_pages}

EXAM TEXT:
{exam_text}

FULL ANSWER KEY TEXT:
{answer_text}

OPTIONAL TRANSCRIPT TEXT:
{transcript_text}

INSTRUCTIONS:
1. Extract every question that appears on the CURRENT EXAM PAGES only.
2. Do not include questions from other pages.
3. Match each extracted question with its correct answer and explanation from the answer key.
4. Preserve the original task numbering, e.g. 6, 7.1, 10.2.
5. Infer question_type as one of: multiple_choice, short_answer, extended.
6. If the task depends on listening or a transcript, use OPTIONAL TRANSCRIPT TEXT as supporting context.

CONTEXT RULES:
- If a question refers to a reading passage, chart, table, diagram, or source text, include only the relevant passage or a concise excerpt.
- Keep context_text under {MAX_CONTEXT_CHARS} characters.
- If the wording says "na podstawie tekstu", "w przytoczonym fragmencie", "odwołaj się do obu tekstów", or otherwise depends on a source shown on the same or previous pages in the batch, context_text must contain that source excerpt.
- Propagate the same source context to later questions until a new source text or fragment appears.
- Use null only for truly standalone questions that do not depend on any external source.

OUTPUT FORMAT:
Return a JSON object with a single key named questions.

Example:
{{
  "questions": [
    {{
      "question_number": "1",
      "question_text": "...",
      "question_type": "short_answer",
      "max_points": 2,
      "correct_answer": "...",
      "answer_explanation": "...",
      "page_number": 6,
      "context_text": "..."
    }}
  ]
}}

Return only valid JSON."""

    print(f"\n🤖 Parsing batch {batch_index}/{batch_count} for {exam_name}")
    print(f"   Exam pages: {batch_pages}")
    
    try:
        response = client.chat.completions.create(
            model=EXTRACTION_MODEL,
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
        questions = data.get('questions', []) if isinstance(data, dict) else []
        normalized_questions = []
        for raw_question in questions:
            normalized = normalize_question(raw_question)
            if normalized is not None:
                normalized_questions.append(normalized)
            
        print(f"✅ Extracted {len(normalized_questions)} questions from batch")
        
        # Calculate cost (approximate)
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cost = (input_tokens * 0.15 + output_tokens * 0.60) / 1_000_000
        print(f"   Cost: ${cost:.4f} ({input_tokens} in + {output_tokens} out tokens)")
        
        return normalized_questions, cost
        
    except Exception as e:
        print(f"❌ Error parsing with AI: {e}")
        import traceback
        traceback.print_exc()
        return [], 0.0


def main():
    """Main extraction workflow"""
    
    print("="*70)
    print("STEP 1: EXTRACT QUESTIONS FROM PDFs")
    print("="*70)
    
    # Check for API key
    if not os.getenv('OPENROUTER_API_KEY'):
        print("\n❌ ERROR: OPENROUTER_API_KEY not found in environment")
        print("Get your API key from: https://openrouter.ai/keys")
        return
    
    # Initialize database
    db = QuestionDatabase()
    print("\n✅ Database initialized")
    
    auto_approve = os.getenv('AUTO_APPROVE_EXTRACT', '0') == '1' or not sys.stdin.isatty()

    exams = [exam.to_dict() for exam in discover_exam_pdf_sets('pdfs')]
    if EXAM_FILTER:
        exams = [
            exam for exam in exams
            if exam['name'] in EXAM_FILTER or any(exam['name'].startswith(prefix) for prefix in EXAM_FILTER)
        ]
    if not exams:
        print("\n❌ No complete exam sets found in pdfs/")
        print("Expected naming: subject_year.pdf + subject_year_odp.pdf")
        print("Optional files: subject_year_transkrypcja.pdf, subject_year_roz*.pdf")
        db.close()
        return

    print("\n📚 Discovered exam sets:")
    for exam in exams:
        metadata = parse_exam_name(exam['name'])
        transcript_marker = ' + transcript' if exam.get('transcript_pdf') else ''
        print(
            f"   - {metadata['session_label']} ({exam['name']}){transcript_marker}"
        )
    
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

        transcript_pages = []
        if exam.get('transcript_pdf'):
            print(f"\n📄 Extracting text from {exam['transcript_pdf']}...")
            transcript_pages = extract_text_from_pdf(exam['transcript_pdf'])
            print(f"   Extracted {len(transcript_pages)} transcript pages")
        
        # Save raw extracted text for debugging
        save_json({
            'exam_pages': exam_pages,
            'answer_pages': answer_pages,
            'transcript_pages': transcript_pages,
        }, f"data/{exam['name']}_raw.json")
        print(f"   Saved raw text to data/{exam['name']}_raw.json")
        
        # Parse with AI in batches to cover the full exam
        batches = build_page_batches(exam_pages)
        print(f"\n🧩 Parsing {len(batches)} batch(es) starting from page {EXAM_START_PAGE}")

        merged_questions = {}
        total_cost = 0.0
        for batch_index, batch in enumerate(batches, start=1):
            batch_questions, batch_cost = parse_exam_batch_with_ai(
                batch,
                answer_pages,
                exam['name'],
                batch_index,
                len(batches),
                transcript_pages=transcript_pages,
            )
            total_cost += batch_cost
            for question in batch_questions:
                number = question['question_number']
                merged_questions[number] = merge_question(merged_questions.get(number), question)

        questions = sorted(merged_questions.values(), key=question_sort_key)
        questions = enrich_questions_with_page_blocks(questions, exam_pages)
        questions = normalize_group_max_points(questions)
        print(f"\n✅ Total unique questions extracted: {len(questions)}")
        print(f"   Total extraction cost: ${total_cost:.4f}")
        
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
        if auto_approve:
            user_input = 'y'
            print("\n💾 Non-interactive mode detected, proceeding automatically.")
        else:
            user_input = input("\n💾 Proceed to save to database? [y/N]: ").lower().strip()
        
        if user_input != 'y':
            print("❌ Skipping database insertion (JSON saved).")
            continue
            
        db.clear_exam_data(exam['name'])
        print(f"🧹 Cleared existing database rows for {exam['name']}")

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
