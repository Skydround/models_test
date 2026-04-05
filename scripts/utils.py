"""
Shared utilities for the models testing project
"""

import sqlite3
import json
import re
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import pdfplumber

@dataclass
class Question:
    """Represents a single exam question"""
    id: int
    exam_name: str
    question_number: str
    question_text: str
    question_type: str  # 'multiple_choice', 'short_answer', 'extended'
    max_points: int
    correct_answer: Optional[str]
    answer_explanation: Optional[str]
    page_number: int
    context_text: Optional[str] = None  # Text passage the question refers to
    
    def to_dict(self):
        return asdict(self)


@dataclass
class ExamFileSet:
    """Represents one normalized exam session discovered from PDF filenames."""
    name: str
    subject: str
    year: int
    level: str
    exam_pdf: str
    answer_pdf: str
    transcript_pdf: Optional[str] = None

    def to_dict(self):
        return asdict(self)


def parse_exam_name(exam_name: str) -> Dict[str, str | int]:
    """Parse normalized exam names like subject_2025 or subject_2025_roz."""
    parts = str(exam_name).split('_')
    if len(parts) < 2:
        return {
            'name': exam_name,
            'subject': exam_name,
            'subject_label': exam_name.replace('_', ' ').title(),
            'year': 0,
            'level': 'unknown',
            'level_label': 'Poziom nieznany',
            'session_label': exam_name,
        }

    level = 'podstawowa'
    base_parts = parts
    if parts[-1] == 'roz':
        level = 'rozszerzona'
        base_parts = parts[:-1]

    if len(base_parts) < 2 or not base_parts[-1].isdigit():
        return {
            'name': exam_name,
            'subject': exam_name,
            'subject_label': exam_name.replace('_', ' ').title(),
            'year': 0,
            'level': level,
            'level_label': f'Matura {level}',
            'session_label': exam_name,
        }

    subject = '_'.join(base_parts[:-1])
    year = int(base_parts[-1])
    subject_label = subject.replace('_', ' ').title()
    return {
        'name': exam_name,
        'subject': subject,
        'subject_label': subject_label,
        'year': year,
        'level': level,
        'level_label': f'Matura {level}',
        'session_label': f'{subject_label} {year} - matura {level}',
    }


def exam_sort_key(exam_name: str):
    """Sort exams by subject, year, and level."""
    metadata = parse_exam_name(exam_name)
    level_order = {'podstawowa': 0, 'rozszerzona': 1}
    return (
        str(metadata['subject']),
        int(metadata['year']),
        level_order.get(str(metadata['level']), 99),
        exam_name,
    )


def excel_safe_sheet_name(name: str, fallback: str = 'Sheet') -> str:
    """Return a valid Excel sheet name capped at 31 characters."""
    cleaned = re.sub(r"[\\/*?:\[\]]", '_', str(name)).strip()
    cleaned = cleaned[:31].rstrip()
    return cleaned or fallback


def discover_exam_pdf_sets(pdf_dir: str = 'pdfs') -> List[ExamFileSet]:
    """Discover exam, answer key, and optional transcript PDFs from normalized filenames."""
    grouped: Dict[str, Dict[str, str]] = {}

    for pdf_path in sorted(Path(pdf_dir).glob('*.pdf')):
        stem = pdf_path.stem
        kind = 'exam'
        exam_name = stem

        if stem.endswith('_odpowiedzi'):
            kind = 'answer'
            exam_name = stem[:-11]
        elif stem.endswith('_odp'):
            kind = 'answer'
            exam_name = stem[:-4]
        elif stem.endswith('_transkrypcja'):
            kind = 'transcript'
            exam_name = stem[:-13]

        metadata = parse_exam_name(exam_name)
        if int(metadata['year']) == 0 or str(metadata['level']) == 'unknown':
            continue

        entry = grouped.setdefault(exam_name, {})
        entry[kind] = str(pdf_path)

    exams: List[ExamFileSet] = []
    for exam_name, files in grouped.items():
        exam_pdf = files.get('exam')
        answer_pdf = files.get('answer')
        if not exam_pdf or not answer_pdf:
            continue

        metadata = parse_exam_name(exam_name)
        exams.append(
            ExamFileSet(
                name=exam_name,
                subject=str(metadata['subject']),
                year=int(metadata['year']),
                level=str(metadata['level']),
                exam_pdf=exam_pdf,
                answer_pdf=answer_pdf,
                transcript_pdf=files.get('transcript'),
            )
        )

    exams.sort(key=lambda exam: exam_sort_key(exam.name))
    return exams


class QuestionDatabase:
    """Manages SQLite database of questions"""
    
    def __init__(self, db_path: str = "data/questions.db"):
        self.db_path = db_path
        self.conn = None
        self._init_db()
    
    def _init_db(self):
        """Initialize database with schema"""
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                exam_name TEXT NOT NULL,
                question_number TEXT NOT NULL,
                question_text TEXT NOT NULL,
                question_type TEXT NOT NULL,
                max_points INTEGER NOT NULL,
                correct_answer TEXT,
                answer_explanation TEXT,
                page_number INTEGER,
                context_text TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                response TEXT NOT NULL,
                latency_ms INTEGER,
                tokens_used INTEGER,
                cost_usd REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (question_id) REFERENCES questions(id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                response_id INTEGER NOT NULL,
                score REAL NOT NULL,
                is_correct BOOLEAN,
                evaluator_notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (response_id) REFERENCES model_responses(id)
            )
        """)
        
        self.conn.commit()

    def _coerce_text_value(self, value):
        """Convert structured values into strings before binding them to SQLite TEXT fields."""
        if value is None:
            return None
        if isinstance(value, list):
            parts = [self._coerce_text_value(item) for item in value]
            filtered = [part for part in parts if part]
            return "\n".join(filtered) if filtered else None
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
        return str(value)
    
    def add_question(self, question: Question) -> int:
        """Add a question to the database"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO questions 
            (exam_name, question_number, question_text, question_type, 
             max_points, correct_answer, answer_explanation, page_number, context_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            question.exam_name,
            question.question_number,
            question.question_text,
            question.question_type,
            question.max_points,
            self._coerce_text_value(question.correct_answer),
            self._coerce_text_value(question.answer_explanation),
            question.page_number,
            self._coerce_text_value(question.context_text)
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_all_questions(self) -> List[Dict]:
        """Get all questions from database"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM questions ORDER BY exam_name, page_number, question_number")
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def clear_exam_data(self, exam_name: str):
        """Remove questions, responses, and evaluations for a single exam."""
        cursor = self.conn.cursor()

        cursor.execute(
            """
            DELETE FROM evaluations
            WHERE response_id IN (
                SELECT mr.id
                FROM model_responses mr
                JOIN questions q ON q.id = mr.question_id
                WHERE q.exam_name = ?
            )
            """,
            (exam_name,),
        )

        cursor.execute(
            """
            DELETE FROM model_responses
            WHERE question_id IN (
                SELECT id FROM questions WHERE exam_name = ?
            )
            """,
            (exam_name,),
        )

        cursor.execute("DELETE FROM questions WHERE exam_name = ?", (exam_name,))
        self.conn.commit()

    def response_exists(self, question_id: int, model_name: str) -> bool:
        """Check whether a model response already exists for a question."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT 1
            FROM model_responses
            WHERE question_id = ? AND model_name = ?
            LIMIT 1
            """,
            (question_id, model_name),
        )
        return cursor.fetchone() is not None

    def get_current_question_id(self, question_id: int, exam_name: str, question_number: str) -> int | None:
        """Return the latest available ID for a question, even after re-extraction rewrites rows."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM questions WHERE id = ?", (question_id,))
        row = cursor.fetchone()
        if row is not None:
            return int(row[0])

        cursor.execute(
            """
            SELECT id
            FROM questions
            WHERE exam_name = ? AND question_number = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (exam_name, question_number),
        )
        row = cursor.fetchone()
        return int(row[0]) if row is not None else None
    
    def add_response(self, question_id: int, model_name: str, response: str,
                     latency_ms: int, tokens_used: int, cost_usd: float) -> int:
        """Add a model response"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO model_responses 
            (question_id, model_name, response, latency_ms, tokens_used, cost_usd)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (question_id, model_name, response, latency_ms, tokens_used, cost_usd))
        self.conn.commit()
        return cursor.lastrowid
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


def extract_text_from_pdf(pdf_path: str) -> List[Dict]:
    """
    Extract text from PDF with page information
    Returns list of dicts with page_number and text
    """
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if text:
                pages.append({
                    'page_number': i,
                    'text': text,
                    'tables': page.extract_tables()
                })
    return pages


def save_json(data: any, filepath: str):
    """Save data as JSON"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(filepath: str) -> any:
    """Load JSON data"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)
