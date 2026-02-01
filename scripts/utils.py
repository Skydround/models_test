"""
Shared utilities for the models testing project
"""

import sqlite3
import json
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
            question.correct_answer,
            question.answer_explanation,
            question.page_number,
            question.context_text
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_all_questions(self) -> List[Dict]:
        """Get all questions from database"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM questions")
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
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
