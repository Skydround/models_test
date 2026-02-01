#!/usr/bin/env python3
"""
Setup checker - Verify everything is ready to run
"""

import os
import sys
from pathlib import Path

def check_mark(condition):
    return "✅" if condition else "❌"

def main():
    print("="*70)
    print("SETUP VERIFICATION")
    print("="*70)
    
    all_good = True
    
    # Check virtual environment
    print("\n📦 Virtual Environment:")
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    print(f"  {check_mark(in_venv)} Virtual environment active")
    if not in_venv:
        print("     Run: source venv/bin/activate")
        all_good = False
    
    # Check dependencies
    print("\n📚 Dependencies:")
    
    try:
        import pdfplumber
        print(f"  ✅ pdfplumber")
    except ImportError:
        print(f"  ❌ pdfplumber")
        all_good = False
    
    try:
        import anthropic
        print(f"  ✅ anthropic")
    except ImportError:
        print(f"  ❌ anthropic")
        all_good = False
    
    try:
        import openai
        print(f"  ✅ openai")
    except ImportError:
        print(f"  ❌ openai")
        all_good = False
    
    try:
        import google.generativeai
        print(f"  ✅ google-generativeai")
    except ImportError:
        print(f"  ❌ google-generativeai")
        all_good = False
    
    try:
        import pandas
        print(f"  ✅ pandas")
    except ImportError:
        print(f"  ❌ pandas")
        all_good = False
    
    try:
        import openpyxl
        print(f"  ✅ openpyxl")
    except ImportError:
        print(f"  ❌ openpyxl")
        all_good = False
    
    try:
        import tqdm
        print(f"  ✅ tqdm")
    except ImportError:
        print(f"  ❌ tqdm")
        all_good = False
    
    # Check PDFs
    print("\n📄 PDF Files:")
    pdfs = [
        'pdfs/polski_2025.pdf',
        'pdfs/polski_2025_odp.pdf',
        'pdfs/matematyka_2025.pdf',
        'pdfs/matematyka_2025_odp.pdf'
    ]
    for pdf in pdfs:
        exists = Path(pdf).exists()
        size = Path(pdf).stat().st_size if exists else 0
        print(f"  {check_mark(exists)} {pdf} ({size/1024:.0f} KB)" if exists else f"  ❌ {pdf} - Missing!")
        if not exists or size == 0:
            all_good = False
    
    # Check .env file
    print("\n🔑 Environment Variables:")
    env_exists = Path('.env').exists()
    print(f"  {check_mark(env_exists)} .env file exists")
    
    if env_exists:
        from dotenv import load_dotenv
        load_dotenv()
        
        keys = {
            'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
            'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
            'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY')
        }
        
        for key, value in keys.items():
            has_key = value and value != f'your_{key.lower().split("_")[0]}_key_here'
            print(f"  {check_mark(has_key)} {key}")
            if not has_key:
                print(f"     Add to .env file")
                all_good = False
    else:
        print("     Run: cp .env.example .env")
        print("     Then edit .env and add your API keys")
        all_good = False
    
    # Check directories
    print("\n📁 Directories:")
    dirs = ['data', 'results', 'scripts']
    for d in dirs:
        exists = Path(d).exists()
        print(f"  {check_mark(exists)} {d}/")
    
    # Summary
    print("\n" + "="*70)
    if all_good:
        print("✅ ALL CHECKS PASSED - Ready to run!")
        print("\nNext step:")
        print("  ./run_pipeline.sh")
        print("  or")
        print("  python scripts/1_extract_questions.py")
    else:
        print("❌ SOME CHECKS FAILED - Fix issues above before running")
        print("\nQuick fixes:")
        print("  1. source venv/bin/activate")
        print("  2. pip install -r requirements.txt")
        print("  3. cp .env.example .env && nano .env")
        print("  4. Add your API keys to .env")
    print("="*70)

if __name__ == "__main__":
    main()
