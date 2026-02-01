#!/usr/bin/env python3
"""
Analyze answer key PDFs to understand their structure
"""

import pdfplumber
import sys

def analyze_answer_pdf(pdf_path, name):
    print(f"\n{'='*70}")
    print(f"📋 ANSWER KEY ANALYSIS: {name}")
    print(f"{'='*70}\n")
    
    with pdfplumber.open(pdf_path) as pdf:
        print(f"Total pages: {len(pdf.pages)}\n")
        
        # Analyze first 2 pages
        for i in range(min(2, len(pdf.pages))):
            page = pdf.pages[i]
            print(f"\n--- Page {i+1} ---")
            
            text = page.extract_text()
            if text:
                lines = [l for l in text.split('\n') if l.strip()]
                print(f"Total lines: {len(lines)}")
                print(f"\nFirst 40 lines:")
                for j, line in enumerate(lines[:40], 1):
                    print(f"{j:3}. {line}")
            
            # Check for tables (answers might be in tables)
            tables = page.extract_tables()
            if tables:
                print(f"\n✓ Found {len(tables)} table(s)")
                for t_idx, table in enumerate(tables[:2], 1):  # Show first 2 tables
                    print(f"\n  Table {t_idx}: {len(table)} rows × {len(table[0]) if table and table[0] else 0} cols")
                    if table:
                        print(f"  First 5 rows:")
                        for row_idx, row in enumerate(table[:5], 1):
                            print(f"    {row_idx}. {row}")

# Analyze both answer keys
analyze_answer_pdf("pdfs/polski_2025_odp.pdf", "Polski 2025 - Answers")
analyze_answer_pdf("pdfs/matematyka_2025_odp.pdf", "Matematyka 2025 - Answers")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
