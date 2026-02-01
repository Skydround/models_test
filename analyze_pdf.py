#!/usr/bin/env python3
"""
Quick PDF Analysis - Focus on the main exam PDF
"""

import pdfplumber
import fitz  # PyMuPDF

pdf_path = "pdfs/polski_2025.pdf"

print("="*70)
print("ANALYZING: polski_2025.pdf")
print("="*70)

# Try pdfplumber first
print("\n📄 PDFPLUMBER ANALYSIS\n")
try:
    with pdfplumber.open(pdf_path) as pdf:
        print(f"Total pages: {len(pdf.pages)}\n")
        
        # Analyze first 2 pages in detail
        for i in range(min(2, len(pdf.pages))):
            page = pdf.pages[i]
            print(f"\n{'='*70}")
            print(f"PAGE {i+1}")
            print(f"{'='*70}")
            
            text = page.extract_text()
            if text:
                lines = [l for l in text.split('\n') if l.strip()]
                print(f"\nTotal lines: {len(lines)}")
                print(f"\n--- First 30 lines ---")
                for j, line in enumerate(lines[:30], 1):
                    print(f"{j:3}. {line}")
            
            # Check for tables
            tables = page.extract_tables()
            if tables:
                print(f"\n✓ Found {len(tables)} table(s)")
                for t_idx, table in enumerate(tables, 1):
                    print(f"\n  Table {t_idx}: {len(table)} rows")
                    if table:
                        print(f"  First row: {table[0]}")
        
        # Quick overview of remaining pages
        if len(pdf.pages) > 2:
            print(f"\n\n{'='*70}")
            print(f"QUICK OVERVIEW OF REMAINING PAGES (3-{len(pdf.pages)})")
            print(f"{'='*70}")
            for i in range(2, len(pdf.pages)):
                page = pdf.pages[i]
                text = page.extract_text()
                if text:
                    lines = [l for l in text.split('\n') if l.strip()]
                    print(f"\nPage {i+1}: {len(lines)} lines")
                    # Show first few lines to identify structure
                    print(f"  First line: {lines[0] if lines else 'N/A'}")
                    if len(lines) > 1:
                        print(f"  Second line: {lines[1]}")
                
except Exception as e:
    print(f"❌ pdfplumber error: {e}")

# Also try PyMuPDF for comparison
print(f"\n\n{'='*70}")
print("📄 PyMuPDF ANALYSIS (for comparison)")
print(f"{'='*70}\n")

try:
    doc = fitz.open(pdf_path)
    print(f"Total pages: {len(doc)}")
    
    # Just show first page
    page = doc[0]
    text = page.get_text()
    lines = [l for l in text.split('\n') if l.strip()]
    print(f"\nPage 1: {len(lines)} lines")
    print("\n--- First 30 lines ---")
    for j, line in enumerate(lines[:30], 1):
        print(f"{j:3}. {line}")
    
    doc.close()
    
except Exception as e:
    print(f"❌ PyMuPDF error: {e}")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
