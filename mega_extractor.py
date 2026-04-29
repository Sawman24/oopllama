import fitz  # PyMuPDF
import re
import os

def is_garbage(line):
    """Detects if a line is likely boilerplate or metadata."""
    garbage_patterns = [
        r'ISBN', r'Copyright', r'All rights reserved', r'Published by',
        r'www\.', r'http', r'\.com', r'Library of Congress',
        r'Printed in', r'First edition', r'Author of', r'Dedication',
        r'Table of Contents', r'Part [IVXLCDM]+', r'Chapter \d+'
    ]
    for pattern in garbage_patterns:
        if re.search(pattern, line, re.IGNORECASE):
            return True
    
    # Remove lines that are just numbers (page numbers)
    if re.match(r'^\d+$', line.strip()):
        return True
        
    return False

def extract_pdf_clean(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = []

    print(f"📖 Surgically Extracting: {pdf_path}...")

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip page numbers and boilerplate
            if is_garbage(line):
                continue
                
            full_text.append(line)

    # Stitch the words into a continuous narrative
    text = " ".join(full_text)

    # Fix spacing artifacts (double spaces, etc.)
    text = re.sub(r'\s+', ' ', text)
    
    # Optional: Break into paragraphs by looking for sentence ends
    # This helps the model understand structure better
    text = re.sub(r'([.!?\u201d\"]) ', r'\1\n\n', text)

    return text

def build_mega_dataset():
    all_books_text = []
    pdfs = [f for f in os.listdir('.') if f.endswith('.pdf')]
    
    if not pdfs:
        print("⚠️ No PDFs found!")
        return

    for pdf in pdfs:
        book_text = extract_pdf_clean(pdf)
        all_books_text.append(book_text)
        print(f"✅ Cleaned {len(book_text)} characters from {pdf}")

    master_text = "\n\n--- NEXT VOLUME ---\n\n".join(all_books_text)
    
    with open("master_training_data.txt", 'w', encoding='utf-8') as f:
        f.write(master_text)

    print(f"\n✨ SURGICAL CLEAN COMPLETE! ✨")
    print(f"Final Data Size: {len(master_text)} characters")

if __name__ == "__main__":
    build_mega_dataset()
