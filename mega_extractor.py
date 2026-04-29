import fitz  # PyMuPDF
import re
import os
import unicodedata

def clean_text_thoroughly(text):
    """Aggressively removes PDF artifacts and weird characters."""
    # 1. Normalize unicode (Fixes weird ligatures like 'fi', 'ff')
    text = unicodedata.normalize('NFKC', text)
    
    # 2. Manual fix for common ligatures if normalization missed them
    ligatures = {
        'ﬁ': 'fi', 'ﬀ': 'ff', 'ﬃ': 'ffi', 'ﬄ': 'ffl', 'ﬂ': 'fl', 
        '’': "'", '‘': "'", '“': '"', '”': '"', '—': '-', '…': '...'
    }
    for k, v in ligatures.items():
        text = text.replace(k, v)

    # 3. Keep only printable ASCII and basic punctuation
    # This kills the "ghost" characters ()
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != 'C' or ch in '\n\r\t')
    
    return text

def is_garbage(line):
    garbage_patterns = [
        r'ISBN', r'Copyright', r'All rights reserved', r'Published by',
        r'www\.', r'http', r'\.com', r'Library of Congress',
        r'Printed in', r'First edition', r'Author of', r'Dedication',
        r'Table of Contents', r'Part [IVXLCDM]+', r'Chapter \d+'
    ]
    for pattern in garbage_patterns:
        if re.search(pattern, line, re.IGNORECASE):
            return True
    if re.match(r'^\d+$', line.strip()):
        return True
    return False

def extract_pdf_clean(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = []

    print(f"📖 Anti-Ghost Extraction: {pdf_path}...")

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line or is_garbage(line):
                continue
            full_text.append(line)

    # Join and perform the thorough scrub
    text = " ".join(full_text)
    text = clean_text_thoroughly(text)

    # Spacing and Paragraphs
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([.!?\"]) ', r'\1\n\n', text)

    return text

def build_mega_dataset():
    all_books_text = []
    pdfs = [f for f in os.listdir('.') if f.endswith('.pdf')]
    
    for pdf in pdfs:
        book_text = extract_pdf_clean(pdf)
        all_books_text.append(book_text)
        print(f"✅ Scrubbed {len(book_text)} characters from {pdf}")

    master_text = "\n\n--- NEXT VOLUME ---\n\n".join(all_books_text)
    
    with open("master_training_data.txt", 'w', encoding='utf-8') as f:
        f.write(master_text)

    print(f"\n✨ ANTI-GHOST SCRUB COMPLETE! ✨")
    print(f"Final Data Size: {len(master_text)} characters")

if __name__ == "__main__":
    build_mega_dataset()
