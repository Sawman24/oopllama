import fitz  # PyMuPDF
import re
import os

def extract_pdf_clean(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = []

    print(f"📖 Extracting: {pdf_path}...")

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        
        # Remove page numbers
        text = re.sub(r'\n\d+\n', '\n', text)
        full_text.append(text)

    text = "\n".join(full_text)

    # Clean the "Legal/Boilerplate"
    # We look for common story starts and ends
    start_markers = ["“What’s two plus two?”", "I’m pretty much fucked.", "I’m in a hospital."]
    found_start = -1
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            found_start = idx
            break
            
    if found_start != -1:
        text = text[found_start:]

    # Stitch lines
    lines = text.split('\n')
    cleaned_lines = []
    current_line = ""

    for line in lines:
        line = line.strip()
        if not line:
            if current_line:
                cleaned_lines.append(current_line)
                current_line = ""
            cleaned_lines.append("")
            continue

        if current_line:
            current_line += " " + line
        else:
            current_line = line

        if line.endswith(('.', '!', '?', '"', '”')):
            cleaned_lines.append(current_line)
            current_line = ""

    if current_line:
        cleaned_lines.append(current_line)

    return "\n\n".join([l for l in cleaned_lines if l.strip()])

def build_mega_dataset():
    all_books_text = []
    
    # Find all PDFs in the current directory
    pdfs = [f for f in os.listdir('.') if f.endswith('.pdf')]
    
    if not pdfs:
        print("⚠️ No PDFs found! Drop them in the folder.")
        return

    for pdf in pdfs:
        book_text = extract_pdf_clean(pdf)
        all_books_text.append(book_text)
        print(f"✅ Added {len(book_text)} characters from {pdf}")

    master_text = "\n\n--- NEW BOOK ---\n\n".join(all_books_text)
    
    with open("master_training_data.txt", 'w', encoding='utf-8') as f:
        f.write(master_text)

    print(f"\n✨ MEGA-DATASET READY! ✨")
    print(f"Total Character Count: {len(master_text)}")
    print(f"Saved to: master_training_data.txt")

if __name__ == "__main__":
    build_mega_dataset()
