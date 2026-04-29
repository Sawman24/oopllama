import fitz  # PyMuPDF
import re

def extract_pdf_clean(pdf_path, output_path):
    doc = fitz.open(pdf_path)
    full_text = []

    print(f"📖 Extracting {doc.page_count} pages from {pdf_path}...")

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        # We use 'text' mode which preserves the flow of words
        text = page.get_text("text")
        
        # Basic cleanup per page
        # Remove headers/footers (page numbers at the bottom)
        text = re.sub(r'\n\d+\n', '\n', text)
        
        full_text.append(text)

    # Join everything
    text = "\n".join(full_text)

    # 1. Fix the "Broken Line" artifacts that PDFs often have
    # Join lines that end in a hyphen or don't end in punctuation
    lines = text.split('\n')
    cleaned_lines = []
    current_line = ""

    for line in lines:
        line = line.strip()
        if not line:
            if current_line:
                cleaned_lines.append(current_line)
                current_line = ""
            cleaned_lines.append("") # Keep paragraph breaks
            continue

        if current_line:
            current_line += " " + line
        else:
            current_line = line

        # If the line ends in punctuation, it's likely the end of a sentence/paragraph
        if line.endswith(('.', '!', '?', '"', '”')):
            cleaned_lines.append(current_line)
            current_line = ""

    if current_line:
        cleaned_lines.append(current_line)

    # Final text
    final_text = "\n\n".join([l for l in cleaned_lines if l.strip()])

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_text)

    print(f"✅ Extraction Complete! Saved to {output_path}")
    print(f"Total Character Count: {len(final_text)}")

if __name__ == "__main__":
    extract_pdf_clean("Project Hail Mary.pdf", "hail_mary_perfect.txt")
