import re

def clean_alice(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    # 1. Strip Gutenberg Header/Footer using Regex
    header_pattern = re.compile(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE | re.DOTALL)
    footer_pattern = re.compile(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*", re.IGNORECASE | re.DOTALL)
    
    header_match = header_pattern.search(text)
    if header_match:
        text = text[header_match.end():]
    
    footer_match = footer_pattern.search(text)
    if footer_match:
        text = text[:footer_match.start()]

    # 2. Convert Smart Quotes and Dashes to ASCII
    text = text.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
    text = text.replace('—', '--').replace('–', '-')

    # 3. Strip non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # 4. Normalize whitespace (remove \r, consolidate spaces)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[ \t]+', ' ', text)
    
    # 5. Save the clean version
    with open(output_path, 'w', encoding='ascii') as f:
        f.write(text.strip())
    
    print(f"✅ Dataset Cleaned! Character count: {len(text)}")

if __name__ == "__main__":
    clean_alice('alice.txt', 'alice_clean.txt')
