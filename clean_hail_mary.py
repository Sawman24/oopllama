import re

def clean_hail_mary(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # 1. Join with spaces
    text = " ".join([line.strip() for line in lines])
    text = re.sub(r'\s+', ' ', text)

    # 2. Targeted "Stitcher": Fix common broken patterns
    # These are the ones we keep seeing in your output
    stitches = {
        r'\ba nd\b': 'and',
        r'\ba re\b': 'are',
        r'\ba s\b': 'as',
        r'\ba t\b': 'at',
        r'\ba n\b': 'an',
        r'\ba a\b': 'a',
        r'\bH ail\b': 'Hail',
        r'\bM ary\b': 'Mary',
        r'\ba t\b': 'at',
        r'\ba l\b': 'al',
        r'\ba r\b': 'ar',
        r'\ba m\b': 'am',
        r'\bi n\b': 'in',
        r'\bi t\b': 'it',
        r'\bi s\b': 'is',
        r'\bf st\b': 'fast',
        r'\bf ster\b': 'faster',
        r'\bsp ace\b': 'space',
        r'\bth t\b': 'that',
        r'\btha t\b': 'that',
        r'\ba ny\b': 'any',
        r'\ba ll\b': 'all',
        r'\ba bout\b': 'about',
        r'\ba fter\b': 'after',
        r'\ba gain\b': 'again',
        r'\ba lways\b': 'always',
        r'\ba nother\b': 'another',
        r'\ba nything\b': 'anything',
        r'\ba round\b': 'around',
        r'\ba sk\b': 'ask',
        r'\ba way\b': 'away',
    }

    for pattern, replacement in stitches.items():
        text = re.sub(pattern, replacement, text)

    # 3. Fuzzy Stitcher: Join single capital letters to the word following them
    # Fixes "H ail", "M ary", "S t tes"
    text = re.sub(r'\b([A-Z])\s+([a-z]{2,})\b', r'\1\2', text)
    
    # 4. Fuzzy Stitcher: Join single 'a' or 'i' to fragments that look like words
    # e.g. "a nd", "i s", "i t"
    text = re.sub(r'\ba\s+([nd|re|s|t|n|l|m|r|y])\b', r'a\1', text)
    text = re.sub(r'\bi\s+([s|t|n])\b', r'i\1', text)

    # 5. Final normalization
    text = re.sub(r'\s+', ' ', text)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)

    print(f"🚀 Project Hail Mary Healed (v3)! Character count: {len(text)}")

if __name__ == "__main__":
    clean_hail_mary("input.txt", "hail_mary_clean.txt")
