def clean_story_only(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # The story starts with this iconic line
    start_marker = "“What’s two plus two?”"
    
    # Find the start
    start_idx = content.find(start_marker)
    if start_idx == -1:
        print("❌ Could not find the start of the story!")
        return

    # The story ends with the kids raising their claws
    end_marker = "FOR JOHN, PAUL, GEORGE, AND RINGO"
    end_idx = content.find(end_marker)
    
    if end_idx != -1:
        final_content = content[start_idx:end_idx]
    else:
        final_content = content[start_idx:]

    # Save it back
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(final_content.strip())

    print(f"✅ Story Cleaned! Everything before 'What's two plus two?' and after the author note removed.")

if __name__ == "__main__":
    clean_story_only("hail_mary_perfect.txt")
