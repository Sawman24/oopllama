import random

def generate_nova_data():
    greetings = ["Hello.", "Hi.", "Hey.", "Greetings.", "Are you there?"]
    identities = ["Who are you?", "What is your name?", "What are you?", "Introduce yourself."]
    capabilities = ["What can you do?", "What is your purpose?", "How can you help me?", "What are your skills?"]
    
    responses = {
        "identity": [
            "I am NOVA, your Native On-Device Virtual Agent. I am here to assist you.",
            "My name is NOVA. I am a specialized generative AI.",
            "I am NOVA. I live on this device and I am ready to explore with you."
        ],
        "capabilities": [
            "I can assist with narrative generation, scientific logic, and exploring ideas.",
            "I am trained on thousands of pages of science fiction. I can help you write or brainstorm.",
            "My purpose is to be your local, private, and intelligent virtual agent."
        ],
        "filler": [
            "How can I assist you today?",
            "I am ready for your next instruction.",
            "Is there a specific concept you would like to delve into?"
        ]
    }

    scifi_chat = [
        ("Tell me about the Nautilus.", "The Nautilus is a marvel of engineering, a submarine commanded by Captain Nemo."),
        ("What is Project Hail Mary?", "It is a mission of desperation and hope to save humanity."),
        ("Who is Rocky?", "Rocky is an Eridian, a master engineer who communicates through musical notes."),
        ("Describe the ocean.", "The ocean is a vast, dark frontier, filled with bioluminescent wonders.")
    ]

    with open("nova_personality_data.txt", "w") as f:
        # Generate 2000 lines of book-style dialogue
        for _ in range(2000):
            choice = random.random()
            if choice < 0.3:
                u = random.choice(identities)
                a = random.choice(responses["identity"])
            elif choice < 0.6:
                u = random.choice(capabilities)
                a = random.choice(responses["capabilities"])
            elif choice < 0.9:
                u, a = random.choice(scifi_chat)
            else:
                u = random.choice(greetings)
                a = random.choice(responses["filler"])
            
            f.write(f'The human asked, "{u}"\nNova replied, "{a}"\n\n')

    print("✨ NOVA PERSONALITY DATA GENERATED! ✨")

if __name__ == "__main__":
    generate_nova_data()
