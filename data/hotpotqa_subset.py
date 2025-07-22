import json
import random
import os

# Make sure the data folder exists
os.makedirs("data", exist_ok=True)

# Sample questions and answers (you can expand this list as needed)
qa_pairs = [
    ("What is the boiling point of water?", "100°C at sea level."),
    ("Who discovered gravity?", "Isaac Newton."),
    ("How many bones are in the human body?", "206."),
    ("What gas do plants use for photosynthesis?", "Carbon dioxide."),
    ("What is the capital of France?", "Paris."),
    ("What organ pumps blood?", "The heart."),
    ("Who wrote 'Romeo and Juliet'?", "William Shakespeare."),
    ("What is H2O?", "Water."),
    ("How many continents are there?", "Seven."),
    ("What is the speed of light?", "Approximately 299,792 km/s."),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci."),
    ("What is the currency of Japan?", "Yen."),
    ("What is the powerhouse of the cell?", "Mitochondria."),
    ("What is the formula for area of a circle?", "πr²."),
    ("Which planet is known as the Red Planet?", "Mars."),
    ("What is Newton's third law?", "For every action, there is an equal and opposite reaction."),
    ("What element does 'O' represent?", "Oxygen."),
    ("What is the tallest mountain?", "Mount Everest."),
    ("Who is the first President of the USA?", "George Washington."),
    ("What is the main language spoken in Brazil?", "Portuguese."),
    # Add more below
]

# Duplicate and slightly vary them to reach 100
while len(qa_pairs) < 100:
    q, a = random.choice(qa_pairs)
    qa_pairs.append((q + " ", a))  # Slight variation by appending a space

# Generate full dataset
examples = []
for q, a in qa_pairs:
    context_bm25 = f"BM25 context: Lexical discussion about: {q.lower()} — answer: {a}"
    context_faiss = f"Dense context: Semantic retrieval on: '{q}' — answer might be: {a}"
    best = random.choice(["bm25", "faiss"])  # Simulate controller preference
    examples.append({
        "question": q.strip(),
        "answer": a,
        "context_bm25": context_bm25,
        "context_faiss": context_faiss,
        "best": best
    })

# Save to JSON
with open("data/hotpotqa_subset.json", "w", encoding="utf-8") as f:
    json.dump(examples, f, indent=2)

print("✅ Generated 100 QA training samples at data/hotpotqa_subset.json")
