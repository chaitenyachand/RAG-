# data/download_nq.py
from datasets import load_dataset
import json

def save_nq_subset(num_samples=500, output_file="naturalquestions_subset.json"):
    dataset = load_dataset("nq_open", split="train")
    subset = dataset.select(range(num_samples))

    with open(output_file, 'w', encoding='utf-8') as f:
        for item in subset:
            entry = {
                "question": item["question"],
                "context": " ".join(item["contexts"]),  # Combine all retrieved contexts
                "answer": item["answers"][0] if item["answers"] else ""
            }
            json.dump(entry, f)
            f.write('\n')

save_nq_subset()
