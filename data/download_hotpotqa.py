# data/download_hotpotqa.py
from datasets import load_dataset
import json

def save_subset(dataset, split, num_samples=500, output_file='hotpotqa_subset.json'):
    data = load_dataset("hotpot_qa", split=split)
    subset = data.select(range(num_samples))

    with open(output_file, 'w', encoding='utf-8') as f:
        for item in subset:
            context = " ".join(item["context"][0][1])  # Only first paragraph
            entry = {
                "question": item["question"],
                "context": context,
                "answer": item["answer"]
            }
            json.dump(entry, f)
            f.write('\n')

save_subset("hotpot_qa", "train", 500)
