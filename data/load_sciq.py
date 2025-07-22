# file: data/load_sciq.py

from datasets import load_dataset

def load_sciq_dataset():
    dataset = load_dataset("sciq")
    return dataset["train"], dataset["validation"], dataset["test"]

# Example usage
if __name__ == "__main__":
    train, val, test = load_sciq_dataset()
    print("Sample:", train[0])
