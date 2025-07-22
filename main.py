
from data.load_sciq import load_sciq_dataset
from retrieval.bm25 import create_bm25_index, retrieve_bm25
from retrieval.faiss_dense import retrieve_faiss  # Now using wrapper function
from controller.train_controller import train_controller
from controller.switcher import choose_retriever
from generation.generator import generate_answer
from evaluation.evaluate import compute_f1
from retrieval.faiss_dense import build_faiss_index

print("📥 Loading dataset...")
train, val, test = load_sciq_dataset()

print("📦 Building FAISS index...")
build_faiss_index(train)

print("📦 Creating BM25 index...")
create_bm25_index(train)

print("🔎 Training adaptive controller...")
classifier = train_controller("data/hotpotqa_subset.json")  # Or SciQ subset

print("\n❓ Running RAG++ pipeline")
query = "What causes the moon to shine?"
ground_truth = "The moon shines because sunlight reflects off its surface."

print(f"\n❓ Query: {query}")
chosen = choose_retriever(query, classifier)
print(f"🔀 Chosen Retriever: {chosen.upper()}")

if chosen == "bm25":
    contexts = retrieve_bm25(query)
else:
    contexts = retrieve_faiss(query)

print("\n📚 Top Retrieved Contexts:")
for i, ctx in enumerate(contexts, 1):
    print(f"{i}. {ctx}")

print("\n🧠 Generating answer...")
answer = generate_answer(contexts, query)
print(f"✅ Answer: {answer}")

print("\n📏 Evaluating...")
f1 = compute_f1(answer, ground_truth)
print(f"🎯 F1 Score: {f1:.4f}")
