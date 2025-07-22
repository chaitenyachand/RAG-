from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

class DenseRetriever:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.contexts = []

    def build_index(self, data, index_path="faiss_index"):
        self.contexts = [item["support"] for item in data]
        embeddings = self.model.encode(self.contexts, convert_to_numpy=True, show_progress_bar=True)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        faiss.write_index(self.index, index_path)
        with open("contexts.txt", "w", encoding="utf-8") as f:
            for context in self.contexts:
                f.write(context.replace("\n", " ") + "\n")

    def load_index(self, index_path="faiss_index"):
        self.index = faiss.read_index(index_path)
        with open("contexts.txt", "r", encoding="utf-8") as f:
            self.contexts = [line.strip() for line in f.readlines()]

    def retrieve(self, query, top_k=3):
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)
        return [self.contexts[i] for i in indices[0]]