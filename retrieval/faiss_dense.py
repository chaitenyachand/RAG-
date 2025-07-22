from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Singleton pattern: shared retriever instance
_faiss_retriever = None

class DenseRetriever:
    _printed_device_msg = False  # Class-level log flag

    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        if not DenseRetriever._printed_device_msg:
            print("Device set to use", self.model.device)
            DenseRetriever._printed_device_msg = True
        self.index = None
        self.contexts = []

    def build_index(self, contexts):
        self.contexts = contexts
        embeddings = self.model.encode(contexts, convert_to_numpy=True, show_progress_bar=False)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def retrieve(self, query, top_k=3):
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)
        return [self.contexts[i] for i in indices[0]]

def build_faiss_index(contexts):
    global _faiss_retriever
    _faiss_retriever = DenseRetriever()
    _faiss_retriever.build_index([item["support"].replace("\n", " ") for item in contexts])

def retrieve_faiss(query, top_k=3):
    global _faiss_retriever
    if _faiss_retriever is None:
        raise ValueError("FAISS retriever not initialized. Call build_faiss_index first.")
    return _faiss_retriever.retrieve(query, top_k)
