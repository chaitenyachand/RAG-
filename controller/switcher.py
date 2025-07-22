def hybrid_score(bm25_results, dense_results, top_k=3, bm25_weight=0.5, dense_weight=0.5):
    scores = {}
    for i, res in enumerate(bm25_results):
        scores[res] = scores.get(res, 0) + bm25_weight * (top_k - i)
    for i, res in enumerate(dense_results):
        scores[res] = scores.get(res, 0) + dense_weight * (top_k - i)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [r[0] for r in ranked[:top_k]]


# controller/switcher.py
import os
import sys
sys.path.append(os.getcwd())
from controller.features import extract_query_features

def choose_retriever(query, classifier_model):
    features = extract_query_features(query).reshape(1, -1)
    pred = classifier_model.predict(features)
    return "bm25" if pred[0] == 0 else "faiss"
