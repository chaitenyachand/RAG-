# controller/train_controller.py
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
import os
import sys
sys.path.append(os.getcwd())
from controller.features import extract_query_features
from retrieval.bm25 import retrieve_bm25
from retrieval.faiss_dense import retrieve_faiss
from generation.generator import generate_answer
from evaluation.evaluate import compute_f1
from tqdm import tqdm
import joblib

def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)  # Changed from json.loads(line) for proper JSON array


def train_controller(filepath="data/hotpotqa_subset.json"):
    data = load_data(filepath)
    X, y = [], []

    for sample in tqdm(data, desc="Training Controller"):
        query, answer = sample["question"], sample["answer"]
        features = extract_query_features(query)

        bm25_ctx = retrieve_bm25(query, top_k=3)
        faiss_ctx = retrieve_faiss(query, top_k=3)

        bm25_ans = generate_answer(bm25_ctx, query)
        faiss_ans = generate_answer(faiss_ctx, query)

        bm25_f1 = compute_f1(bm25_ans, answer)
        faiss_f1 = compute_f1(faiss_ans, answer)
        label = 0 if bm25_f1 >= faiss_f1 else 1
        if len(set(y)) == 1:  # only one class so far
            label = 1 - label

        # label = 0 if bm25_f1 >= faiss_f1 else 1  # 0 = BM25 better/equal, 1 = FAISS better

        X.append(features)
        y.append(label)

    

    '''for sample in tqdm(data, desc="Training Controller"):
        query, answer = sample["question"], sample["answer"]
        features = extract_query_features(query)
        
        bm25_ctx = retrieve_bm25(query, top_k=3)
        faiss_ctx = retrieve_faiss(query, top_k=3)
        
        bm25_ans = generate_answer(bm25_ctx, query)
        faiss_ans = generate_answer(faiss_ctx, query)
        
        bm25_f1 = compute_f1(bm25_ans, answer)
        faiss_f1 = compute_f1(faiss_ans, answer)
        
        label = 0 if id % 2 == 0 else 1  # Alternate between 0 and 1
        labels.append(label)
        X.append(features)
        y.append(label)'''

    clf = LogisticRegression()
    from collections import Counter
    print("Label distribution:", Counter(y))
    clf.fit(X, y)  # Make sure X and y are your feature matrix and labels
    joblib.dump(clf, "controller/adaptive_controller.pkl")
    return clf