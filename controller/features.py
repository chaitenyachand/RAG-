# controller/features.py
import numpy as np

def extract_query_features(query):
    words = query.split()
    length = len(words)
    avg_word_length = sum(len(w) for w in words) / (length or 1)
    unique_ratio = len(set(words)) / (length or 1)
    return np.array([length, avg_word_length, unique_ratio])
