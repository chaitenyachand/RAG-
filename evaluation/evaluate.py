from sklearn.metrics import f1_score

def compute_f1(predicted, ground_truth):
    def normalize(text):
        return text.strip().lower()
    return f1_score([normalize(ground_truth)], [normalize(predicted)], average='macro')
