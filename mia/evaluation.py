import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

def evaluate_MIA(score, ground_truth):
    auc = roc_auc_score(y_true=ground_truth, y_score=score)
    return auc # Simple initial metric
