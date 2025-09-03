import os
import glob
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

def evaluate_MIA(score, ground_truth):
    if not isinstance(score, np.ndarray):
        score = np.array(score)
    if not isinstance(ground_truth, np.ndarray):
        ground_truth = np.array(ground_truth)
    auc = roc_auc_score(y_true=ground_truth, y_score=score)
    fpr, tpr, thresholds = roc_curve(y_true=ground_truth, y_score=score)
    return {
        "AUC": auc,
        "FPR": fpr,
        "TPR": tpr,
        "thresholds": thresholds,
    }

def plot_average_roc_curve(resdir, low_exponent=-4):
    mean_fpr = np.logspace(low_exponent, 0, 1000)
    # TODO: complete function
