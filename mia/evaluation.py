import utils

import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
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

def plot_average_roc_curves(resdir, low_exponent=-4):
    fpr_space = np.logspace(low_exponent, 0, 1000)
    for attack_folder in os.listdir(resdir):
        interpolated_tprs = []
        aurocs = []
        for pickle_file in glob.glob(f"{resdir}/{attack_folder}/*.pkl"):
            with open(pickle_file, "rb") as f:
                metrics = pickle.load(f)
            fpr = metrics["FPR"]
            tpr = metrics["TPR"]
            aurocs.append(metrics["AUC"])
            interp_tpr = np.interp(fpr_space, fpr, tpr)
            interp_tpr[0] = 0.0
            interpolated_tprs.append(interp_tpr)
        if not interpolated_tprs:
            continue
        interpolated_tprs = np.stack(interpolated_tprs, axis=0)
        tpr_mean = np.mean(interpolated_tprs, axis=0)
        auc_mean = np.stack(aurocs, axis=0).mean()
        plt.loglog(fpr_space, tpr_mean, label=f"{attack_folder} | AUC: {auc_mean:.4f}")
    plt.loglog(fpr_space, fpr_space, "k--", label="")
    plt.xlim(1e-4, 1)
    plt.ylim(1e-4, 1)
    plt.grid(True)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.savefig(f"{resdir}/average_roc_curves.png")

if __name__ == "__main__":
    resdir = f"{utils.get_root()}/mia/results/CIFAR10-VAE"
    plot_average_roc_curves(resdir)
