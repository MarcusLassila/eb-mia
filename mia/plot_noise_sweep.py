from . import attacks_sample
from . import path_utils
from .utils import indices_of_shadow_models, load_loss_signals

import argparse
from pathlib import Path
import csv
import pickle

import numpy as np
import torch

def loss_signal_paths_by_noise_level(loss_dir):
    '''
    Group loss-signal pickle paths by query noise level.
    Args:
        loss_dir (str | Path): Directory containing loss-signal pickles.
    Returns:
        dict[float, list[Path]]: Loss-signal paths grouped by noise level.
    '''
    loss_dir = Path(loss_dir)
    loss_paths = sorted(loss_dir.glob("loss-signals-*.pkl"))
    if not loss_paths:
        loss_paths = sorted(loss_dir.glob("loss_signals-*.pkl"))
    if not loss_paths:
        loss_paths = sorted(loss_dir.rglob("loss-signals-*.pkl"))
    if not loss_paths:
        loss_paths = sorted(loss_dir.rglob("loss_signals-*.pkl"))
    if not loss_paths:
        raise ValueError(f"No loss-signal pickle files found in {loss_dir}.")
    grouped_paths = {}
    for loss_path in loss_paths:
        noise_level = path_utils.parse_loss_signal_path(loss_path)["noise_level"]
        grouped_paths.setdefault(noise_level, [])
        grouped_paths[noise_level].append(loss_path)
    return grouped_paths

def run_base_offline_round_robin(loss_paths, prior=0.5):
    '''
    Run BASE offline sample-level round-robin auditing for one noise level.
    Args:
        loss_paths (list[Path]): Loss-signal pickle paths sharing one noise level.
        prior (float): Membership prior for the BASE attack.
    Returns:
        list[dict]: Per-target audit metrics.
    '''
    from .evaluation import evaluate_MIA

    loss_sigs = []
    train_masks = []
    for loss_path in loss_paths:
        loss_sig, train_mask = load_loss_signals(loss_path)
        loss_sigs.append(loss_sig)
        train_masks.append(train_mask)
    loss_sigs = torch.stack(loss_sigs)
    train_masks = torch.stack(train_masks)

    metrics_list = []
    for target_idx, target_loss_path in enumerate(loss_paths):
        shadow_indices = indices_of_shadow_models(target_idx, train_masks)
        attacker = attacks_sample.BASE(
            shadow_loss_sigs=loss_sigs[shadow_indices],
            shadow_train_mask=train_masks[shadow_indices],
            offline=True,
            prior=prior,
        )
        scores = attacker.run_attack(loss_sigs[target_idx])
        metrics = evaluate_MIA(
            score=scores,
            ground_truth=train_masks[target_idx].to(dtype=torch.long),
        )
        metrics["target_loss_path"] = str(target_loss_path)
        metrics_list.append(metrics)
    return metrics_list

def summarize_metrics(noise_level, metrics_list):
    '''
    Summarize per-target audit metrics for one noise level.
    Args:
        noise_level (float): Query noise level.
        metrics_list (list[dict]): Per-target audit metrics.
    Returns:
        dict: Mean and standard deviation summary.
    '''
    summary = {
        "noise_level": float(noise_level),
        "n_targets": len(metrics_list),
    }
    for metric_name in ("AUC", "pAUC@1%FPR", "TPR@1%FPR", "TPR@0.1%FPR"):
        values = np.asarray([metrics[metric_name] for metrics in metrics_list], dtype=float)
        summary[f"{metric_name}_mean"] = float(values.mean())
        summary[f"{metric_name}_std"] = float(values.std())
    return summary

def run_noise_sweep(loss_dir, prior=0.5):
    '''
    Run BASE offline round-robin audits grouped by noise level.
    Args:
        loss_dir (str | Path): Directory containing loss-signal pickles.
        prior (float): Membership prior for the BASE attack.
    Returns:
        tuple[list[dict], dict[float, list[dict]]]: Summaries and per-target metrics.
    '''
    grouped_paths = loss_signal_paths_by_noise_level(loss_dir)
    summaries = []
    metrics_by_noise_level = {}
    for noise_level, loss_paths in sorted(grouped_paths.items()):
        metrics_list = run_base_offline_round_robin(loss_paths, prior=prior)
        metrics_by_noise_level[noise_level] = metrics_list
        summaries.append(summarize_metrics(noise_level, metrics_list))
    return summaries, metrics_by_noise_level

def write_summary_csv(output_dir, summaries):
    '''
    Write noise-sweep metric summaries as CSV.
    Args:
        output_dir (str | Path): Output directory.
        summaries (list[dict]): Metric summaries.
    Returns:
        Path: Saved CSV path.
    '''
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "base_offline_noise_sweep_summary.csv"
    fieldnames = [
        "noise_level",
        "n_targets",
        "AUC_mean",
        "AUC_std",
        "pAUC@1%FPR_mean",
        "pAUC@1%FPR_std",
        "TPR@1%FPR_mean",
        "TPR@1%FPR_std",
        "TPR@0.1%FPR_mean",
        "TPR@0.1%FPR_std",
    ]
    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary)
    return csv_path

def write_metrics_pickle(output_dir, summaries, metrics_by_noise_level):
    '''
    Write noise-sweep summaries and per-target metrics as pickle.
    Args:
        output_dir (str | Path): Output directory.
        summaries (list[dict]): Metric summaries.
        metrics_by_noise_level (dict[float, list[dict]]): Per-target metrics.
    Returns:
        Path: Saved pickle path.
    '''
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pickle_path = output_dir / "base_offline_noise_sweep_metrics.pkl"
    with open(pickle_path, "wb") as file:
        pickle.dump(
            {
                "summaries": summaries,
                "metrics_by_noise_level": metrics_by_noise_level,
            },
            file,
        )
    return pickle_path

def plot_metric_vs_noise_level(output_dir, summaries, metric_name, ylabel, filename):
    '''
    Plot one MIA metric against query noise level.
    Args:
        output_dir (str | Path): Output directory.
        summaries (list[dict]): Metric summaries.
        metric_name (str): Metric key prefix.
        ylabel (str): Y-axis label.
        filename (str): Output PNG filename.
    Returns:
        Path: Saved plot path.
    '''
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    noise_levels = np.asarray([summary["noise_level"] for summary in summaries], dtype=float)
    means = np.asarray([summary[f"{metric_name}_mean"] for summary in summaries], dtype=float)
    stds = np.asarray([summary[f"{metric_name}_std"] for summary in summaries], dtype=float)

    plt.figure(figsize=(7, 4.5))
    plt.plot(noise_levels, means, marker="o", color="#1b4d3e")
    plt.fill_between(noise_levels, means - stds, means + stds, color="#1b4d3e", alpha=0.18)
    plt.xlabel("Noise level")
    plt.ylabel(ylabel)
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = output_dir / filename
    plt.savefig(plot_path, dpi=200)
    plt.close()
    return plot_path

def write_noise_sweep_plots(output_dir, summaries):
    '''
    Write the requested MIA-performance-vs-noise plots.
    Args:
        output_dir (str | Path): Output directory.
        summaries (list[dict]): Metric summaries.
    Returns:
        list[Path]: Saved plot paths.
    '''
    return [
        plot_metric_vs_noise_level(output_dir, summaries, "AUC", "AUC", "auc_vs_noise_level.png"),
        plot_metric_vs_noise_level(output_dir, summaries, "pAUC@1%FPR", "pAUC@1%FPR", "pauc_1pct_fpr_vs_noise_level.png"),
        plot_metric_vs_noise_level(output_dir, summaries, "TPR@1%FPR", "TPR@1%FPR", "tpr_1pct_fpr_vs_noise_level.png"),
        plot_metric_vs_noise_level(output_dir, summaries, "TPR@0.1%FPR", "TPR@0.1%FPR", "tpr_0p1pct_fpr_vs_noise_level.png"),
    ]

def parse_args(argv=None):
    '''
    Parse CLI arguments for noise-sweep plotting.
    Args:
        argv (list[str] | None): Optional command-line arguments.
    Returns:
        argparse.Namespace: Parsed CLI arguments.
    '''
    parser = argparse.ArgumentParser(description="Plot BASE offline MIA performance against query noise level.")
    parser.add_argument(
        "--loss-dir",
        default="mia/results/FM-CelebA-loss-sigs",
        help="Directory containing loss-signal pickle files.",
    )
    parser.add_argument(
        "--output-dir",
        default="mia/results/FM-CelebA-loss-sigs/noise_sweep",
        help="Directory for summary files and plots.",
    )
    parser.add_argument(
        "--prior",
        type=float,
        default=0.5,
        help="Membership prior for BASE offline.",
    )
    return parser.parse_args(argv)

def main(argv=None):
    '''
    Entry point for noise-sweep plotting.
    Args:
        argv (list[str] | None): Optional command-line arguments.
    Returns:
        None
    '''
    args = parse_args(argv)
    summaries, metrics_by_noise_level = run_noise_sweep(args.loss_dir, prior=args.prior)
    output_dir = Path(args.output_dir)
    csv_path = write_summary_csv(output_dir, summaries)
    pickle_path = write_metrics_pickle(output_dir, summaries, metrics_by_noise_level)
    plot_paths = write_noise_sweep_plots(output_dir, summaries)
    print(f"Saved summary CSV to {csv_path}")
    print(f"Saved metrics pickle to {pickle_path}")
    for plot_path in plot_paths:
        print(f"Saved plot to {plot_path}")

if __name__ == "__main__":
    main()
