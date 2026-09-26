'''Compare sample-level membership attacks across query noise levels.'''

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from .attacks_sample import BASE, LiRA
from .distribution_test import write_tikz_plot
from .evaluation import evaluate_MIA
from .path_utils import parse_loss_signal_path
from .utils import indices_of_shadow_models, load_loss_signals, standardize_signals


ATTACK_NAMES = ("BASE", "LiRA")
METRICS = (
    ("pAUC@1%FPR", "pauc_1pct_fpr_vs_noise_level", "pAUC at 1% FPR", r"pAUC at 1\% FPR"),
    ("TPR@0.1%FPR", "tpr_0p1pct_fpr_vs_noise_level", "TPR at 0.1% FPR", r"TPR at 0.1\% FPR"),
)

def group_loss_paths(loss_dir):
    '''Group loss pickles by noise level and checkpoint; return a complete sweep map.'''
    grouped_paths = {}
    for path in sorted(Path(loss_dir).rglob("*.pkl")):
        metadata = parse_loss_signal_path(path)
        if metadata["signal_type"] != "loss":
            continue
        noise_level = metadata["noise_level"]
        target_stem = metadata["target_stem"]
        level_paths = grouped_paths.setdefault(noise_level, {})
        if target_stem in level_paths:
            raise ValueError(f"Duplicate checkpoint at noise level {noise_level}: {target_stem}")
        level_paths[target_stem] = path
    if len(grouped_paths) < 2:
        raise ValueError("Noise sweep needs loss signals at two or more noise levels.")
    expected_stems = set(next(iter(grouped_paths.values())))
    for noise_level, level_paths in grouped_paths.items():
        if set(level_paths) != expected_stems:
            raise ValueError(f"Checkpoint set differs at noise level {noise_level}.")
    return grouped_paths

def run_noise_sweep(loss_dir, normalization="log_standardized", invert_noise_levels=False):
    '''Run offline attacks on matched checkpoints; return target and mean metrics by noise level.'''
    grouped_paths = group_loss_paths(loss_dir)
    target_stems = sorted(next(iter(grouped_paths.values())))
    target_rows = []
    summary_rows = []
    baseline_masks = None
    expected_shape = None
    for file_level, level_paths in sorted(grouped_paths.items(), reverse=invert_noise_levels):
        noise_level = round(1.0 - file_level, 10) if invert_noise_levels else file_level
        loaded = [load_loss_signals(level_paths[stem]) for stem in target_stems]
        signals = torch.stack([signal for signal, _ in loaded])
        masks = torch.stack([mask for _, mask in loaded])
        if baseline_masks is None:
            baseline_masks = masks
            expected_shape = tuple(signals.shape)
            shadow_indices = [indices_of_shadow_models(index, masks) for index in range(len(target_stems))]
        elif tuple(signals.shape) != expected_shape or not torch.equal(masks, baseline_masks):
            raise ValueError(f"Loss shapes or membership masks differ at noise level {noise_level}.")
        if normalization == "log_standardized":
            signals = standardize_signals(signals, "log")
        elif normalization != "none":
            raise ValueError(f"Unsupported normalization: {normalization}")
        level_rows = []
        for target_index, target_stem in enumerate(target_stems):
            reference_index = shadow_indices[target_index]
            reference_signals = signals[reference_index]
            reference_masks = masks[reference_index]
            target_signals = signals[target_index]
            ground_truth = masks[target_index].to(dtype=torch.long)
            attackers = {
                "BASE": BASE(reference_signals, reference_masks, offline=True),
                "LiRA": LiRA(reference_signals, reference_masks, offline=True),
            }
            for attack_name, attacker in attackers.items():
                scores = attacker.run_attack(target_signals)
                metrics = evaluate_MIA(scores, ground_truth)
                row = {
                    "noise_level": noise_level,
                    "file_level": file_level,
                    "attack": attack_name,
                    "target_stem": target_stem,
                    "n_points": len(ground_truth),
                    "pAUC@1%FPR": metrics["pAUC@1%FPR"],
                    "TPR@0.1%FPR": metrics["TPR@0.1%FPR"],
                }
                level_rows.append(row)
        target_rows.extend(level_rows)
        for attack_name in ATTACK_NAMES:
            attack_rows = [row for row in level_rows if row["attack"] == attack_name]
            summary = {"noise_level": noise_level, "file_level": file_level,
                       "attack": attack_name, "n_targets": len(attack_rows)}
            for metric_name, _, _, _ in METRICS:
                values = np.asarray([row[metric_name] for row in attack_rows])
                summary[f"{metric_name}_mean"] = float(values.mean())
                summary[f"{metric_name}_std"] = float(values.std())
            summary_rows.append(summary)
    return target_rows, summary_rows

def plot_metric(summary_rows, output_dir, metric_name, filename, ylabel, tikz_ylabel):
    '''Save mean attack curves as PNG and standalone TikZ; return both paths.'''
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")
    figure, axis = plt.subplots(figsize=(7, 4.5))
    curves = []
    colors = ("blue", "orange")
    for attack_name, color in zip(ATTACK_NAMES, colors):
        attack_rows = [row for row in summary_rows if row["attack"] == attack_name]
        noise_levels = [row["noise_level"] for row in attack_rows]
        means = [row[f"{metric_name}_mean"] for row in attack_rows]
        axis.plot(noise_levels, means, marker="o", label=attack_name, color=color)
        curves.append((f"{color},thick,mark=*", noise_levels, means, attack_name))
    ticks = np.linspace(0, 1, 11)
    axis.set(xlabel="Noise level", ylabel=ylabel, xlim=(0, 1), ylim=(0, 1),
             xticks=ticks, yticks=ticks)
    axis.tick_params(axis="x", labelrotation=90)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    png_path = output_dir / f"{filename}.png"
    figure.savefig(png_path, dpi=200)
    plt.close(figure)
    tex_path = output_dir / f"{filename}.tex"
    tick_values = ",".join(f"{tick:.2f}" for tick in ticks)
    tick_options = (
        f"xmin=0,xmax=1,ymin=0,ymax=1,xtick={{{tick_values}}},ytick={{{tick_values}}},"
        "xticklabel style={rotate=90,anchor=east}"
    )
    write_tikz_plot(tex_path, "MIA performance by noise level", "Noise level", tikz_ylabel,
                    curves, axis_options=tick_options)
    return png_path, tex_path

def main(argv=None):
    '''Parse CLI options, run the sweep, and write CSV and plot files.'''
    parser = argparse.ArgumentParser(description="Compare MIA performance across loss-query noise levels.")
    parser.add_argument("--loss-dir", required=True, help="Folder of matched loss-signal pickles.")
    parser.add_argument("--output-dir", default="temp_results/noise_sweep", help="Output folder for CSV and plots.")
    parser.add_argument("--normalization", choices=("none", "log_standardized"), default="log_standardized")
    parser.add_argument("--invert-noise-levels", action="store_true", help="Plot 1 minus the filename level when it represents a reverse time index.")
    args = parser.parse_args(argv)
    target_rows, summary_rows = run_noise_sweep(args.loss_dir, args.normalization, args.invert_noise_levels)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / "target_metrics.csv"
    summary_path = output_dir / "summary.csv"
    with target_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=target_rows[0])
        writer.writeheader()
        writer.writerows(target_rows)
    with summary_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=summary_rows[0])
        writer.writeheader()
        writer.writerows(summary_rows)
    plot_paths = []
    for metric_name, filename, ylabel, tikz_ylabel in METRICS:
        plot_paths.extend(plot_metric(summary_rows, output_dir, metric_name, filename, ylabel, tikz_ylabel))
    return {"target_csv": target_path, "summary_csv": summary_path, "plots": plot_paths}

if __name__ == "__main__":
    main()
