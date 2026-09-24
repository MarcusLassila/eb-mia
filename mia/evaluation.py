import argparse
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from . import path_utils


_SUMMARY_METRICS = (
    "AUC",
    "pAUC@1%FPR",
    "TPR@1%FPR",
    "TPR@0.1%FPR",
    "TPR@0.01%FPR",
)

def _as_numpy_1d(value):
    '''Convert an array-like value to a flat NumPy array.'''
    if isinstance(value, np.ndarray):
        array = value
    elif hasattr(value, "detach"):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    return array.reshape(-1)

def evaluate_MIA(score, ground_truth):
    '''Compute ROC metrics from membership scores and labels.'''
    score = _as_numpy_1d(score)
    ground_truth = _as_numpy_1d(ground_truth)
    auc = roc_auc_score(y_true=ground_truth, y_score=score)
    partial_auc = roc_auc_score(y_true=ground_truth, y_score=score, max_fpr=1e-2)
    fpr, tpr, thresholds = roc_curve(y_true=ground_truth, y_score=score)
    return {
        "AUC": auc,
        "pAUC@1%FPR": partial_auc,
        "TPR@1%FPR": float(np.interp(1e-2, fpr, tpr)),
        "TPR@0.1%FPR": float(np.interp(1e-3, fpr, tpr)),
        "TPR@0.01%FPR": float(np.interp(1e-4, fpr, tpr)),
        "FPR": fpr,
        "TPR": tpr,
        "thresholds": thresholds,
        "n_audit_points": len(ground_truth),
        "score": score,
        "ground_truth": ground_truth,
    }

def collect_grouped_metrics(results_dir):
    '''Load metrics and group target results by model, dataset, and attack.'''
    results_dir = Path(results_dir)
    grouped_metrics = {}
    for metrics_path in sorted(results_dir.glob("metrics_attack-*_target-*.pkl")):
        with open(metrics_path, "rb") as file:
            metrics = pickle.load(file)
        required_keys = {"attack", "target_model", "target_dataset"}
        missing_keys = required_keys.difference(metrics)
        if missing_keys:
            missing_names = ", ".join(sorted(missing_keys))
            raise ValueError(f"Missing {missing_names} in {metrics_path}")
        group = (
            str(metrics["target_model"]),
            str(metrics["target_dataset"]),
            str(metrics["attack"]),
        )
        grouped_metrics.setdefault(group, []).append(metrics)
    if not grouped_metrics:
        raise ValueError(f"No audit metrics found in {results_dir}")
    return grouped_metrics

def summarize_metrics(metrics_list):
    '''Average scalar audit metrics over target models.'''
    summary = {"n_targets": len(metrics_list)}
    for metric_name in _SUMMARY_METRICS:
        values = np.asarray([metrics[metric_name] for metrics in metrics_list], dtype=float)
        summary[metric_name] = float(values.mean())
    return summary

def _write_average_roc_tikz(path, fpr_space, curves, title):
    '''Write averaged ROC curves as a standalone PGFPlots document.'''
    lines = [
        r"\documentclass[tikz]{standalone}",
        r"\usepackage{pgfplots}",
        r"\pgfplotsset{compat=1.18}",
        r"\begin{document}",
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"xmode=log,ymode=log,xlabel={FPR},ylabel={TPR},",
        rf"title={{{title}}},grid=both,legend pos=south east]",
    ]
    for attack, mean_tpr, mean_auc in curves:
        coordinates = " ".join(
            f"({fpr:.8e},{tpr:.8e})"
            for fpr, tpr in zip(fpr_space, mean_tpr)
        )
        label = attack.replace("_", r"\_")
        lines.extend([
            rf"\addplot+[mark=none] coordinates {{{coordinates}}};",
            rf"\addlegendentry{{{label}, AUC={mean_auc:.4f}}}",
        ])
    baseline = " ".join(
        f"({value:.8e},{value:.8e})"
        for value in fpr_space
    )
    lines.extend([
        rf"\addplot+[black,dashed,mark=none] coordinates {{{baseline}}};",
        r"\end{axis}",
        r"\end{tikzpicture}",
        r"\end{document}",
    ])
    path.write_text("\n".join(lines) + "\n")

def plot_average_roc_curves(results_dir, low_exponent=-4):
    '''Plot target-averaged log-log ROC curves by model and dataset.'''
    results_dir = Path(results_dir)
    grouped_metrics = collect_grouped_metrics(results_dir)
    fpr_space = np.logspace(low_exponent, 0, 1000)
    model_dataset_groups = {}
    for (model, dataset, attack), metrics_list in grouped_metrics.items():
        interpolated_tprs = []
        for metrics in metrics_list:
            fpr = np.asarray(metrics["FPR"], dtype=float)
            tpr = np.asarray(metrics["TPR"], dtype=float)
            interpolated_tprs.append(np.interp(fpr_space, fpr, tpr))
        mean_tpr = np.mean(interpolated_tprs, axis=0)
        mean_auc = float(np.mean([metrics["AUC"] for metrics in metrics_list]))
        group = (model, dataset)
        model_dataset_groups.setdefault(group, [])
        model_dataset_groups[group].append((attack, mean_tpr, mean_auc))

    output_paths = []
    for (model, dataset), curves in sorted(model_dataset_groups.items()):
        curves = sorted(curves, key=lambda curve: curve[0])
        title = f"{model} {dataset}"
        output_stem = f"average_roc_curves_{model}-{dataset}"
        png_path = results_dir / f"{output_stem}.png"
        tikz_path = results_dir / f"{output_stem}.tex"

        plt.figure()
        for attack, mean_tpr, mean_auc in curves:
            plt.loglog(
                fpr_space,
                mean_tpr,
                label=f"{attack}, AUC={mean_auc:.4f}",
            )
        plt.loglog(fpr_space, fpr_space, "k--")
        plt.xlim(fpr_space[0], 1)
        plt.ylim(fpr_space[0], 1)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.savefig(png_path, bbox_inches="tight")
        plt.close()

        _write_average_roc_tikz(
            tikz_path,
            fpr_space,
            curves,
            title,
        )
        output_paths.extend([png_path, tikz_path])
    return output_paths

def print_metrics_table(columns, rows, text_columns):
    '''Print string headers and rows with automatic widths and numeric right alignment.
    The first text_columns columns are left aligned; remaining columns are numeric.
    '''
    widths = [max(len(value) for value in column) for column in zip(columns, *rows)]
    for row_index, row in enumerate([columns, *rows]):
        cells = []
        for column_index, (value, width) in enumerate(zip(row, widths)):
            alignment = "<" if column_index < text_columns else ">"
            cells.append(f"{value:{alignment}{width}}")
        print("  ".join(cells))
        if row_index == 0:
            print("  ".join("-" * width for width in widths))

def print_grouped_metrics(results_dir, show_auc=False):
    '''Print target-averaged metrics, optionally including AUC.'''
    grouped_metrics = collect_grouped_metrics(results_dir)
    print("")
    print("Benchmark summary")
    columns = ["Model", "Dataset", "Attack", "Targets"]
    if show_auc:
        columns.append("AUC")
    columns.extend(["pAUC@1%", "TPR@1%", "TPR@0.1%"])
    rows = []
    for (model, dataset, attack), metrics_list in sorted(grouped_metrics.items()):
        summary = summarize_metrics(metrics_list)
        values = [model, dataset, attack, str(summary["n_targets"])]
        if show_auc:
            values.append(f"{summary['AUC']:.4f}")
        values.extend([
            f"{summary['pAUC@1%FPR']:.4f}",
            f"{summary['TPR@1%FPR']:.4f}",
            f"{summary['TPR@0.1%FPR']:.4f}",
        ])
        rows.append(values)
    print_metrics_table(columns, rows, text_columns=3)

def parse_args(argv=None):
    '''Parse the grouped benchmark-summary command line.'''
    parser = argparse.ArgumentParser(description="Summarize an MIA audit benchmark.")
    parser.add_argument("results_dir_name", help="Named directory below the audit results root.")
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--show-auc", action="store_true", help="Include AUC in the printed table.")
    return parser.parse_args(argv)

def main(argv=None):
    '''Print grouped metrics for a completed benchmark.'''
    args = parse_args(argv)
    results_dir = path_utils.audit_results_dir(args.results_dir_name, args.results_root)
    print_grouped_metrics(results_dir, show_auc=args.show_auc)
    plot_average_roc_curves(results_dir)

if __name__ == "__main__":
    main()
