from . import path_utils
import utils

import argparse
from pathlib import Path
import pickle
import yaml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

def evaluate_MIA(score, ground_truth):
    '''
    Evaluate MIA scores and return ROC-derived metrics.
    Args:
        score (array-like): Audit scores.
        ground_truth (array-like): Membership labels.
    Returns:
        dict: ROC-derived audit metrics.
    '''
    if not isinstance(score, np.ndarray):
        score = np.array(score)
    if not isinstance(ground_truth, np.ndarray):
        ground_truth = np.array(ground_truth)
    auc = roc_auc_score(y_true=ground_truth, y_score=score)
    fpr, tpr, thresholds = roc_curve(y_true=ground_truth, y_score=score)
    tpr_at_1pct_fpr = float(np.interp(1e-2, fpr, tpr))
    tpr_at_0p1pct_fpr = float(np.interp(1e-3, fpr, tpr))
    return {
        "AUC": auc,
        "TPR@1%FPR": tpr_at_1pct_fpr,
        "TPR@0.1%FPR": tpr_at_0p1pct_fpr,
        "FPR": fpr,
        "TPR": tpr,
        "thresholds": thresholds,
        "n_audit_points": len(ground_truth),
    }

def _assert_fixed_fpr_metrics_consistent(metrics):
    '''
    Assert stored TPR@FPR metrics match the saved ROC curve.
    Args:
        metrics (dict): Metrics dictionary to validate.
    Returns:
        None
    '''
    fpr = np.asarray(metrics["FPR"], dtype=float)
    tpr = np.asarray(metrics["TPR"], dtype=float)
    assert np.isclose(float(metrics["TPR@1%FPR"]), float(np.interp(1e-2, fpr, tpr)), atol=1e-7)
    assert np.isclose(float(metrics["TPR@0.1%FPR"]), float(np.interp(1e-3, fpr, tpr)), atol=1e-7)

def _collect_metrics_folder_summary(metrics_dir, fpr_space):
    '''
    Collect summary stats and an averaged ROC curve for one metrics folder.
    Args:
        metrics_dir (str | Path): Folder containing metrics pickles.
        fpr_space (np.ndarray): FPR grid used for interpolation.
    Returns:
        dict: Summary metrics for the folder.
    '''
    metrics_dir = Path(metrics_dir)
    metrics_files = sorted(metrics_dir.glob("*.pkl"))
    if not metrics_files:
        raise ValueError(f"No metrics pickle files found in {metrics_dir}")

    target_stems = []
    auc_values = []
    tpr_1pct_values = []
    tpr_0p1pct_values = []
    interpolated_tprs = []
    for metrics_file in metrics_files:
        with open(metrics_file, "rb") as f:
            metrics = pickle.load(f)
        _assert_fixed_fpr_metrics_consistent(metrics)
        target_stems.append(path_utils.target_stem_from_metrics_pickle_path(metrics_file))
        fpr = np.asarray(metrics["FPR"], dtype=float)
        tpr = np.asarray(metrics["TPR"], dtype=float)
        interp_tpr = np.interp(fpr_space, fpr, tpr)
        interpolated_tprs.append(interp_tpr)
        auc_values.append(float(metrics["AUC"]))
        tpr_1pct_values.append(float(metrics["TPR@1%FPR"]))
        tpr_0p1pct_values.append(float(metrics["TPR@0.1%FPR"]))

    auc_values = np.asarray(auc_values, dtype=float)
    tpr_1pct_values = np.asarray(tpr_1pct_values, dtype=float)
    tpr_0p1pct_values = np.asarray(tpr_0p1pct_values, dtype=float)
    interpolated_tprs = np.stack(interpolated_tprs, axis=0)
    return {
        "path": metrics_dir,
        "target_stems": tuple(sorted(set(target_stems))),
        "mean_tpr": np.mean(interpolated_tprs, axis=0),
        "AUC": {"mean": float(np.mean(auc_values)), "std": float(np.std(auc_values))},
        "TPR@1%FPR": {"mean": float(np.mean(tpr_1pct_values)), "std": float(np.std(tpr_1pct_values))},
        "TPR@0.1%FPR": {"mean": float(np.mean(tpr_0p1pct_values)), "std": float(np.std(tpr_0p1pct_values))},
    }

def collect_metrics_folder_summaries(metrics_dirs, low_exponent=-4):
    '''
    Collect summaries for metrics folders and validate shared target models.
    Args:
        metrics_dirs (list[str | Path]): Metrics folders to evaluate.
        low_exponent (int): Lowest exponent used for the log-spaced FPR grid.
    Returns:
        tuple[np.ndarray, list[dict]]: FPR grid and folder summaries.
    '''
    fpr_space = np.logspace(low_exponent, 0, 1000)
    summaries = [_collect_metrics_folder_summary(metrics_dir, fpr_space) for metrics_dir in metrics_dirs]
    if not summaries:
        raise ValueError("No metrics folders provided.")
    reference_targets = summaries[0]["target_stems"]
    common_meta = path_utils.common_target_meta(reference_targets)
    for summary in summaries[1:]:
        if summary["target_stems"] != reference_targets:
            raise ValueError("All metrics folders must be computed on the same target models.")
        path_utils.common_target_meta(summary["target_stems"])
    for summary in summaries:
        summary["label"] = path_utils.metrics_folder_label(summary["path"], common_meta)
    return fpr_space, summaries

def _latex_escape(text):
    '''
    Escape special LaTeX characters in plain text.
    Args:
        text (str): Input text.
    Returns:
        str: Escaped text safe for LaTeX.
    '''
    escaped_text = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for source, target in replacements.items():
        escaped_text = escaped_text.replace(source, target)
    return escaped_text

def _tikz_coordinates(x_values, y_values):
    '''
    Format paired x/y values as pgfplots coordinates.
    Args:
        x_values (np.ndarray): X-axis values.
        y_values (np.ndarray): Y-axis values.
    Returns:
        str: TikZ coordinate list.
    '''
    coordinate_rows = []
    for x_value, y_value in zip(x_values, y_values):
        coordinate_rows.append(f"({float(x_value):.8e},{float(y_value):.8e})")
    return " ".join(coordinate_rows)

def write_average_roc_tikz_plot(output_dir, fpr_space, summaries):
    '''
    Write a pgfplots TikZ file for averaged ROC curves.
    Args:
        output_dir (str | Path): Directory for the saved figure.
        fpr_space (np.ndarray): FPR grid used for plotting.
        summaries (list[dict]): Metrics summaries to plot.
    Returns:
        Path: Saved TikZ file path.
    '''
    output_dir = Path(output_dir)
    reference_targets = summaries[0]["target_stems"]
    for summary in summaries[1:]:
        if summary["target_stems"] != reference_targets:
            raise ValueError("All metrics folders must be computed on the same target models.")
    common_meta = path_utils.common_target_meta(reference_targets)
    size_label = f"sz{common_meta['size']}" + ("-gray" if common_meta["gray"] else "")
    title = (
        f"{common_meta['model']}-{common_meta['dataset']}-{size_label}"
        f" | {len(reference_targets)} target models"
    )
    tikz_lines = [
        r"\documentclass[tikz]{standalone}",
        r"\usepackage{pgfplots}",
        r"\pgfplotsset{compat=1.18}",
        r"\begin{document}",
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"width=12cm,",
        r"height=9cm,",
        r"xmode=log,",
        r"ymode=log,",
        f"xmin={float(fpr_space[0]):.8e},",
        r"xmax=1,",
        f"ymin={float(fpr_space[0]):.8e},",
        r"ymax=1,",
        r"xlabel={FPR},",
        r"ylabel={TPR},",
        f"title={{{_latex_escape(title)}}},",
        r"grid=both,",
        r"legend pos=south east,",
        r"]",
    ]
    for summary in summaries:
        auc_stats = summary["AUC"]
        label = (
            f"{summary['label']} | "
            f"AUC: {100*auc_stats['mean']:.2f}% ± {100*auc_stats['std']:.2f}%"
        )
        coordinates = _tikz_coordinates(fpr_space, summary["mean_tpr"])
        tikz_lines.append(r"\addplot+[mark=none] coordinates {")
        tikz_lines.append(coordinates)
        tikz_lines.append(r"};")
        tikz_lines.append(f"\\addlegendentry{{{_latex_escape(label)}}}")
    baseline_coordinates = _tikz_coordinates(fpr_space, fpr_space)
    tikz_lines.append(r"\addplot+[black, dashed, mark=none] coordinates {")
    tikz_lines.append(baseline_coordinates)
    tikz_lines.append(r"};")
    tikz_lines.extend([
        r"\end{axis}",
        r"\end{tikzpicture}",
        r"\end{document}",
    ])
    output_dir.mkdir(parents=True, exist_ok=True)
    tikz_path = output_dir / f"average_roc_curves_{summaries[0]['path'].stem}.tex"
    tikz_path.write_text("\n".join(tikz_lines) + "\n")
    return tikz_path

def plot_average_roc_curves(output_dir, fpr_space, summaries):
    '''
    Plot averaged ROC curves for summaries.
    Args:
        output_dir (str | Path): Directory for the saved figure.
        fpr_space (np.ndarray): FPR grid used for plotting.
        summaries (list[dict]): Metrics summaries to plot.
    Returns:
        None
    '''
    output_dir = Path(output_dir)
    reference_targets = summaries[0]["target_stems"]
    for summary in summaries[1:]:
        if summary["target_stems"] != reference_targets:
            raise ValueError("All metrics folders must be computed on the same target models.")
    common_meta = path_utils.common_target_meta(reference_targets)
    size_label = f"sz{common_meta['size']}" + ("-gray" if common_meta["gray"] else "")
    title = (
        f"{common_meta['model']}-{common_meta['dataset']}-{size_label}"
        f" | {len(reference_targets)} target models"
    )

    plt.figure()
    for summary in summaries:
        auc_stats = summary["AUC"]
        plt.loglog(
            fpr_space,
            summary["mean_tpr"],
            label=(
                f"{summary['label']} | "
                f"AUC: {100*auc_stats['mean']:.2f}% ± {100*auc_stats['std']:.2f}%"
            ),
        )
    plt.loglog(fpr_space, fpr_space, "k--", label="")
    plt.xlim(float(fpr_space[0]), 1)
    plt.ylim(float(fpr_space[0]), 1)
    plt.grid(True)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(title)
    plt.legend()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"average_roc_curves_{summaries[0]['path'].stem}.png")
    plt.close()
    write_average_roc_tikz_plot(output_dir, fpr_space, summaries)
    print(f"Saved average roc plot in {output_dir}")

def _format_pct_mean_std(stats):
    '''
    Format a metric as percentage mean plus-minus standard deviation.
    Args:
        stats (dict): Metric statistics with mean and std values.
    Returns:
        str: Formatted percentage string.
    '''
    return f"{100*stats['mean']:.2f}% ± {100*stats['std']:.2f}%"

def print_metrics_table(summaries):
    '''
    Print one combined table for all metrics folders.
    Args:
        summaries (list[dict]): Metrics summaries to print.
    Returns:
        None
    '''
    headers = [
        "Metrics folder",
        "AUC (%)",
        "TPR@1%FPR (%)",
        "TPR@0.1%FPR (%)",
    ]
    rows = [
        [
            summary["label"],
            _format_pct_mean_std(summary["AUC"]),
            _format_pct_mean_std(summary["TPR@1%FPR"]),
            _format_pct_mean_std(summary["TPR@0.1%FPR"]),
        ]
        for summary in summaries
    ]
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))
    print("")
    print("Evaluation summary")
    print(
        f"{headers[0]:<{widths[0]}} "
        f"{headers[1]:>{widths[1]}} "
        f"{headers[2]:>{widths[2]}} "
        f"{headers[3]:>{widths[3]}}"
    )
    print(f"{'-' * widths[0]} {'-' * widths[1]} {'-' * widths[2]} {'-' * widths[3]}")
    for row in rows:
        print(
            f"{row[0]:<{widths[0]}} "
            f"{row[1]:>{widths[1]}} "
            f"{row[2]:>{widths[2]}} "
            f"{row[3]:>{widths[3]}}"
        )

def parse_args(argv=None):
    '''
    Parse CLI arguments for evaluation.
    Args:
        argv (list[str] | None): Optional command-line arguments.
    Returns:
        argparse.Namespace: Parsed CLI arguments.
    '''
    parser = argparse.ArgumentParser(description="Evaluate MIA audit metrics and plot average ROC curves.")
    default_config_path = str(utils.resolve_path(Path("mia") / "configs" / "config_evaluation.yaml", utils.get_root()))
    parser.add_argument("--config", default=default_config_path, help="Path to evaluation config yaml file.")
    parser.add_argument(
        "--metrics-folders",
        nargs="+",
        default=None,
        help="Metrics folder paths to evaluate. Overrides config.",
    )
    parser.add_argument(
        "--low-exponent",
        type=int,
        default=None,
        help="Low exponent for logspace FPR. Overrides config.",
    )
    return parser.parse_args(argv)

def run_evaluation(config, metrics_folders_override=None, low_exponent_override=None):
    '''
    Run evaluation for selected metrics folders.
    Args:
        config (Config): Evaluation configuration.
        metrics_folders_override (list[str] | None): Optional metrics folder overrides.
        low_exponent_override (int | None): Optional low exponent override.
    Returns:
        list[dict]: Computed metrics summaries.
    '''
    output_dir, metrics_dirs = path_utils.resolve_evaluation_paths(config, metrics_folders_override=metrics_folders_override)
    low_exponent = low_exponent_override if low_exponent_override is not None else getattr(config, "low_exponent", -4)
    fpr_space, summaries = collect_metrics_folder_summaries(metrics_dirs, low_exponent=low_exponent)
    plot_average_roc_curves(output_dir, fpr_space, summaries)
    print_metrics_table(summaries)
    return summaries

def main(argv=None):
    '''
    Entry point for the evaluation CLI.
    Args:
        argv (list[str] | None): Optional command-line arguments.
    Returns:
        None
    '''
    args = parse_args(argv)
    config_path = utils.resolve_path(args.config, utils.get_root())
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)
    config = utils.Config(config_dict)
    run_evaluation(
        config=config,
        metrics_folders_override=args.metrics_folders,
        low_exponent_override=args.low_exponent,
    )

if __name__ == "__main__":
    main()
