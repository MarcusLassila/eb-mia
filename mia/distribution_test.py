import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats
import torch

from .utils import load_loss_signals, standardize_signals

def load_selected_signals(signal_paths, max_points, seed, normalization):
    '''Load sampled data points across model files; return values, masks, and indices.'''
    if max_points < 1:
        raise ValueError("max_points must be positive.")
    if normalization not in ("none", "log", "log_standardized"):
        raise ValueError(f"Unsupported normalization: {normalization}")
    rng = np.random.default_rng(seed)
    selected_signals = []
    selected_masks = []
    point_indices = None
    expected_shape = None
    for signal_path in signal_paths:
        signals, train_mask = load_loss_signals(signal_path)
        if expected_shape is None:
            expected_shape = tuple(signals.shape)
            point_count = min(max_points, expected_shape[0])
            point_indices = np.sort(rng.choice(expected_shape[0], size=point_count, replace=False))
        elif tuple(signals.shape) != expected_shape:
            raise ValueError(f"Signal shape mismatch in {signal_path}: expected {expected_shape}, got {tuple(signals.shape)}.")
        if signals.shape[1] < 3:
            raise ValueError("Normality tests require at least three query outputs per data point.")
        if not torch.isfinite(signals).all():
            raise ValueError(f"Non-finite signal values in {signal_path}.")
        if normalization == "log":
            signals = torch.log(signals)
        elif normalization == "log_standardized":
            signals = standardize_signals(signals, "log")
        if not torch.isfinite(signals).all():
            raise ValueError(f"Normalization produced non-finite signal values in {signal_path}.")
        point_signals = signals[point_indices]
        point_mask = train_mask[point_indices]
        selected_signals.append(point_signals.numpy())
        selected_masks.append(point_mask.numpy())
    return np.stack(selected_signals), np.stack(selected_masks), point_indices

def normality_statistics(values):
    '''Test finite sample values for Gaussian shape; return fit and test statistics.'''
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    result = {
        "n": len(values),
        "mean": mean,
        "std": std,
        "shapiro_stat": None,
        "shapiro_p": None,
        "anderson_stat": None,
        "anderson_5pct_critical": None,
        "anderson_reject_5pct": None,
    }
    if len(values) < 3 or std == 0 or not np.isfinite(std):
        return result
    if len(values) <= 5000:
        shapiro_result = stats.shapiro(values)
        result["shapiro_stat"] = float(shapiro_result.statistic)
        result["shapiro_p"] = float(shapiro_result.pvalue)
    anderson_result = stats.anderson(values, dist="norm")
    level_index = np.flatnonzero(np.isclose(anderson_result.significance_level, 5.0))[0]
    critical_value = float(anderson_result.critical_values[level_index])
    result["anderson_stat"] = float(anderson_result.statistic)
    result["anderson_5pct_critical"] = critical_value
    result["anderson_reject_5pct"] = bool(anderson_result.statistic > critical_value)
    return result

def write_tikz_plot(path, title, x_label, y_label, curves, axis_options=""):
    '''Write plot coordinates and styles to a standalone PGFPlots file; return its path.'''
    axis_settings = "grid=major"
    if axis_options:
        axis_settings = f"{axis_settings},{axis_options}"
    lines = [
        r"\documentclass[tikz]{standalone}",
        r"\usepackage{pgfplots}",
        r"\pgfplotsset{compat=1.18}",
        r"\begin{document}",
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        "width=12cm,height=8cm,",
        f"title={{{title}}},xlabel={{{x_label}}},ylabel={{{y_label}}},",
        f"{axis_settings}]",
    ]
    for style, x_values, y_values, label in curves:
        coordinates = " ".join(f"({float(x):.8e},{float(y):.8e})" for x, y in zip(x_values, y_values))
        lines.append(rf"\addplot+[{style}] coordinates {{{coordinates}}};")
        if label:
            lines.append(rf"\addlegendentry{{{label}}}")
    lines.extend([r"\end{axis}", r"\end{tikzpicture}", r"\end{document}"])
    path.write_text("\n".join(lines) + "\n")
    return path

def save_diagnostic_plots(values, result, plot_prefix, title, bins):
    '''Save Gaussian Q-Q and histogram plots as PNG and TikZ; return their paths.'''
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")
    mean = result["mean"]
    std = result["std"]
    if result["anderson_stat"] is None:
        return []
    theoretical, observed = stats.probplot(values, dist="norm", sparams=(mean, std), fit=False)
    lower = min(float(np.min(theoretical)), float(np.min(observed)))
    upper = max(float(np.max(theoretical)), float(np.max(observed)))
    qq_path = plot_prefix.with_name(f"{plot_prefix.name}_qq.png")
    figure, axis = plt.subplots(figsize=(5, 5))
    axis.scatter(theoretical, observed, s=12, alpha=0.7)
    axis.plot([lower, upper], [lower, upper], color="black", linestyle="--")
    axis.set(xlabel="Fitted Gaussian quantile", ylabel="Observed quantile", title=title)
    figure.tight_layout()
    figure.savefig(qq_path, dpi=150)
    plt.close(figure)
    qq_curves = [
        ("only marks,mark=*,mark size=0.7pt,blue", theoretical, observed, None),
        ("mark=none,black,dashed", [lower, upper], [lower, upper], None),
    ]
    qq_tikz_path = write_tikz_plot(qq_path.with_suffix(".tex"), title,
                                    "Fitted Gaussian quantile", "Observed quantile", qq_curves)

    histogram_path = plot_prefix.with_name(f"{plot_prefix.name}_hist.png")
    x_values = np.linspace(float(np.min(values)), float(np.max(values)), 300)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.hist(values, bins=bins, density=True, alpha=0.65, label="Observed")
    axis.plot(x_values, stats.norm.pdf(x_values, loc=mean, scale=std), label=f"Gaussian (μ={mean:.3g}, σ={std:.3g})")
    axis.set(xlabel="Signal value", ylabel="Density", title=title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(histogram_path, dpi=150)
    plt.close(figure)
    heights, edges = np.histogram(values, bins=bins, density=True)
    histogram_curves = [
        ("ybar interval,fill=blue!35,draw=blue!60", edges, np.r_[heights, 0.0], "Observed"),
        ("mark=none,orange,thick", x_values, stats.norm.pdf(x_values, loc=mean, scale=std),
         f"Gaussian (mu={mean:.3g}, sigma={std:.3g})"),
    ]
    histogram_tikz_path = write_tikz_plot(histogram_path.with_suffix(".tex"), title,
                                           "Signal value", "Density", histogram_curves)
    return [qq_path, qq_tikz_path, histogram_path, histogram_tikz_path]

def run_distribution_analysis(input_dir, output_dir="temp_results", normalization="none", max_points=200, plots_per_group=1, seed=0, bins=40, max_plot_points=1):
    '''Analyze query and across-model distributions; write plots and statistics.'''
    if plots_per_group < 0 or max_plot_points < 0 or bins < 1:
        raise ValueError("Plot counts must be nonnegative and bins must be positive.")
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise ValueError(f"Signal folder does not exist: {input_dir}")
    signal_paths = sorted(input_dir.rglob("*.pkl"))
    if not signal_paths:
        raise ValueError(f"No signal pickle files found in {input_dir}.")
    signals, train_masks, point_indices = load_selected_signals(signal_paths, max_points, seed, normalization)
    analysis_dir = Path(output_dir) / "distributional_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    plot_point_count = min(max_plot_points, len(point_indices))
    plot_point_positions = set(rng.choice(len(point_indices), size=plot_point_count, replace=False))
    records = []
    plot_paths = []
    for point_position, data_index in enumerate(point_indices):
        for model_index, signal_path in enumerate(signal_paths):
            group = "member" if train_masks[model_index, point_position] else "nonmember"
            values = signals[model_index, point_position]
            result = normality_statistics(values)
            records.append({"scope": "query_outputs", "data_index": int(data_index), "group": group,
                            "model_file": str(signal_path), **result})
        for group, membership in (("member", True), ("nonmember", False)):
            model_indices = np.flatnonzero(train_masks[:, point_position] == membership)
            if not len(model_indices):
                continue
            if point_position in plot_point_positions:
                plot_count = min(plots_per_group, len(model_indices))
                plot_indices = rng.choice(model_indices, size=plot_count, replace=False)
                for model_index in plot_indices:
                    values = signals[model_index, point_position]
                    result = normality_statistics(values)
                    prefix = analysis_dir / f"query_point{data_index}_model{model_index}_{group}"
                    title = f"Query outputs: point {data_index}, model {model_index}, {group}"
                    plot_paths.extend(save_diagnostic_plots(values, result, prefix, title, bins))
            means = signals[model_indices, point_position].mean(axis=1)
            result = normality_statistics(means)
            records.append({"scope": "model_means", "data_index": int(data_index), "group": group,
                            "model_file": "", **result})
            if point_position in plot_point_positions:
                prefix = analysis_dir / f"model_means_point{data_index}_{group}"
                title = f"Query means across models: point {data_index}, {group}"
                plot_paths.extend(save_diagnostic_plots(means, result, prefix, title, bins))
    csv_path = analysis_dir / "normality_tests.csv"
    fieldnames = ("scope", "data_index", "group", "model_file", "n", "mean", "std",
                  "shapiro_stat", "shapiro_p", "anderson_stat", "anderson_5pct_critical",
                  "anderson_reject_5pct")
    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    summary = {"input_dir": str(input_dir), "normalization": normalization,
               "n_models": len(signal_paths), "n_queries": int(signals.shape[2]),
               "data_indices": point_indices.tolist(), "seed": seed,
               "n_query_tests": sum(record["scope"] == "query_outputs" for record in records),
               "n_model_mean_tests": sum(record["scope"] == "model_means" for record in records),
               "plot_paths": [str(path) for path in plot_paths]}
    summary_path = analysis_dir / "summary.json"
    with open(summary_path, "w") as file:
        json.dump(summary, file, indent=2)
    return {"summary_path": summary_path, "csv_path": csv_path, "plot_paths": plot_paths}

def parse_args(argv=None):
    '''Parse CLI options; return the configured argument namespace.'''
    parser = argparse.ArgumentParser(description="Test Gaussian assumptions for checkpoint loss signals.")
    parser.add_argument("--input-dir", required=True, help="Folder containing checkpoint signal pickles.")
    parser.add_argument("--output-dir", default="temp_results", help="Output root directory.")
    parser.add_argument(
        "--normalization",
        choices=("none", "log", "log_standardized"),
        default="none",
        help="Signal transform: none, log, or per-model standardized log.",
    )
    parser.add_argument("--max-points", type=int, default=200)
    parser.add_argument("--max-plot-points", type=int, default=1)
    parser.add_argument("--plots-per-group", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bins", type=int, default=40)
    return parser.parse_args(argv)

def main(argv=None):
    '''Run CLI distribution analysis; return paths to generated outputs.'''
    args = parse_args(argv)
    return run_distribution_analysis(args.input_dir, args.output_dir, args.normalization,
                                     args.max_points, args.plots_per_group, args.seed, args.bins,
                                     args.max_plot_points)


if __name__ == "__main__":
    main()
