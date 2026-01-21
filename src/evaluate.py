import sys
import json
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import confusion_matrix
from scipy import stats
import wandb

# Ensure non-interactive backend for CI runners
plt.switch_backend("Agg")
sns.set(style="whitegrid")

PRIMARY_METRIC = "accuracy"  # For aggregated gap calculation

###############################################################################
# ---------------------------  I/O helpers  ---------------------------------- #
###############################################################################

def save_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fp:
        json.dump(obj, fp, indent=2)

###############################################################################
# ---------------------------  Plotting utils  ------------------------------ #
###############################################################################

def plot_learning_curve(history: pd.DataFrame, run_id: str, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    if "train_acc" in history.columns:
        ax.plot(history["epoch"], history["train_acc"], label="Train")
    if "val_acc" in history.columns:
        ax.plot(history["epoch"], history["val_acc"], label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Learning Curve – {run_id}")
    ax.legend()
    fig.tight_layout()
    fname = out_dir / f"{run_id}_learning_curve.pdf"
    fig.savefig(fname)
    plt.close(fig)
    print(fname)


def plot_confusion_matrix(y_true: List[int], y_pred: List[int], run_id: str, out_dir: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix – {run_id}")
    fig.tight_layout()
    fname = out_dir / f"{run_id}_confusion_matrix.pdf"
    fig.savefig(fname)
    plt.close(fig)
    print(fname)


def plot_reliability_diagram(y_true: List[int], y_conf: List[float], y_pred: List[int], run_id: str, out_dir: Path, n_bins: int = 15):
    y_true = np.array(y_true)
    y_conf = np.array(y_conf)
    y_pred = np.array(y_pred)
    confidence_bins = np.linspace(0.0, 1.0, n_bins + 1)
    accuracies, confidences = [], []
    for i in range(n_bins):
        idx = (y_conf > confidence_bins[i]) & (y_conf <= confidence_bins[i + 1])
        if idx.any():
            accuracies.append((y_pred[idx] == y_true[idx]).mean())
            confidences.append(y_conf[idx].mean())
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(confidences, accuracies, marker="o")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability Diagram – {run_id}")
    fig.tight_layout()
    fname = out_dir / f"{run_id}_reliability_diagram.pdf"
    fig.savefig(fname)
    plt.close(fig)
    print(fname)


def plot_comparison_bar(metric_vals: Dict[str, float], out_dir: Path, metric_name: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=list(metric_vals.keys()), y=list(metric_vals.values()), ax=ax)
    ax.set_ylabel(metric_name.capitalize())
    ax.set_title(f"Run-wise {metric_name.capitalize()} Comparison")
    for idx, v in enumerate(metric_vals.values()):
        ax.text(idx, v + 0.002, f"{v:.3f}", ha="center")
    fig.tight_layout()
    fname = out_dir / f"comparison_{metric_name}_bar_chart.pdf"
    fig.savefig(fname)
    plt.close(fig)
    print(fname)


def plot_box_plot(values: Dict[str, Dict[str, float]], out_dir: Path, metric_name: str):
    data = []
    for run_id, val in values.items():
        group = "proposed" if "proposed" in run_id else "baseline"
        data.append({"group": group, metric_name: val})
    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x="group", y=metric_name, data=df, ax=ax)
    ax.set_title(f"{metric_name.capitalize()} Distribution Across Groups")
    fig.tight_layout()
    fname = out_dir / f"comparison_{metric_name}_box_plot.pdf"
    fig.savefig(fname)
    plt.close(fig)
    print(fname)

###############################################################################
# -------------------------  Aggregation logic  ----------------------------- #
###############################################################################

def aggregate_metrics(per_run: Dict[str, Dict[str, float]]) -> Dict:
    aggregated = {"primary_metric": PRIMARY_METRIC, "metrics": {}}
    for rid, metrics in per_run.items():
        for k, v in metrics.items():
            aggregated["metrics"].setdefault(k, {})[rid] = v

    proposed = {rid: d[PRIMARY_METRIC] for rid, d in per_run.items() if "proposed" in rid}
    baseline = {rid: d[PRIMARY_METRIC] for rid, d in per_run.items() if ("baseline" in rid or "comparative" in rid)}

    best_proposed_id, best_proposed_val = (max(proposed, key=proposed.get), proposed[max(proposed, key=proposed.get)]) if proposed else (None, None)
    best_baseline_id, best_baseline_val = (max(baseline, key=baseline.get), baseline[max(baseline, key=baseline.get)]) if baseline else (None, None)

    gap = None
    if best_baseline_val not in [None, 0]:
        gap = (best_proposed_val - best_baseline_val) / best_baseline_val * 100

    aggregated.update({
        "best_proposed": {"run_id": best_proposed_id, "value": best_proposed_val},
        "best_baseline": {"run_id": best_baseline_id, "value": best_baseline_val},
        "gap": gap,
    })

    # Statistical significance (if multiple runs per group)
    if len(proposed) >= 2 and len(baseline) >= 2:
        stat, p_val = stats.ttest_ind(list(proposed.values()), list(baseline.values()), equal_var=False)
        aggregated["statistical_test"] = {"t_stat": stat, "p_value": p_val}
    return aggregated

###############################################################################
# -----------------------------  CLI parser  --------------------------------- #
###############################################################################

def parse_cli() -> (Path, List[str]):
    """Parse both classic '--arg value' style and 'arg=value' style tokens."""
    # First, attempt to capture '--results_dir' / '--run_ids' style using argparse.
    import argparse
    parser = argparse.ArgumentParser(description="Comprehensive evaluation of multiple runs.")
    parser.add_argument("--results_dir", type=str, required=False)
    parser.add_argument("--run_ids", type=str, required=False, help="JSON list, e.g. '[\"run1\", \"run2\"]'")
    known, unknown = parser.parse_known_args()

    cli_kv: Dict[str, str] = {}
    for token in unknown:
        if "=" in token:
            k, v = token.split("=", 1)
            cli_kv[k.lstrip("--")] = v

    results_dir_str = known.results_dir or cli_kv.get("results_dir")
    run_ids_str = known.run_ids or cli_kv.get("run_ids")

    if results_dir_str is None or run_ids_str is None:
        parser.error("Both 'results_dir' and 'run_ids' must be provided. Acceptable formats: \n"
                     "  • --results_dir /path --run_ids '[...]' \n"
                     "  • results_dir=/path run_ids='[...]'")

    try:
        run_ids = json.loads(run_ids_str)
        assert isinstance(run_ids, list) and all(isinstance(r, str) for r in run_ids)
    except Exception as e:
        raise ValueError("'run_ids' must be a JSON list of strings.") from e

    return Path(results_dir_str).expanduser().resolve(), run_ids

###############################################################################
# -------------------------------- Main ------------------------------------- #
###############################################################################

def main():
    # ---------------------------------------------------------------------
    # Robust CLI parsing (handles key=value syntax mandated by the paper)
    # ---------------------------------------------------------------------
    results_dir, run_ids = parse_cli()

    # ---------------------------------------------------------------------
    # Load global config to obtain WandB credentials/project info
    # ---------------------------------------------------------------------
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    with open(cfg_path, "r") as fp:
        cfg_global = yaml.safe_load(fp)

    ENTITY = cfg_global["wandb"]["entity"]
    PROJECT = cfg_global["wandb"]["project"]

    api = wandb.Api()

    per_run_metrics: Dict[str, Dict[str, float]] = {}

    for rid in run_ids:
        # -----------------------  Fetch run  ---------------------------- #
        run = api.run(f"{ENTITY}/{PROJECT}/{rid}")
        history = run.history(pandas=True)
        summary = dict(run.summary)  # explicit copy
        config = dict(run.config)

        # --------------------  Per-run outputs ------------------------- #
        out_dir = results_dir / rid
        out_dir.mkdir(parents=True, exist_ok=True)

        save_json(summary, out_dir / "metrics.json")
        save_json(config, out_dir / "config.json")
        print(out_dir / "metrics.json")
        print(out_dir / "config.json")

        # Learning curve
        if not history.empty and {"epoch", "train_acc"}.issubset(history.columns):
            plot_learning_curve(history, rid, out_dir)

        # Confusion matrix & reliability diagram if predictions available
        if all(key in summary for key in ["y_true", "y_pred", "y_conf"]):
            y_true, y_pred, y_conf = summary["y_true"], summary["y_pred"], summary["y_conf"]
            plot_confusion_matrix(y_true, y_pred, rid, out_dir)
            plot_reliability_diagram(y_true, y_conf, y_pred, rid, out_dir)

        # Collect primary/secondary metrics
        per_run_metrics[rid] = {
            "accuracy": summary.get("test_acc", summary.get("best_val_accuracy", None)),
            "ece": summary.get("test_ece", None),
        }

    # ----------------------------------------------------------------- #
    #                       Aggregated analysis                         #
    # ----------------------------------------------------------------- #
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    aggregated = aggregate_metrics(per_run_metrics)
    save_json(aggregated, comparison_dir / "aggregated_metrics.json")
    print(comparison_dir / "aggregated_metrics.json")

    # Generate comparison plots
    if PRIMARY_METRIC in aggregated["metrics"]:
        plot_comparison_bar(aggregated["metrics"][PRIMARY_METRIC], comparison_dir, PRIMARY_METRIC)
        plot_box_plot(aggregated["metrics"][PRIMARY_METRIC], comparison_dir, PRIMARY_METRIC)


if __name__ == "__main__":
    main()
