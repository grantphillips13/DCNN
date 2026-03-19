"""
compare_models.py
=================
Compares DCNN glioma-parameter-prediction models trained on:
  - Smaller dataset  : 50 samples  (35 train / 7 val / 8 test)
  - Larger  dataset  : 250 samples (175 train / 37 val / 38 test)

Models compared
---------------
Smaller set (modelsTrainedOnSmallerSet/):
  1. og_dcnn_param_predict              – Base CNN, physics inputs (P,N,S,seg), 80 epochs
  2. og_dcnn_param_predict_cache        – CNN, physics inputs, cached loading, 80 epochs
  3. og_dcnn_param_predict_cache_log    – CNN, physics+log(Dw), 300 epochs, best-val ckpt
  4. og_dcnn_param_predict_cache_log_segm – CNN, segm-only, log(Dw), 300 epochs, best-val ckpt

Larger set (root/):
  5. dcnn_param_predict_cache_log       – CNN, physics+log(Dw), 300 epochs, best-val ckpt
  6. dcnn_param_predict_cache_log_segm  – CNN, segm-only,       300 epochs, best-val ckpt
  7. dcnn_ablation_pns_only             – CNN, P+N+S (no seg),  300 epochs, best-val ckpt

Usage
-----
    python compare_models.py

Outputs (saved in comparison_results/):
  - training_curves.png   : val MSE curves for each model, small vs large side-by-side
  - test_metrics.png      : bar charts of Dw and rho MAE / RMSE across all models
  - summary_table.txt     : printable metrics summary
"""

import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless – no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent
SMALL_DIR = BASE / "modelsTrainedOnSmallerSet"
OUT_DIR   = BASE / "comparison_results"
OUT_DIR.mkdir(exist_ok=True)

# ─── Notebook registry ────────────────────────────────────────────────────────
# Each entry: (short_label, notebook_path, dataset_size, group_key)
# group_key links "same model, different dataset size" for direct comparison
NOTEBOOKS = [
    ("Base CNN\n(50 samp)",        SMALL_DIR / "og_dcnn_param_predict.ipynb",              50,  "base_nocache"),
    ("Cached CNN\n(50 samp)",      SMALL_DIR / "og_dcnn_param_predict_cache.ipynb",         50,  "cached"),
    ("Physics+log(Dw)\n(50 samp)", SMALL_DIR / "og_dcnn_param_predict_cache_log.ipynb",     50,  "physics_log"),
    ("Segm-only\n(50 samp)",       SMALL_DIR / "og_dcnn_param_predict_cache_log_segm.ipynb",50,  "segm_only"),
    ("Physics+log(Dw)\n(250 samp)",BASE       / "dcnn_param_predict_cache_log.ipynb",        250, "physics_log"),
    ("Segm-only\n(250 samp)",      BASE       / "dcnn_param_predict_cache_log_segm.ipynb",   250, "segm_only"),
    ("PNS-only\n(250 samp)",       BASE       / "dcnn_ablation_pns_only.ipynb",              250, "pns_only"),
]

# Colour palette: consistent per group, lighter for 50-sample runs
GROUP_COLORS = {
    "base_nocache": "#1f77b4",
    "cached":       "#ff7f0e",
    "physics_log":  "#2ca02c",
    "segm_only":    "#d62728",
    "pns_only":     "#9467bd",
}

# ─── Regex patterns ───────────────────────────────────────────────────────────
EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)\s*\|"
    r"\s*train MSE\s*([\d.]+)\s*MAE\s*([\d.]+)"
    r"\s*\|\s*val MSE\s*([\d.]+)\s*MAE\s*([\d.]+)"
)
TEST_RE   = re.compile(r"(\w+):\s*MAE=([\d.eE+\-]+)\s*RMSE=([\d.eE+\-]+)")
# Matches the "best val X.XXXX @ XXX" suffix on each epoch line (always updated)
BEST_RE   = re.compile(r"best val\s*([\d.]+)\s*@\s*(\d+)")
# Matches the final checkpoint confirmation line
LOADED_RE = re.compile(r"Loaded best model from epoch\s+(\d+)\s+with val MSE\s+([\d.]+)")

# ─── Parser ───────────────────────────────────────────────────────────────────

def extract_cell_text(cell: dict) -> str:
    """Concatenate all text from a notebook cell's outputs."""
    parts = []
    for output in cell.get("outputs", []):
        otype = output.get("output_type", "")
        if otype in ("stream", "display_data", "execute_result"):
            text = output.get("text", output.get("data", {}).get("text/plain", []))
            if isinstance(text, list):
                parts.append("".join(text))
            elif isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def parse_notebook(nb_path: Path) -> dict:
    """
    Returns:
        {
            "epochs":     list of epoch ints,
            "train_mse":  list of floats,
            "train_mae":  list of floats,
            "val_mse":    list of floats,
            "val_mae":    list of floats,
            "best_val_mse": float | None,
            "best_epoch":   int   | None,
            "test": {param: {"mae": float, "rmse": float}, ...}
        }
    """
    result = dict(
        epochs=[], train_mse=[], train_mae=[], val_mse=[], val_mae=[],
        best_val_mse=None, best_epoch=None, test={}
    )

    if not nb_path.exists():
        print(f"  [SKIP] not found: {nb_path}")
        return result

    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    all_text = ""
    for cell in nb.get("cells", []):
        cell_text = extract_cell_text(cell)
        all_text += cell_text + "\n"

        # ---- training log lines ----
        for m in EPOCH_RE.finditer(cell_text):
            result["epochs"].append(int(m.group(1)))
            result["train_mse"].append(float(m.group(2)))
            result["train_mae"].append(float(m.group(3)))
            result["val_mse"].append(float(m.group(4)))
            result["val_mae"].append(float(m.group(5)))

        # ---- best val (from epoch lines — last occurrence = final best) ----
        for bm in BEST_RE.finditer(cell_text):
            result["best_val_mse"] = float(bm.group(1))
            result["best_epoch"]   = int(bm.group(2))

        # ---- checkpoint confirmation line overrides if present ----
        lm = LOADED_RE.search(cell_text)
        if lm:
            result["best_epoch"]   = int(lm.group(1))
            result["best_val_mse"] = float(lm.group(2))

        # ---- test metrics (e.g. "Dw: MAE=0.123 RMSE=0.456") ----
        for m in TEST_RE.finditer(cell_text):
            param = m.group(1)
            result["test"][param] = {
                "mae":  float(m.group(2)),
                "rmse": float(m.group(3)),
            }

    # Deduplicate epochs (some notebooks print a "best val" line on the same epoch line)
    if result["epochs"]:
        seen = {}
        for i, ep in enumerate(result["epochs"]):
            seen[ep] = i          # keep last occurrence (matches val_mse behaviour)
        idx = sorted(seen.values())
        for key in ("epochs", "train_mse", "train_mae", "val_mse", "val_mae"):
            result[key] = [result[key][i] for i in idx]

    return result


# ─── Load all notebooks ───────────────────────────────────────────────────────
print("Parsing notebooks …")
parsed = {}
for label, path, n_samples, group in NOTEBOOKS:
    print(f"  {label.replace(chr(10), ' '):40s} ({n_samples:3d} samples)  {path.name}")
    data = parse_notebook(path)
    data["label"]     = label
    data["n_samples"] = n_samples
    data["group"]     = group
    data["path"]      = path
    parsed[label]     = data

    n_ep  = len(data["epochs"])
    n_test = len(data["test"])
    print(f"    → {n_ep} epoch rows,  {n_test} test params found")
    if data["best_val_mse"] is not None:
        print(f"    → best val MSE {data['best_val_mse']:.4f} @ epoch {data['best_epoch']}")
    for p, v in data["test"].items():
        print(f"    → test {p}: MAE={v['mae']:.6f}  RMSE={v['rmse']:.6f}")

# ─── Figure 1 : Training curves ───────────────────────────────────────────────
print("\nGenerating training_curves.png …")

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(
    "Training & Validation Loss Curves\n"
    "Solid = 250-sample | Dashed = 50-sample",
    fontsize=14, fontweight="bold"
)

metric_pairs = [
    (axes[0, 0], "train_mse", "Train MSE (normalised space)"),
    (axes[0, 1], "val_mse",   "Val MSE (normalised space)"),
    (axes[1, 0], "train_mae", "Train MAE (normalised space)"),
    (axes[1, 1], "val_mae",   "Val MAE (normalised space)"),
]

legend_handles = []
seen_labels = set()

for ax, metric_key, ylabel in metric_pairs:
    for label, data in parsed.items():
        if not data["epochs"]:
            continue
        color     = GROUP_COLORS.get(data["group"], "grey")
        linestyle = "-" if data["n_samples"] == 250 else "--"
        alpha     = 1.0 if data["n_samples"] == 250 else 0.65
        lw        = 1.8 if data["n_samples"] == 250 else 1.2

        ax.plot(
            data["epochs"],
            data[metric_key],
            color=color, linestyle=linestyle, linewidth=lw, alpha=alpha,
            label=label.replace("\n", " ")
        )

        # mark best-val epoch on val_mse subplot
        if metric_key == "val_mse" and data["best_epoch"] is not None:
            ep_idx = data["best_epoch"] - 1
            if 0 <= ep_idx < len(data["epochs"]):
                ax.axvline(
                    data["best_epoch"], color=color,
                    linestyle=":", linewidth=0.8, alpha=0.5
                )

        if label not in seen_labels:
            seen_labels.add(label)
            legend_handles.append(
                mpatches.Patch(
                    facecolor=color,
                    linestyle=linestyle,
                    label=label.replace("\n", " "),
                    alpha=alpha
                )
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=4,
    fontsize=8,
    bbox_to_anchor=(0.5, -0.04),
    framealpha=0.9,
)
plt.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig(OUT_DIR / "training_curves.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved → comparison_results/training_curves.png")

# ─── Figure 2 : Training-curve comparison for matched pairs ───────────────────
print("Generating paired_val_mse.png …")

# Group models that share the same architecture but differ in dataset size
groups_with_both = {}
for label, data in parsed.items():
    g = data["group"]
    groups_with_both.setdefault(g, {})[data["n_samples"]] = data

pair_groups = {g: v for g, v in groups_with_both.items() if 50 in v and 250 in v}

if pair_groups:
    fig, axes = plt.subplots(1, len(pair_groups), figsize=(6 * len(pair_groups), 5), squeeze=False)
    fig.suptitle("Val MSE: 50-sample vs 250-sample (same architecture)", fontsize=13, fontweight="bold")

    for col, (group_key, variants) in enumerate(pair_groups.items()):
        ax = axes[0, col]
        color = GROUP_COLORS.get(group_key, "grey")

        for n_samp, data in sorted(variants.items()):
            ls  = "--" if n_samp == 50 else "-"
            lbl = f"{n_samp} samples"
            ax.plot(data["epochs"], data["val_mse"], color=color, linestyle=ls, linewidth=1.8, label=lbl)
            if data["best_epoch"] is not None:
                ax.axvline(data["best_epoch"], color=color, linestyle=":", linewidth=0.9, alpha=0.6)

        title = group_key.replace("_", " ").title()
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val MSE")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "paired_val_mse.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved → comparison_results/paired_val_mse.png")

# ─── Figure 3 : Test metrics bar chart ────────────────────────────────────────
print("Generating test_metrics.png …")

models_with_test = [(label, data) for label, data in parsed.items() if data["test"]]

if models_with_test:
    params_found = sorted({p for _, d in models_with_test for p in d["test"]})
    metrics      = ["mae", "rmse"]
    metric_titles = {"mae": "MAE", "rmse": "RMSE"}

    n_params  = len(params_found)
    n_metrics = len(metrics)
    n_models  = len(models_with_test)

    fig, axes = plt.subplots(n_params, n_metrics, figsize=(7 * n_metrics, 5 * n_params), squeeze=False)
    fig.suptitle("Test Set Metrics (original parameter scale)", fontsize=14, fontweight="bold")

    x      = np.arange(n_models)
    width  = 0.6

    for row, param in enumerate(params_found):
        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            vals   = [d["test"].get(param, {}).get(metric, np.nan) for _, d in models_with_test]
            colors = [
                GROUP_COLORS.get(d["group"], "grey")
                for _, d in models_with_test
            ]
            bars = ax.bar(x, vals, width=width, color=colors, edgecolor="white", linewidth=0.8)

            # value labels on bars
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.003,
                        f"{v:.4f}",
                        ha="center", va="bottom", fontsize=7.5, rotation=0
                    )

            # hatching to distinguish dataset sizes
            for bar, (_, d) in zip(bars, models_with_test):
                if d["n_samples"] == 50:
                    bar.set_hatch("//")

            ax.set_title(f"{param} — {metric_titles[metric]}")
            ax.set_xticks(x)
            ax.set_xticklabels(
                [lbl.replace("\n", "\n") for lbl, _ in models_with_test],
                fontsize=8, rotation=20, ha="right"
            )
            ax.set_ylabel(metric_titles[metric])
            ax.grid(axis="y", alpha=0.3)
            ax.set_ylim(bottom=0)

    # Legend for hatching
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="grey", hatch="//", label="50-sample models"),
        Patch(facecolor="grey",              label="250-sample models"),
    ]
    fig.legend(handles=legend_elems, loc="upper right", fontsize=9, framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    fig.savefig(OUT_DIR / "test_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved → comparison_results/test_metrics.png")

# ─── Figure 4 : Final-epoch val MSE comparison ────────────────────────────────
print("Generating final_val_mse.png …")

models_with_curve = [(lbl, d) for lbl, d in parsed.items() if d["val_mse"]]

if models_with_curve:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Best Validation MSE Achieved (normalised space)\n(dotted line = 50-sample, solid = 250-sample)", fontsize=12)

    x      = np.arange(len(models_with_curve))
    width  = 0.6
    colors = [GROUP_COLORS.get(d["group"], "grey") for _, d in models_with_curve]

    # Use the logged best_val_mse if available, else min(val_mse)
    best_vals = []
    for lbl, d in models_with_curve:
        if d["best_val_mse"] is not None:
            best_vals.append(d["best_val_mse"])
        else:
            best_vals.append(min(d["val_mse"]))

    bars = ax.bar(x, best_vals, width=width, color=colors, edgecolor="white")
    for bar, (_, d) in zip(bars, models_with_curve):
        if d["n_samples"] == 50:
            bar.set_hatch("//")

    for bar, v in zip(bars, best_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{v:.4f}",
            ha="center", va="bottom", fontsize=8
        )

    ax.set_xticks(x)
    ax.set_xticklabels([lbl.replace("\n", "\n") for lbl, _ in models_with_curve], fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("Best Val MSE")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="grey", hatch="//", label="50-sample models"),
        Patch(facecolor="grey",              label="250-sample models"),
    ] + [
        Patch(facecolor=c, label=k.replace("_", " ").title())
        for k, c in GROUP_COLORS.items()
    ]
    ax.legend(handles=legend_elems, fontsize=8, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "final_val_mse.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved → comparison_results/final_val_mse.png")

# ─── Text summary ─────────────────────────────────────────────────────────────
print("Writing summary_table.txt …")

lines = []
lines.append("=" * 90)
lines.append("DCNN GLIOMA PARAMETER PREDICTION — MODEL COMPARISON SUMMARY")
lines.append("=" * 90)
lines.append("")
lines.append(f"{'Model':<42} {'Data':>6} {'Epochs':>7} {'BestEp':>7} {'BestValMSE':>11} {'Dw MAE':>9} {'rho MAE':>9}")
lines.append("-" * 90)

for label, data in parsed.items():
    short = label.replace("\n", " ")
    n     = data["n_samples"]
    n_ep  = data["epochs"][-1] if data["epochs"] else 0
    bep   = data["best_epoch"] or "—"
    bvmse = f"{data['best_val_mse']:.4f}" if data["best_val_mse"] is not None else (
            f"{min(data['val_mse']):.4f}" if data["val_mse"] else "—"
    )
    dw_mae  = f"{data['test'].get('Dw',  {}).get('mae',  float('nan')):.4f}"
    rho_mae = f"{data['test'].get('rho', {}).get('mae',  float('nan')):.4f}"
    lines.append(f"{short:<42} {n:>6} {n_ep:>7} {str(bep):>7} {bvmse:>11} {dw_mae:>9} {rho_mae:>9}")

lines.append("")
lines.append("=" * 90)
lines.append("FULL TEST METRICS")
lines.append("=" * 90)
lines.append("")
lines.append(f"{'Model':<42} {'Data':>6} {'Param':>6} {'MAE':>12} {'RMSE':>12}")
lines.append("-" * 90)

for label, data in parsed.items():
    if not data["test"]:
        continue
    short = label.replace("\n", " ")
    n     = data["n_samples"]
    for param, metrics in sorted(data["test"].items()):
        lines.append(
            f"{short:<42} {n:>6} {param:>6} {metrics['mae']:>12.6f} {metrics['rmse']:>12.6f}"
        )
    lines.append("")

lines.append("=" * 90)
lines.append("NOTES")
lines.append("=" * 90)
lines.append("  • Metrics in training/val curves are in normalised parameter space.")
lines.append("  • Test MAE/RMSE are in original physical units: Dw [mm²/day], rho [1/day].")
lines.append("  • log(Dw) transform applied in all 'cache_log' and 'pns_only' variants.")
lines.append("  • Dw test values are exponentiated back to original scale after prediction.")
lines.append("  • Best-val epoch marked with dotted vertical lines on val MSE plots.")
lines.append("  • Hatched bars (///) = 50-sample models, solid = 250-sample models.")

summary_text = "\n".join(lines)
(OUT_DIR / "summary_table.txt").write_text(summary_text, encoding="utf-8")
print("  Saved → comparison_results/summary_table.txt")
print()
print(summary_text)
print()
print("Done. All outputs in:", OUT_DIR)
