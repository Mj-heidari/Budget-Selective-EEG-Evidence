"""Figures and summary tables for a completed R3-Budget run.

Reads only the artefacts an evaluation writes, so a report can be regenerated
without refitting. Every view is built to stay readable when a contrast is
null: comparisons are drawn as effect sizes with intervals through zero rather
than as bars whose heights invite over-reading.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = {"ink": "#18324a", "accent": "#ec8d26", "muted": "#6b7a8c", "pale": "#c9d4de"}

LABELS = {
    "bands": "Conventional bands", "static": "Static fusion", "mi": "Association only",
    "reliability": "+ consistency", "proposed": "R3 selection",
    "demographics": "Age and sex only", "proposed_svm": "R3 + SVM",
    "proposed_logistic": "R3 + logistic", "proposed_ros": "+ duplication",
    "proposed_jitter": "+ Gaussian jitter", "proposed_gan": "+ unguided GAN",
    "proposed_gated": "+ support-filtered GAN", "proposed_guided": "+ reliability-guided GAN",
}


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": PALETTE["ink"], "text.color": PALETTE["ink"],
        "xtick.color": PALETTE["muted"], "ytick.color": PALETTE["muted"],
        "axes.edgecolor": PALETTE["muted"], "figure.dpi": 200,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
    })


def load(folder):
    root = Path(folder)
    out = {"pooled": pd.DataFrame(json.loads((root / "pooled_metrics.json").read_text())),
           "release": pd.read_csv(root / "release_metrics.csv"),
           "predictions": pd.read_csv(root / "oof_predictions.csv"),
           "comparisons": json.loads((root / "paired_comparisons.json").read_text())}
    for name, filename in [("frequency", "selection_frequency.csv"),
                           ("budget", "budget_predictions.csv")]:
        path = root / filename
        if path.exists():
            out[name] = pd.read_csv(path)
    path = root / "budget_metrics.json"
    if path.exists():
        out["budget_metrics"] = pd.DataFrame(json.loads(path.read_text()))
    return out


def plot_model_accuracy(pooled, duration, path, title):
    """Pooled held-out balanced accuracy with conditional intervals."""
    style()
    rows = pooled[pooled.duration == duration].copy()
    rows["lo"] = [c[0] for c in rows.ba_conditional_ci95]
    rows["hi"] = [c[1] for c in rows.ba_conditional_ci95]
    rows = rows.sort_values("balanced_accuracy")
    names = [LABELS.get(m, m) for m in rows.model]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(3.6, max(2.0, len(rows) * .3)))
    ax.hlines(y, rows.lo, rows.hi, color=PALETTE["pale"], lw=3, zorder=1)
    ax.scatter(rows.balanced_accuracy, y, s=26, color=PALETTE["ink"], zorder=3)
    ax.axvline(.5, ls="--", lw=1, color=PALETTE["muted"])
    ax.set(yticks=y, yticklabels=names, xlabel="Pooled balanced accuracy",
           xlim=(min(.42, rows.lo.min() - .02), max(.62, rows.hi.max() + .02)))
    ax.set_title(title, fontsize=9, pad=8)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}")
    plt.close(fig)
    return rows[["model", "balanced_accuracy", "lo", "hi"]]


def plot_paired_effects(comparisons, path, title):
    """Paired differences with family-cluster intervals, centred on zero."""
    style()
    rows = [{"label": f"{LABELS.get(c['model'], c['model'])}\nvs {LABELS.get(c['baseline'], c['baseline'])}",
             "delta": c["delta_balanced_accuracy"],
             "lo": c["conditional_family_ci95"][0], "hi": c["conditional_family_ci95"][1]}
            for c in comparisons]
    frame = pd.DataFrame(rows)
    y = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(3.6, max(1.8, len(frame) * .44)))
    ax.hlines(y, frame.lo, frame.hi, color=PALETTE["pale"], lw=3, zorder=1)
    ax.scatter(frame.delta, y, s=26, color=PALETTE["accent"], zorder=3)
    ax.axvline(0, ls="--", lw=1, color=PALETTE["muted"])
    ax.set(yticks=y, yticklabels=frame.label, xlabel="Difference in balanced accuracy")
    ax.set_title(title, fontsize=9, pad=8)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}")
    plt.close(fig)
    return frame


def plot_budget_tradeoff(budget_metrics, path, title):
    """Accepted-case accuracy against coverage, annotated with mean budget."""
    style()
    rows = budget_metrics[budget_metrics.release == "pooled"].copy()
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    markers = {"agreement": "o", "margin": "s"}
    for (model, policy), part in rows.groupby(["model", "policy"]):
        ax.scatter(part.coverage, part.accepted_balanced_accuracy,
                   s=44, marker=markers.get(policy, "^"),
                   color=PALETTE["accent"] if "guided" in model else PALETTE["ink"],
                   label=f"{LABELS.get(model, model)} ({policy})")
        for r in part.itertuples():
            if np.isfinite(r.coverage) and np.isfinite(r.accepted_balanced_accuracy):
                ax.annotate(f"{r.mean_clean_seconds_per_condition:.0f}s",
                            (r.coverage, r.accepted_balanced_accuracy),
                            fontsize=6, xytext=(4, 3), textcoords="offset points",
                            color=PALETTE["muted"])
    ax.axhline(.5, ls="--", lw=1, color=PALETTE["muted"])
    ax.set(xlabel="Coverage", ylabel="Accepted-case balanced accuracy",
           xlim=(0, 1.02), ylim=(.35, 1.0))
    ax.legend(fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.28), ncol=2)
    ax.set_title(title, fontsize=9, pad=22)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}")
    plt.close(fig)
    return rows


def plot_selection_frequency(frequency, model, path, title, top=18):
    """How often each feature is selected across the four outer folds."""
    style()
    rows = frequency[frequency.model == model].copy()
    if rows.empty:
        return None
    rows = rows.sort_values(["outer_folds_selected", "feature"],
                            ascending=[False, True]).head(top)
    total = int(rows.total_outer_folds.iloc[0])
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(3.8, max(2.2, len(rows) * .25)))
    ax.barh(y, rows.outer_folds_selected, color=PALETTE["ink"], height=.68)
    ax.set(yticks=y, yticklabels=[f.replace("f__", "").replace("_", " ") for f in rows.feature],
           xlabel=f"Outer folds selected (of {total})", xlim=(0, total))
    ax.invert_yaxis()
    ax.set_title(title, fontsize=9, pad=8)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}")
    plt.close(fig)
    return rows


def summarise(folder, out_dir, duration=40):
    """Build every figure a completed run supports."""
    data = load(folder)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    written["accuracy"] = plot_model_accuracy(
        data["pooled"], duration, out / "model_accuracy",
        f"Untouched-release accuracy at {duration}s")
    written["effects"] = plot_paired_effects(
        data["comparisons"], out / "paired_effects", "Paired differences")
    if "budget_metrics" in data:
        written["budget"] = plot_budget_tradeoff(
            data["budget_metrics"], out / "budget_tradeoff",
            "Selective policy trade-off")
    if "frequency" in data and not data["frequency"].empty:
        model = data["frequency"].model.iloc[0]
        written["frequency"] = plot_selection_frequency(
            data["frequency"], model, out / "selection_frequency",
            f"Selection frequency: {LABELS.get(model, model)}")
    return written
