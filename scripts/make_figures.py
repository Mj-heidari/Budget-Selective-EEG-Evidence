"""Render the manuscript figures from the exported run artefacts."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN = Path("/home/claude/build/p8run/out")
OUT = Path(__file__).resolve().parents[1] / "paper/figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 200, "savefig.bbox": "tight"})
BLUE, ORANGE, GREY = "#2b6cb0", "#dd6b20", "#718096"

LABELS = {"proposed": "no augmentation", "proposed_ros": "oversampling",
          "proposed_jitter": "Gaussian jitter", "proposed_gan": "cLSGAN",
          "proposed_gated": "cLSGAN + support filter", "proposed_guided": "reliability-guided"}


def augmentation_effects():
    rows = json.loads((RUN / "augmentation_effects_all.json").read_text())
    order = ["proposed_guided", "proposed_ros", "proposed_gated", "proposed_gan", "proposed_jitter"]
    rows = sorted(rows, key=lambda r: order.index(r["model"]))
    fig, ax = plt.subplots(figsize=(3.3, 1.85))
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        d = r["delta_balanced_accuracy"] * 100
        lo, hi = [v * 100 for v in r["conditional_cluster_ci95"]]
        colour = BLUE if d >= 0 else ORANGE
        ax.plot([lo, hi], [y, y], color=colour, lw=1.6, solid_capstyle="round")
        ax.plot([d], [y], "o", color=colour, ms=4.5, zorder=3)
    ax.axvline(0, color=GREY, lw=.8, ls="--")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([LABELS[r["model"]] for r in reversed(rows)])
    ax.set_xlabel("balanced accuracy vs no augmentation (points)")
    ax.set_ylim(-.6, len(rows) - .4)
    fig.savefig(OUT / "augmentation_effects.pdf")
    plt.close(fig)


def budget_tradeoff():
    d = pd.read_csv(RUN / "budget_sweep_pooled.csv")
    pooled = pd.DataFrame(json.loads((RUN / "augmentation/pooled_metrics.json").read_text()))
    fixed = {int(r.duration): float(r.balanced_accuracy)
             for _, r in pooled[pooled.model == "proposed_guided"].iterrows()}
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for policy, colour, marker in (("agreement", BLUE, "o"), ("margin", ORANGE, "s")):
        sub = d[(d.policy == policy) & (d.model == "proposed_guided")].sort_values("coverage")
        ax.plot(sub.coverage * 100, sub.accepted_balanced_accuracy, marker + "-",
                color=colour, ms=4.5, lw=1.4, label=f"selective ({policy})")
        for _, r in sub.iterrows():
            ax.annotate(f"{r.mean_clean_seconds_per_condition:.0f}s",
                        (r.coverage * 100, r.accepted_balanced_accuracy),
                        textcoords="offset points", xytext=(4, 4), fontsize=6, color=colour)
    for duration, style in ((20, ":"), (40, "--")):
        ax.axhline(fixed[duration], color=GREY, lw=.9, ls=style)
        ax.annotate(f"fixed {duration}\u2009s, all children", (41.5, fixed[duration] + .004),
                    fontsize=6, color=GREY, va="bottom", ha="left")
    ax.set_xlabel("coverage (% of held-out children given a decision)")
    ax.set_ylabel("balanced accuracy on accepted")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.set_xlim(40, 106)
    fig.savefig(OUT / "budget_tradeoff.pdf")
    plt.close(fig)


def release_spread():
    r = pd.read_csv(RUN / "augmentation/release_metrics.csv")
    r = r[(r.model == "proposed_guided")]
    releases = sorted(r.held_release.unique())
    fig, ax = plt.subplots(figsize=(3.3, 1.9))
    width = .36
    x = np.arange(len(releases))
    for offset, duration, colour in ((-width / 2, 20, GREY), (width / 2, 40, BLUE)):
        vals = [float(r[(r.held_release == s) & (r.duration == duration)].balanced_accuracy.iloc[0])
                for s in releases]
        ax.bar(x + offset, vals, width, color=colour, label=f"{duration}s budget")
    ax.axhline(.5, color="#c53030", lw=.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([s.split("_")[0] for s in releases])
    ax.set_xlabel("held-out release")
    ax.set_ylabel("balanced accuracy")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper right")
    fig.savefig(OUT / "release_spread.pdf")
    plt.close(fig)


if __name__ == "__main__":
    augmentation_effects()
    budget_tradeoff()
    release_spread()
    print("figures written to", OUT)
