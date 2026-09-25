"""Static report figures (matplotlib), drawn only from metrics.json records."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed (CI, servers)
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from skytrust.config import REPO_ROOT  # noqa: E402
from skytrust.report import display_name  # noqa: E402

FIG_DIR = REPO_ROOT / "docs" / "figures"
# PNGs carry no timestamp/version metadata, so re-running `report` doesn't churn git diffs.
SAVE_KW = {"dpi": 130, "bbox_inches": "tight", "metadata": {"Software": None}}
EMPHASIS = {"blend": "#d62728", "equal_weight": "#1f77b4", "climatology": "#7f7f7f"}
# Fixed colour per forecast model so the same model looks the same in every figure, and no
# model can be drawn in an emphasis colour.
MODEL_COLORS = {"gfs": "#17becf", "hrrr": "#ff7f0e", "ecmwf": "#2ca02c", "gem": "#8c564b",
                "icon": "#9467bd"}  # fmt: skip


def color_for(method: str) -> str:
    if method in EMPHASIS:
        return EMPHASIS[method]
    return MODEL_COLORS.get(method.rsplit("_", 1)[0], "#bcbd22")


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)
    return path


def lead_curves(records: pd.DataFrame, label: str, path: Path) -> Path:
    """BSS and false-clear rate vs lead for every probabilistic method; 95 % CI bands on the
    emphasised ones (equal-weight and blend)."""
    o = records[(records["label"] == label) & (records["subset_type"] == "overall")]
    o = o[o["kind"] == "prob"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric, title in [
        (axes[0], "bss", "Brier Skill Score vs climatology (higher = better)"),
        (axes[1], "false_clear_rate", "False-clear rate at P ≥ 0.5 (lower = better)"),
    ]:
        for method, g in o.groupby("method", sort=False):
            g = g.sort_values("lead")
            emphasised = method in EMPHASIS
            color = color_for(method)
            ax.plot(g["lead"], g[metric], marker="o", ms=3.5, lw=2.4 if emphasised else 1.1,
                    color=color, alpha=1 if emphasised else 0.8,
                    label=display_name(method))  # fmt: skip
            if emphasised and method != "climatology":
                ax.fill_between(
                    g["lead"], g[f"{metric}_lo"], g[f"{metric}_hi"], color=color, alpha=0.15
                )
        ax.set_xlabel("Lead time (days ahead)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xticks(sorted(o["lead"].unique()))
    axes[0].axhline(0, color="black", lw=0.8)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle(f"Skill decay with lead time (test set, {label} label)", fontsize=11)
    return _save(fig, path)


def reliability(
    reliability_rows: list[dict], methods: list[str], label: str, lead: int, path: Path
) -> Path:
    """Reliability diagram: observed frequency vs forecast probability, with bin counts."""
    rows = [
        r
        for r in reliability_rows
        if r["label"] == label and r["lead"] == lead and r["method"] in methods
    ]
    fig, (ax, axn) = plt.subplots(2, 1, figsize=(5.6, 6.4), height_ratios=[3, 1], sharex=True)
    ax.plot([0, 1], [0, 1], color="black", lw=0.8, ls="--", label="Perfectly calibrated")
    width = 0.1 / max(len(rows), 1)
    for i, r in enumerate(sorted(rows, key=lambda r: methods.index(r["method"]))):
        bins = pd.DataFrame(r["bins"])
        filled = bins[bins["n"] > 0]
        color = color_for(r["method"])
        ax.plot(
            filled["mean_p"],
            filled["observed"],
            marker="o",
            color=color,
            label=display_name(r["method"]),
        )
        axn.bar(bins["lo"] + width * (i + 0.5), bins["n"], width=width, color=color, alpha=0.8)
    ax.set_ylabel("Observed frequency of usable nights")
    ax.set_title(f"Reliability, lead {lead} ({label} label, test set)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    axn.set_ylabel("Nights per bin")
    axn.set_xlabel("Forecast P(usable)")
    axn.grid(alpha=0.3)
    return _save(fig, path)


def site_skill(
    records: pd.DataFrame, methods: list[str], label: str, lead: int, path: Path
) -> Path:
    s = records[(records["label"] == label) & (records["lead"] == lead)
                & (records["subset_type"] == "site") & records["method"].isin(methods)]  # fmt: skip
    sites = sorted(s["subset"].unique())
    fig, ax = plt.subplots(figsize=(7, 3.8))
    width = 0.8 / len(methods)
    for i, method in enumerate(methods):
        g = s[s["method"] == method].set_index("subset").reindex(sites)
        x = [j + (i - (len(methods) - 1) / 2) * width for j in range(len(sites))]
        err = [g["bss"] - g["bss_lo"], g["bss_hi"] - g["bss"]]
        ax.bar(x, g["bss"], width=width, yerr=err, capsize=3, color=color_for(method),
               label=display_name(method))  # fmt: skip
    ax.set_xticks(range(len(sites)), sites)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Brier Skill Score (95% CI)")
    ax.set_title(f"Skill by site, lead {lead} ({label} label)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, path)
