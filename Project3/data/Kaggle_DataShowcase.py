"""Section 3: baseline diagnostics on the Kaggle S&P 500 panel + SPY.

Produces:
    public/figs/s3_panel_summary.svg
    public/figs/s3_coverage_distribution.svg
    public/figs/s3_returns_benchmark.svg
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import sys

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
from data import Kaggle_Data_download

FIG = Path("Project3/plots"); FIG.mkdir(parents=True, exist_ok=True)


def panel_summary(panel, benchmark="SPY"):
    """One-row-per-description summary table, Oslo-Section-2 style."""
    valid = panel.dropna(how="all")
    per_inst = valid.notna().sum()
    returns = np.log(valid / valid.shift(1))

    rows = [
        ("instruments (incl. benchmark)", len(panel.columns)),
        ("benchmark",                     benchmark),
        ("first date",                    str(valid.index.min().date())),
        ("last date",                     str(valid.index.max().date())),
        ("trading sessions",              len(valid)),
        ("median per-instrument obs",     int(per_inst.median())),
        ("min per-instrument obs",        int(per_inst.min())),
        ("instruments < 100 obs",         int((per_inst < 100).sum())),
        ("panel missingness %",
         100 * valid.isna().sum().sum() / valid.size),
        ("benchmark missing %",
         100 * valid[benchmark].isna().mean()),
        ("median ann. return %",
         100 * 252 * returns.median().median()),
        ("median ann. vol %",
         100 * np.sqrt(252) * returns.std().median()),
        ("worst single-day return %",
         100 * returns.min().min()),
        ("best single-day return %",
         100 * returns.max().max()),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"]).set_index("metric")


def render_summary(df, path, title):
    fig, ax = plt.subplots(figsize=(11, 0.55 + 0.42 * len(df)))
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=12)
    t = ax.table(
        cellText=[[f"{v:.2f}" if isinstance(v, float) else v] for v in df["value"]],
        rowLabels=df.index,
        colLabels=["value"],
        loc="center", cellLoc="left",
        colWidths=[0.35],
    )
    t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1, 1.5)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def coverage_distribution(panel, path):
    """How many sessions does each instrument cover?"""
    counts = panel.notna().sum().sort_values()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(counts)), counts.values, lw=0.8, color="#0072B2")
    ax.set_yscale("log")
    ax.set_xlabel("instrument rank (sorted by coverage)")
    ax.set_ylabel("non-missing sessions")
    ax.set_title(f"Coverage distribution across {len(counts)} instruments")
    ax.grid(alpha=0.25, which="both")
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def returns_vs_benchmark(panel, benchmark, path):
    """Compare benchmark returns to the equal-weight panel returns."""
    log_ret = np.log(panel / panel.shift(1))
    equal_w = log_ret.drop(columns=[benchmark]).mean(axis=1)
    bench   = log_ret[benchmark]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(equal_w.index, equal_w.values, lw=0.5,
            color="#56B4E9", label="equal-weight panel", alpha=0.7)
    ax.plot(bench.index, bench.values, lw=0.5,
            color="#D55E00", label=benchmark, alpha=0.7)
    ax.set_ylabel("daily log return")
    ax.set_title("Benchmark vs equal-weight constituents")
    ax.legend(frameon=False); ax.grid(alpha=0.2)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)

def flag_extreme_days(panel, threshold=0.5):
    """Return log returns whose |r| > threshold — candidates for data errors."""
    log_ret = np.log(panel / panel.shift(1))
    mask = log_ret.abs() > threshold
    if mask.any().any():
        offending = mask.stack()
        offending = offending[offending]
        print(offending.head(20))
    return mask

if __name__ == "__main__":
    panel = Kaggle_Data_download.load_kaggle_panel_with_benchmark("SPY")

    summary = panel_summary(panel, "SPY")
    print(summary)

    render_summary(summary,
                   FIG / "s3_panel_summary.svg",
                   "Kaggle S&P 500 + SPY — panel summary")
    coverage_distribution(panel, FIG / "s3_coverage_distribution.svg")
    returns_vs_benchmark(panel, "SPY", FIG / "s3_returns_benchmark.svg")