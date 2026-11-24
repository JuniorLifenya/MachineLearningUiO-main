"""Kaggle S&P 500 + SPY — baseline diagnostics.
Produces: Project3/plots/s3_kaggle_{summary,coverage_over_time,
coverage_dist,extreme_returns,outliers,benchmark_vs_ew}.svg
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
from data import Kaggle_Data_download

FIG = Path("Project3/plots"); FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300,
                     "font.size": 10, "axes.titlesize": 11})


def kaggle_summary(panel, benchmark="SPY"):
    valid = panel.dropna(how="all")
    per_inst = valid.notna().sum()
    log_ret = np.log(valid / valid.shift(1))
    ann_ret = 252 * log_ret.mean()
    ann_vol = np.sqrt(252) * log_ret.std()
    abs_ret = log_ret.abs()

    rows = [
        ("instruments (incl. benchmark)", len(panel.columns)),
        ("benchmark",                     benchmark),
        ("first date",                    str(valid.index.min().date())),
        ("last date",                     str(valid.index.max().date())),
        ("trading sessions",              len(valid)),
        ("panel missingness %",
         100 * valid.isna().sum().sum() / valid.size),
        ("benchmark missing %",
         100 * valid[benchmark].isna().mean()),
        ("obs (p10/p50/p90)",
         f"{int(per_inst.quantile(.10))} / "
         f"{int(per_inst.median())} / "
         f"{int(per_inst.quantile(.90))}"),
        ("instruments < 100 obs",         int((per_inst < 100).sum())),
        ("ann return % (p10/p50/p90)",
         f"{100*ann_ret.quantile(.10):.1f} / "
         f"{100*ann_ret.median():.1f} / "
         f"{100*ann_ret.quantile(.90):.1f}"),
        ("ann vol % (p10/p50/p90)",
         f"{100*ann_vol.quantile(.10):.1f} / "
         f"{100*ann_vol.median():.1f} / "
         f"{100*ann_vol.quantile(.90):.1f}"),
        ("worst single-day log ret %",
         f"{100*log_ret.min().min():.2f}"),
        ("best single-day log ret %",
         f"{100*log_ret.max().max():.2f}"),
        ("|log ret| > 50% events",        int((abs_ret > 0.50).sum().sum())),
        ("|log ret| > 100% events",       int((abs_ret > 1.00).sum().sum())),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"]).set_index("metric")


def render_table(df, path, title):
    fig, ax = plt.subplots(figsize=(12, 0.5 + 0.40 * len(df)))
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=14, weight="bold")
    fmt = df.map(lambda v: f"{v:.2f}" if isinstance(v, float) else v)
    t = ax.table(cellText=fmt.values, rowLabels=df.index,
                 colLabels=df.columns.tolist(),
                 loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.55)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def coverage_over_time(panel, path):
    valid = panel.notna()
    counts = valid.resample("YE").sum()
    n_instr = (counts > 0).sum(axis=1)
    n_full = (counts >= 200).sum(axis=1)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(n_instr.index.year, n_instr.values, "o-",
            color="#0072B2", label="any observation that year")
    ax.plot(n_full.index.year,  n_full.values,  "s-",
            color="#D55E00", label="≥ 200 observations that year")
    ax.set_ylabel("number of instruments")
    ax.set_xlabel("year")
    ax.set_title("Kaggle — active instrument count by year")
    ax.legend(frameon=False); ax.grid(alpha=0.25)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def coverage_distribution(panel, path):
    per_inst = panel.notna().sum().sort_values()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.hist(per_inst.values, bins=40, color="#0072B2",
            edgecolor="white", lw=0.4)
    ax.axvline(per_inst.median(), color="#D55E00", ls="--",
               lw=1.2, label=f"median = {int(per_inst.median())}")
    ax.set_xlabel("non-missing sessions per instrument")
    ax.set_ylabel("number of instruments")
    ax.set_title(f"Kaggle — coverage distribution "
                 f"({len(per_inst)} instruments)")
    ax.legend(frameon=False); ax.grid(alpha=0.25)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def extreme_returns_scatter(panel, path, threshold=0.5):
    log_ret = np.log(panel / panel.shift(1))
    fig, ax = plt.subplots(figsize=(12, 4))

    # base layer: all returns as light gray
    stack = log_ret.stack()
    ax.scatter(stack.index.get_level_values(0), stack.values,
               s=1, c="#bbbbbb", alpha=0.15, rasterized=True,
               label="all returns")

    # overlay: extreme events
    extreme = stack[stack.abs() > threshold]
    if len(extreme):
        colors = ["#D55E00" if v > 0 else "#0072B2" for v in extreme.values]
        ax.scatter(extreme.index.get_level_values(0), extreme.values,
                   s=16, c=colors, edgecolors="black", linewidths=0.3,
                   label=f"|r| > {threshold:.0%}  (n={len(extreme)})")

    ax.set_ylim(-1.3, 1.3)
    ax.set_ylabel("daily log return")
    ax.set_xlabel("date")
    ax.set_title(f"Kaggle — |log return| > {threshold:.0%} events")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.2)

    # annotate the five most extreme
    top5 = extreme.reindex(extreme.abs().sort_values(ascending=False).index[:5])
    for (date, ticker), val in top5.items():
        ax.annotate(f"{ticker}\n{date.date()}\n{val:.2f}",
                    (date, val), fontsize=7,
                    xytext=(6, 0), textcoords="offset points",
                    color="#333")
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def outlier_instruments(panel, path, top_n=10, threshold=0.5):
    log_ret = np.log(panel / panel.shift(1))
    worst_abs_val = log_ret.abs().max()
    worst_abs_idx = log_ret.abs().idxmax()
    worst_neg_val = log_ret.min()
    worst_neg_idx = log_ret.idxmin()
    n_extreme = (log_ret.abs() > threshold).sum()

    top_abs = worst_abs_val.sort_values(ascending=False).index[:top_n]

    table = pd.DataFrame({
        "worst_abs_%": (100 * worst_abs_val[top_abs]).round(1),
        "worst_abs_date": worst_abs_idx[top_abs].dt.date,
        "worst_neg_%": (100 * worst_neg_val[top_abs]).round(1),
        "worst_neg_date": worst_neg_idx[top_abs].dt.date,
        "n_|r|>50%": n_extreme[top_abs],
    })

    render_table(table,
                 FIG / "s3_kaggle_outliers.svg",
                 f"Kaggle — top {top_n} instruments by |worst-day return|")
    return table


def benchmark_vs_ew(panel, benchmark, path):
    """Monthly aggregation — daily returns are too noisy for this plot."""
    log_ret = np.log(panel / panel.shift(1))
    eq_w = log_ret.drop(columns=[benchmark]).mean(axis=1).dropna()
    bench = log_ret[benchmark].dropna()

    eq_m = eq_w.resample("ME").apply(lambda x: x.sum())
    bench_m = bench.resample("ME").apply(lambda x: x.sum())

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].plot(eq_w.index, eq_w.values, lw=0.4,
                 color="#56B4E9", label="equal-weight", alpha=0.7)
    axes[0].plot(bench.index, bench.values, lw=0.4,
                 color="#D55E00", label=benchmark, alpha=0.7)
    axes[0].set_ylabel("daily log return")
    axes[0].set_title("Daily returns")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.2)

    axes[1].plot(eq_m.index, eq_m.values, "o-", ms=2,
                 color="#56B4E9", label="equal-weight")
    axes[1].plot(bench_m.index, bench_m.values, "s-", ms=2,
                 color="#D55E00", label=benchmark)
    rho = eq_m.corr(bench_m)
    axes[1].set_ylabel("monthly log return")
    axes[1].set_title(f"Monthly returns (ρ = {rho:.3f})")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(alpha=0.2)

    fig.suptitle("Benchmark vs equal-weight constituents", y=1.02)
    fig.tight_layout()
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return rho


if __name__ == "__main__":
    panel = Kaggle_Data_download.load_kaggle_panel_with_benchmark("SPY")

    s = kaggle_summary(panel)
    print(s); print()

    render_table(s, FIG / "s3_kaggle_summary.svg",
                 "Kaggle S&P 500 + SPY — summary")

    coverage_over_time(panel,   FIG / "s3_kaggle_coverage_time.svg")
    coverage_distribution(panel, FIG / "s3_kaggle_coverage_dist.svg")
    extreme_returns_scatter(panel, FIG / "s3_kaggle_extreme.svg")
    outliers = outlier_instruments(panel, FIG)
    print(); print(outliers)
    rho = benchmark_vs_ew(panel, "SPY",
                          FIG / "s3_kaggle_benchmark.svg")
    print(f"\nSPY vs equal-weight monthly correlation: {rho:.3f}")