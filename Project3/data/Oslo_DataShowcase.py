"""Oslo panel — baseline diagnostics.
Produces: Project3/plots/s2_oslo_{summary,per_instrument,returns,
rolling_vol,correlation}.svg
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
from data.oslo_prices import load_oslo_prices, audit_price_panel

FIG = Path("Project3/plots"); FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300,
                     "font.size": 10, "axes.titlesize": 11})

OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442"]


def oslo_summary(prices):
    audit = audit_price_panel(prices)
    log_ret = np.log(prices / prices.shift(1))
    ann_ret = 252 * log_ret.mean()
    ann_vol = np.sqrt(252) * log_ret.std()
    n_extreme = int((log_ret.abs() > 0.05).sum().sum())

    rows = [
        ("instruments",             audit["instruments"]),
        ("first date",              audit["start_date"]),
        ("last date",               audit["end_date"]),
        ("trading sessions",        audit["observations"]),
        ("missing cells",           audit["missing_cells"]),
        ("non-positive cells",      audit["nonpositive_cells"]),
        ("passes basic contract",   audit["passes_basic_structural_audit"]),
        ("expected instruments present", audit["checks"]["expected_instruments"]),
        ("chronological order",          audit["checks"]["chronological_order"]),
        ("numeric prices",               audit["checks"]["numeric_prices"]),
        ("non-numeric cells", audit["non_numeric_cells"]),
        ("all-missing rows",  audit["all_missing_rows"]),
        ("panel missingness %",
         100 * prices.isna().sum().sum() / prices.size),
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
        ("|log ret| > 5% events",   n_extreme),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"]).set_index("metric")


def oslo_per_instrument(prices):
    log_ret = np.log(prices / prices.shift(1))
    rows = []
    for col in prices.columns:
        r = log_ret[col].dropna()
        rows.append({
            "instrument": col,
            "obs":        int(prices[col].notna().sum()),
            "ann_ret_%":  round(100 * 252 * r.mean(), 2),
            "ann_vol_%":  round(100 * np.sqrt(252) * r.std(), 2),
            "skew":       round(float(r.skew()), 2),
            "kurt":       round(float(r.kurtosis()), 2),
            "worst_%":    round(100 * r.min(), 2),
            "best_%":     round(100 * r.max(), 2),
            "n_|r|>5%":   int((r.abs() > 0.05).sum()),
        })
    return pd.DataFrame(rows).set_index("instrument")


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


def oslo_returns_drawdown(prices, path):
    log_ret = np.log(prices / prices.shift(1))
    ew = log_ret.mean(axis=1)
    q05 = ew.quantile(0.05)

    fig, ax = plt.subplots(figsize=(11, 4))
    for i, col in enumerate(log_ret.columns):
        ax.plot(log_ret.index, log_ret[col], lw=0.5,
                color=OKABE[i], label=col, alpha=0.85)
    ax.axhline(q05, color="#999", ls=":", lw=0.9,
               label=f"panel bottom 5% ({q05:.3f})")
    ax.set_ylabel("daily log return")
    ax.set_title("Oslo — daily returns by instrument")
    ax.legend(frameon=False, ncol=6, fontsize=8, loc="lower left")
    ax.grid(alpha=0.2)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def oslo_rolling_vol(prices, path, window=21):
    log_ret = np.log(prices / prices.shift(1))
    rv = np.sqrt(252) * log_ret.rolling(window).std()

    fig, ax = plt.subplots(figsize=(11, 4))
    for i, col in enumerate(rv.columns):
        ax.plot(rv.index, 100 * rv[col], lw=0.9,
                color=OKABE[i], label=col)
    ax.set_ylabel("annualized volatility (%)")
    ax.set_title(f"Oslo — {window}-session rolling volatility")
    ax.legend(frameon=False, ncol=6, fontsize=8)
    ax.grid(alpha=0.25)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def oslo_correlation(prices, path):
    log_ret = np.log(prices / prices.shift(1))
    corr = log_ret.corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns)
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i,j]:.2f}",
                    ha="center", va="center",
                    color="white" if abs(corr.values[i,j]) > 0.6 else "black",
                    fontsize=9)
    ax.set_title("Oslo — daily log-return correlation")
    fig.colorbar(im, ax=ax, label="Pearson r", shrink=0.85)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    prices = load_oslo_prices("oslo-stocks-2023-2025.xlsx")

    s = oslo_summary(prices)
    p = oslo_per_instrument(prices)
    print(s); print()
    print(p)

    render_table(s, FIG / "s2_oslo_summary.svg",
                 "Oslo panel — summary")
    render_table(p, FIG / "s2_oslo_per_instrument.svg",
                 "Oslo panel — per instrument")
    oslo_returns_drawdown(prices, FIG / "s2_oslo_returns.svg")
    oslo_rolling_vol(prices,    FIG / "s2_oslo_rolling_vol.svg")
    oslo_correlation(prices,    FIG / "s2_oslo_correlation.svg")