"""Section 2 (revised): baseline diagnostics on the Oslo panel.

Produces:
    public/figs/s2_oslo_summary.svg
    public/figs/s2_oslo_per_instrument.svg
    public/figs/s2_oslo_returns.svg
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


def oslo_summary(prices):
    audit = audit_price_panel(prices)
    log_ret = np.log(prices / prices.shift(1))
    rows = [
        ("instruments",             audit["instruments"]),
        ("first date",              audit["start_date"]),
        ("last date",               audit["end_date"]),
        ("trading sessions",        audit["observations"]),
        ("missing cells",           audit["missing_cells"]),
        ("non-positive cells",      audit["nonpositive_cells"]),
        ("passes basic contract",   audit["passes_basic_contract"]),
        ("panel missingness %",
         100 * prices.isna().sum().sum() / prices.size),
        ("median ann. return %",
         100 * 252 * log_ret.median().median()),
        ("median ann. vol %",
         100 * np.sqrt(252) * log_ret.std().median()),
        ("worst single-day log ret %",
         100 * log_ret.min().min()),
        ("best single-day log ret %",
         100 * log_ret.max().max()),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"]).set_index("metric")


def oslo_per_instrument(prices):
    log_ret = np.log(prices / prices.shift(1))
    audit = audit_price_panel(prices)
    rows = []
    for col in prices.columns:
        r = log_ret[col].dropna()
        rows.append({
            "instrument": col,
            "obs":        int(prices[col].notna().sum()),
            "ann_ret_%":  round(100 * 252 * r.mean(), 2),
            "ann_vol_%":  round(100 * np.sqrt(252) * r.std(), 2),
            "worst_%":    round(100 * r.min(), 2),
            "best_%":     round(100 * r.max(), 2),
        })
    return pd.DataFrame(rows).set_index("instrument")


def render_table(df, path, title):
    fig, ax = plt.subplots(figsize=(11, 0.55 + 0.42 * len(df)))
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=12)
    fmt = df.map(lambda v: f"{v:.2f}" if isinstance(v, float) else v)
    t = ax.table(
        cellText=fmt.values,
        rowLabels=df.index,
        colLabels=df.columns.tolist(),
        loc="center", cellLoc="center",
    )
    t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1, 1.5)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def oslo_returns(prices, path):
    log_ret = np.log(prices / prices.shift(1))
    fig, ax = plt.subplots(figsize=(10, 4))
    for col in log_ret.columns:
        ax.plot(log_ret.index, log_ret[col], lw=0.4, label=col, alpha=0.7)
    ax.set_ylabel("daily log return")
    ax.set_title("Oslo panel — daily returns, all instruments")
    ax.legend(frameon=False, ncol=5, fontsize=8)
    ax.grid(alpha=0.2)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    prices = load_oslo_prices("oslo-stocks-2023-2025.xlsx")

    summary = oslo_summary(prices)
    print(summary)

    per_inst = oslo_per_instrument(prices)
    print()
    print(per_inst)

    render_table(summary,  FIG / "s2_oslo_summary.svg",
                 "Oslo panel — summary")
    render_table(per_inst, FIG / "s2_oslo_per_instrument.svg",
                 "Oslo panel — per instrument")
    oslo_returns(prices,   FIG / "s2_oslo_returns.svg")