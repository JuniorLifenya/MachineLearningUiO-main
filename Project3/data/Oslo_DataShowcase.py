"""Oslo panel — baseline diagnostics for Part A."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
from oslo_Dataprices_download import load_oslo_prices, audit_price_panel

try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

FIG = Path("Project3/plots"); FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300,
                     "font.size": 10, "axes.titlesize": 11})

OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442"]


# --- helpers ---------------------------------------------------------------

def log_returns(prices):
    """Daily log returns: r_t = ln(P_t) - ln(P_{t-1}). First row NaN."""
    return np.log(prices).diff()


def realized_vol(prices, window):
    """sqrt(sum of squared returns over `window`). Non-annualized."""
    r = log_returns(prices)
    return np.sqrt((r ** 2).rolling(window, min_periods=window).sum())


# --- tables ----------------------------------------------------------------

def oslo_summary(prices):
    audit   = audit_price_panel(prices)
    r       = log_returns(prices)
    ann_ret = np.exp(252 * r.mean()) - 1        # geometric
    ann_vol = np.sqrt(252) * r.std()
    rv5     = realized_vol(prices, 5)
    n_ext   = int((r.abs() > 0.05).sum().sum())

    rows = [
        ("instruments",                  audit["instruments"]),
        ("first date",                   audit["start_date"]),
        ("last date",                    audit["end_date"]),
        ("trading sessions",             audit["observations"]),
        ("missing cells",                audit["missing_cells"]),
        ("non-positive cells",           audit["nonpositive_cells"]),
        ("passes structural audit",      audit["passes_basic_structural_audit"]),
        ("panel missingness %",          100 * prices.isna().sum().sum() / prices.size),
        ("median ann return %",          100 * ann_ret.median()),
        ("median ann vol %",             100 * ann_vol.median()),
        ("median RV5 (raw)",             rv5.median().median()),
        ("median RV5 (annualized %)",    100 * np.sqrt(252 / 5) * rv5.median().median()),
        ("worst single-day log ret %",   100 * r.min().min()),
        ("best single-day log ret %",    100 * r.max().max()),
        ("|log ret| > 5% events",        n_ext),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"]).set_index("metric")


def oslo_per_instrument(prices):
    r = log_returns(prices)
    rows = []
    for col in prices.columns:
        s = r[col].dropna()
        rows.append({
            "instrument":     col,
            "obs":            int(prices[col].notna().sum()),
            "ann_ret_%":      round(100 * (np.exp(252 * s.mean()) - 1), 2),
            "ann_vol_%":      round(100 * np.sqrt(252) * s.std(), 2),
            "skew":           round(float(s.skew()), 2),
            "kurt":           round(float(s.kurtosis()), 2),
            "worst_%":        round(100 * s.min(), 2),
            "best_%":         round(100 * s.max(), 2),
            "n_|r|>5%":       int((s.abs() > 0.05).sum()),
            "n_zero":         int((s == 0.0).sum()),      # stale-price hint
            "acf_lag1":       round(float(s.autocorr(1)), 3),
        })
    return pd.DataFrame(rows).set_index("instrument")


def ljung_box_pvalues(x, lags=(5, 20)):
    """Ljung-Box Q test on x. Uses scipy.special only (avoids scipy.signal)."""
    from scipy.special import gammaincc   # chi2 survival fn, no signal dep
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < max(lags) + 1:
        return {f"LB{m}": np.nan for m in lags}
    xc = x - x.mean()
    var = np.mean(xc ** 2)
    acf = np.array([np.sum(xc[k:] * xc[:-k]) / (n * var)
                    for k in range(1, max(lags) + 1)])
    out = {}
    for m in lags:
        Q = n * (n + 2) * np.sum(acf[:m] ** 2 / (n - np.arange(1, m + 1)))
        p = gammaincc(m / 2.0, Q / 2.0)   # chi2.sf(Q, df=m)
        out[f"LB{m}_r2_p"] = f"{p:.2e}"
    return out


def oslo_dependence(prices):
    r = log_returns(prices)
    rows = []
    for col in r.columns:
        s = r[col].dropna()
        row = {
            "instrument":   col,
            "acf_r_lag1":   round(float(s.autocorr(1)), 3),
            "acf_r_lag5":   round(float(s.autocorr(5)), 3),
            "acf_r_lag10":  round(float(s.autocorr(10)), 3),
        }
        row.update(ljung_box_pvalues(s ** 2, lags=(5, 20)))
        rows.append(row)
    return pd.DataFrame(rows).set_index("instrument")


# --- figures ---------------------------------------------------------------

def render_table(df, path, title):
    fig, ax = plt.subplots(figsize=(12, 0.5 + 0.40 * len(df)))
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=14, weight="bold")
    fmt = df.map(lambda v: f"{v:.2f}" if isinstance(v, float) else v)
    t = ax.table(cellText=fmt.values, rowLabels=df.index,
                 colLabels=df.columns.tolist(), loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.55)
    fig.savefig(path, format="svg", bbox_inches="tight"); plt.close(fig)


def oslo_normalized_prices(prices, path):
    """Base-100 chart so all 5 stocks start at 100 — cumulative performance."""
    normed = 100 * prices / prices.iloc[0]
    fig, ax = plt.subplots(figsize=(11, 4))
    for i, col in enumerate(normed.columns):
        ax.plot(normed.index, normed[col], lw=0.9, color=OKABE[i], label=col)
    ax.set_ylabel("cumulative price (base 100)")
    ax.set_title("Oslo equities: cumulative performance")
    ax.legend(ncol=5, fontsize=8, frameon=False); ax.grid(alpha=0.25)
    fig.savefig(path, format="svg", bbox_inches="tight"); plt.close(fig)


def oslo_returns(prices, path):
    r = log_returns(prices)
    fig, ax = plt.subplots(figsize=(11, 4))
    for i, col in enumerate(r.columns):
        ax.plot(r.index, r[col], lw=0.5, color=OKABE[i], label=col, alpha=0.85)
    ax.set_ylabel("daily log return")
    ax.set_title("Oslo equities: daily log returns")
    ax.legend(ncol=5, fontsize=8, frameon=False); ax.grid(alpha=0.2)
    fig.savefig(path, format="svg", bbox_inches="tight"); plt.close(fig)


def oslo_rolling_vol(prices, path, windows=(5, 20)):
    """Two panels: 5-session and 20-session RV — the same windows as features."""
    fig, axes = plt.subplots(len(windows), 1, figsize=(11, 3.5 * len(windows)), sharex=True)
    for ax, w in zip(axes, windows):
        rv = realized_vol(prices, w)
        for i, col in enumerate(rv.columns):
            ax.plot(rv.index, 100 * rv[col], lw=0.9, color=OKABE[i], label=col)
        ax.set_ylabel("RV (%)")
        ax.set_title(f"{w}-session realized volatility (non-annualized)")
        ax.legend(ncol=5, fontsize=8, frameon=False); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, format="svg", bbox_inches="tight"); plt.close(fig)


def oslo_correlation(prices, path):
    corr = log_returns(prices).corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns); ax.set_yticklabels(corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center",
                    color="white" if abs(corr.values[i,j]) > 0.6 else "black", fontsize=9)
    ax.set_title("Oslo: daily log-return correlation")
    fig.colorbar(im, ax=ax, label="Pearson r", shrink=0.85)
    fig.savefig(path, format="svg", bbox_inches="tight"); plt.close(fig)


# --- main ------------------------------------------------------------------

if __name__ == "__main__":
    prices = load_oslo_prices("oslo-stocks-2023-2025.xlsx")

    s = oslo_summary(prices)
    p = oslo_per_instrument(prices)
    d = oslo_dependence(prices)

    print(s, "\n"); print(p, "\n"); print(d)

    render_table(s, FIG / "s2_oslo_summary.svg",        "Oslo panel — summary")
    render_table(p, FIG / "s2_oslo_per_instrument.svg", "Oslo panel — per instrument")
    render_table(d, FIG / "s2_oslo_dependence.svg",     "Oslo panel — temporal dependence")

    oslo_normalized_prices(prices, FIG / "s2_oslo_normalized.svg")
    oslo_returns(prices,           FIG / "s2_oslo_returns.svg")
    oslo_rolling_vol(prices,       FIG / "s2_oslo_rolling_vol.svg")
    oslo_correlation(prices,       FIG / "s2_oslo_correlation.svg")

    # TEL zero-return investigation
    r = log_returns(prices)
    zero_dates = r.index[r["TEL"].fillna(1.0).eq(0.0)]
    if len(zero_dates) > 0:
        print(f"\nTEL zero-return days: {len(zero_dates)}")
        print(pd.Series(zero_dates).dt.day_name().value_counts())
        print(pd.Series(zero_dates).dt.month.value_counts())

    # Cross-stock correlation range
    c = r.corr().values
    off = c[np.triu_indices_from(c, k=1)]
    print(f"\nCorrelation: min={off.min():.3f} median={np.median(off):.3f} max={off.max():.3f}")