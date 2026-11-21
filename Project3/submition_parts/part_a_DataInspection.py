
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.oslo_prices import (
    load_oslo_prices,
    audit_price_panel
)


def main():

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    prices = load_oslo_prices(
        "oslo-stocks-2023-2025.xlsx"
    )

    # ========================================================
    # 2. DATA QUALITY AUDIT
    # ========================================================

    report = audit_price_panel(prices)

    print("\n========== DATA AUDIT ==========")

    for key, value in report.items():
        print(f"{key}: {value}")

    # ========================================================
    # 3. DATA DIMENSIONS AND COVERAGE
    # ========================================================

    coverage = prices.notna().sum().sort_values(
        ascending=False
    )

    print("\n========== DATA DIMENSIONS ==========")
    print("Observations:", prices.shape[0])
    print("Instruments:", prices.shape[1])
    print("Date range:", prices.index.min(), "to", prices.index.max())

    print("\n========== TOP 15 INSTRUMENTS BY COVERAGE ==========")
    print(coverage.head(15))

    print("\n========== LOWEST 15 COVERAGE ==========")
    print(coverage.tail(15))

    # ========================================================
    # 4. MISSING DATA
    # ========================================================

    missing = prices.isna().sum()

    print("\n========== MISSING DATA ==========")
    print("Total missing cells:", prices.isna().sum().sum())
    print("Percentage missing:",
          100 * prices.isna().sum().sum() / prices.size)

    print("\nMissing by instrument:")
    print(missing.sort_values(ascending=False).head(15))

    # ========================================================
    # 5. EXAMPLE OBSERVATIONS
    # ========================================================

    selected = prices.columns.tolist()

    print("\n========== EXAMPLE INSTRUMENTS ==========")
    print("Selected:", selected)
    print(prices[selected].head(10))

    # ========================================================
    # 6. INITIAL PRICE VISUALIZATION
    # ========================================================

    fig, ax = plt.subplots(figsize=(12, 5))

    for ticker in selected:
        ax.plot(
            prices.index,
            prices[ticker],
            label=ticker
    )

    ax.set_title("Oslo equities: Adjusted closing prices")
    ax.set_xlabel("Date")
    ax.set_ylabel("Adjusted closing price")
    ax.legend()
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.show()

    # ========================================================
    # 7. RETURNS AND REALIZED VOLATILITY
    # ========================================================

    log_ret = np.log(prices).diff()

    squared_ret = log_ret ** 2
    rv5 = np.sqrt(
        squared_ret.rolling(window=5, min_periods=5).sum()
    )

    # ========================================================
    # 8. THREE-PANEL FIGURE (prices, returns, RV)
    # ========================================================

    selected = prices.columns.tolist()   # all 5 — no selection to defend

    fig, axes = plt.subplots(
        2, 1, figsize=(14, 12), sharex=True,
    )

    for ticker in selected:
        axes[0].plot(log_ret.index, log_ret[ticker],
                    label=ticker, linewidth=0.8)
        axes[1].plot(rv5.index, rv5[ticker], label=ticker)


    axes[0].set_title("Oslo equities: Daily log returns")
    axes[0].set_ylabel("Log return")

    axes[1].set_title("Oslo equities: 5-session realized volatility")
    axes[1].set_ylabel("RV (5)")

    for ax in axes:
        ax.grid(alpha=0.25)
    axes[0].legend(ncol=5, loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
