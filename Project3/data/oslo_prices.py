from pathlib import Path
import pandas as pd
import numpy as np

_PKG_DIR     = Path(__file__).resolve().parent          # data/
_PROJECT_DIR = _PKG_DIR.parent                          # proj_1_OsloQualityPipeline/

_DEFAULT_CANDIDATES = (
    _PKG_DIR     / "oslo-stocks-2023-2025.xlsx",
    _PROJECT_DIR / "oslo-stocks-2023-2025.xlsx",
)


def load_oslo_prices(
    path: str | Path = "oslo-stocks-2023-2025.xlsx",
) -> pd.DataFrame:
    """Load the adjusted closing-price panel for the Oslo portfolio."""

    if path is None:
        candidates = list(_DEFAULT_CANDIDATES)
    else:
        p = Path(path)
        if p.is_absolute():
            candidates = [p]
        else:
            candidates = [_PKG_DIR / p, _PROJECT_DIR / p, p]

    for candidate in candidates:
        if candidate.exists():
            path = candidate
            break
    else:
        raise FileNotFoundError(
            f"Price file not found. Searched: {[str(c) for c in candidates]}"
        )

    raw = pd.read_excel(
        path,
        sheet_name="Adjusted Prices",
        header=2,
    )

    raw.columns = raw.columns.astype(str).str.strip()

    if "Date" not in raw.columns:
        raise ValueError("Expected a 'Date' column.")

    raw["Date"] = pd.to_datetime(
        raw["Date"], errors="coerce"
    )

    if raw["Date"].isna().any():
        raise ValueError("One or more rows have invalid dates.")

    prices = (
        raw.set_index("Date")
        .sort_index()
        .dropna(how="all")
    )

    if prices.columns.has_duplicates:
        raise ValueError("Duplicate instrument columns found.")

    for col in prices.columns:
        original = prices[col]
        converted = pd.to_numeric(
            original, errors="coerce"
        )

        # Do not silently convert unexpected text into missing data.
        bad = original.notna() & converted.isna()

        if bad.any():
            raise ValueError(
                f"Non-numeric price values in {col}."
            )

        prices[col] = converted

    prices.index.name = "Date"

    return prices

def audit_price_panel(prices: pd.DataFrame) -> dict:
    """Report basic structural properties of a price panel."""

    values = prices.to_numpy(dtype=float)

    missing_cells = int(prices.isna().sum().sum())
    infinite_cells = int(np.isinf(values).sum())

    nonpositive_cells = int(
        ((prices <= 0) & prices.notna()).sum().sum()
    )

    duplicate_dates = int(
        prices.index.duplicated().sum()
    )

    checks = {
        "unique_dates": duplicate_dates == 0,
        "no_missing_price_cells": missing_cells == 0,
        "finite_prices": infinite_cells == 0,
        "positive_prices": nonpositive_cells == 0,
    }

    return {
        "observations": len(prices),
        "instruments": len(prices.columns),
        "start_date": str(prices.index.min().date()),
        "end_date": str(prices.index.max().date()),
        "missing_cells": missing_cells,
        "infinite_cells": infinite_cells,
        "nonpositive_cells": nonpositive_cells,
        "duplicate_dates": duplicate_dates,
        "missing_by_instrument": (
            prices.isna().sum().to_dict()
        ),
        "checks": checks,
        "passes_basic_contract": all(checks.values()),
    }

if __name__ == "__main__":
    prices = load_oslo_prices("oslo-stocks-2023-2025.xlsx")
    report = audit_price_panel(prices)
    for key, value in report.items():
        print(f"{key}: {value}")