from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

_PKG_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _PKG_DIR.parent

DEFAULT_FILENAME = "oslo-stocks-2023-2025.xlsx"

EXPECTED_INSTRUMENTS = (
    "EQNR",
    "DNB",
    "TEL",
    "NAS",
    "NHY",
)


# ------------------------------------------------------------
# Path resolution
# ------------------------------------------------------------

def resolve_price_file(
    path: str | Path | None = DEFAULT_FILENAME,
) -> Path:
    """
    Locate the Oslo Excel workbook.

    Relative paths are searched in:
        1. data/
        2. project root/
        3. current working directory

    An absolute path is used directly.
    """

    if path is None:
        candidates = [
            _PKG_DIR / DEFAULT_FILENAME,
            _PROJECT_DIR / DEFAULT_FILENAME,
        ]

    else:
        p = Path(path)

        if p.is_absolute():
            candidates = [p]

        else:
            candidates = [
                _PKG_DIR / p,
                _PROJECT_DIR / p,
                Path.cwd() / p,
            ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(
        "Price file not found. Searched:\n"
        + "\n".join(str(p) for p in candidates)
    )


# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------

def load_oslo_prices(
    path: str | Path | None = DEFAULT_FILENAME,
) -> pd.DataFrame:
    """
    Load the five-stock Oslo adjusted-price panel.

    Returns
    -------
    pd.DataFrame
        Rows: recorded dates
        Columns: EQNR, DNB, TEL, NAS, NHY

    Missing price values are preserved for auditing.
    """

    file_path = resolve_price_file(path)

    raw = pd.read_excel(
        file_path,
        sheet_name="Adjusted Prices",
        header=2,
    )

    # Normalize column labels.
    raw.columns = raw.columns.astype(str).str.strip()

    # Remove only completely blank spreadsheet rows.
    # A row with a valid Date but no prices must be retained.
    raw = raw.dropna(how="all").copy()

    if raw.empty:
        raise ValueError("The selected worksheet is empty.")

    if raw.columns.has_duplicates:
        raise ValueError("Duplicate column names found.")

    if "Date" not in raw.columns:
        raise ValueError("Expected a 'Date' column.")

    # Verify the expected instrument schema.
    observed_instruments = [
        col for col in raw.columns if col != "Date"
    ]

    missing_columns = sorted(
        set(EXPECTED_INSTRUMENTS) - set(observed_instruments)
    )

    unexpected_columns = sorted(
        set(observed_instruments) - set(EXPECTED_INSTRUMENTS)
    )

    if missing_columns or unexpected_columns:
        raise ValueError(
            "Unexpected instrument columns.\n"
            f"Missing: {missing_columns}\n"
            f"Unexpected: {unexpected_columns}"
        )

    # Parse dates.
    dates = pd.to_datetime(
        raw.pop("Date"),
        errors="coerce",
    )

    invalid_dates = dates.isna()

    if invalid_dates.any():
        rows = invalid_dates[invalid_dates].index.tolist()

        raise ValueError(
            f"Missing or invalid dates in spreadsheet rows: "
            f"{rows[:10]}"
        )

    # This dataset is intended to contain daily observations,
    # not intraday timestamps.
    has_time_component = (
        dates != dates.dt.normalize()
    )

    if has_time_component.any():
        raise ValueError(
            "Unexpected intraday timestamps found in Date column."
        )

    # Convert price columns without silently accepting text errors.
    prices = pd.DataFrame(index=raw.index)

    for instrument in EXPECTED_INSTRUMENTS:

        original = raw[instrument]

        converted = pd.to_numeric(
            original,
            errors="coerce",
        )

        bad_values = original.notna() & converted.isna()

        if bad_values.any():
            rows = bad_values[bad_values].index.tolist()

            raise ValueError(
                f"Non-numeric values in {instrument}, "
                f"spreadsheet rows: {rows[:10]}"
            )

        prices[instrument] = converted

    prices.index = pd.DatetimeIndex(
        dates.to_numpy(),
        name="Date",
    )

    # Sort chronologically. Duplicate dates are deliberately
    # left visible so the audit can report them.
    prices = prices.sort_index()

    return prices


# ------------------------------------------------------------
# Structural data audit
# ------------------------------------------------------------

def audit_price_panel(prices: pd.DataFrame) -> dict:
    """
    Audit the structure and numerical validity of a price panel.

    This does not establish economic correctness, trading activity,
    or completeness against the official exchange calendar.
    """

    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame.")

    if prices.columns.has_duplicates:
        raise ValueError("Duplicate instrument columns found.")

    # Convert safely for numerical checks.
    numeric = prices.apply(
        lambda column: pd.to_numeric(
            column,
            errors="coerce",
        )
    )

    non_numeric_cells = int(
        (prices.notna() & numeric.isna()).sum().sum()
    )

    values = numeric.to_numpy(dtype=float)

    missing_cells = int(numeric.isna().sum().sum())

    infinite_cells = int(
        np.isinf(values).sum()
    )

    nonpositive_cells = int(
        ((numeric <= 0) & numeric.notna()).sum().sum()
    )

    duplicate_dates = int(
        prices.index.duplicated().sum()
    )

    all_missing_rows = int(
        numeric.isna().all(axis=1).sum()
    )

    valid_datetime_index = (
        isinstance(prices.index, pd.DatetimeIndex)
        and not prices.index.hasnans
    )

    chronological_order = (
        prices.index.is_monotonic_increasing
    )

    expected_schema = (
        set(prices.columns) == set(EXPECTED_INSTRUMENTS)
        and len(prices.columns) == len(EXPECTED_INSTRUMENTS)
    )

    missing_by_instrument = {
        instrument: int(numeric[instrument].isna().sum())
        for instrument in numeric.columns
    }

    checks = {
        "expected_instruments": expected_schema,
        "valid_datetime_index": valid_datetime_index,
        "unique_dates": duplicate_dates == 0,
        "chronological_order": chronological_order,
        "no_missing_price_cells": missing_cells == 0,
        "numeric_prices": non_numeric_cells == 0,
        "finite_prices": infinite_cells == 0,
        "positive_prices": nonpositive_cells == 0,
    }

    report = {
        "observations": len(prices),
        "instruments": len(prices.columns),
        "instrument_names": list(prices.columns),

        "start_date": (
            str(prices.index.min().date())
            if len(prices) and valid_datetime_index
            else None
        ),

        "end_date": (
            str(prices.index.max().date())
            if len(prices) and valid_datetime_index
            else None
        ),

        "missing_cells": missing_cells,
        "infinite_cells": infinite_cells,
        "nonpositive_cells": nonpositive_cells,
        "non_numeric_cells": non_numeric_cells,
        "duplicate_dates": duplicate_dates,
        "all_missing_rows": all_missing_rows,

        "missing_by_instrument": missing_by_instrument,

        "checks": checks,

        "passes_basic_structural_audit": all(checks.values()),
    }

    return report


# ------------------------------------------------------------
# Preliminary price-change diagnostics
# ------------------------------------------------------------

def inspect_price_changes(prices: pd.DataFrame) -> dict:
    """
    Calculate log changes between consecutive recorded rows.

    These are preliminary diagnostics, not yet model features.
    The function does not classify large changes as errors.
    """

    if (prices <= 0).any().any():
        raise ValueError(
            "Cannot calculate log changes with non-positive prices."
        )

    log_changes = np.log(prices).diff()

    results = {}

    for instrument in prices.columns:

        series = log_changes[instrument].dropna()

        if series.empty:
            results[instrument] = {
                "valid_changes": 0,
                "exact_zero_changes": 0,
                "largest_abs_log_change": None,
                "date_of_largest_change": None,
                "signed_log_change": None,
            }
            continue

        largest_date = series.abs().idxmax()

        results[instrument] = {
            "valid_changes": int(len(series)),

            "exact_zero_changes": int(
                (series == 0).sum()
            ),

            "largest_abs_log_change": float(
                abs(series.loc[largest_date])
            ),

            "date_of_largest_change": str(
                largest_date.date()
            ),

            "signed_log_change": float(
                series.loc[largest_date]
            ),
        }

    return results


# ------------------------------------------------------------
# Command-line execution
# ------------------------------------------------------------

def main() -> None:

    prices = load_oslo_prices()

    report = audit_price_panel(prices)

    print("=== OSLO PRICE PANEL ===")
    print(f"Shape: {prices.shape}")
    print(f"Instruments: {list(prices.columns)}")

    print("\n=== STRUCTURAL AUDIT ===")

    for key, value in report.items():
        print(f"{key}: {value}")

    if report["passes_basic_structural_audit"]:

        print("\n=== PRELIMINARY PRICE-CHANGE DIAGNOSTICS ===")

        changes = inspect_price_changes(prices)

        for instrument, details in changes.items():
            print(f"{instrument}: {details}")

    else:
        print(
            "\nPrice-change diagnostics skipped: "
            "the structural audit did not pass."
        )



if __name__ == "__main__":
    main()
