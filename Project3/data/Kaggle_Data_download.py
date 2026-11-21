# Kaggle_Data_download.py
from pathlib import Path
import pandas as pd
import yfinance as yf
import kagglehub

DATASET = "darkmatternet/s-and-p-500-stocks-25-years-of-data-updated-daily"


def dataset_dir() -> Path:
    """Download (cached) and return the local directory of the dataset."""
    return Path(kagglehub.dataset_download(DATASET))


def inspect_dataset() -> None:
    """Print the CSV files and their column headers. Run this once."""
    root = dataset_dir()
    print("Dataset directory:", root)
    for f in sorted(root.rglob("*.csv")):
        df = pd.read_csv(f, nrows=3)
        print(f"\n{f.name}")
        print("  columns:", list(df.columns))
        print(df.head(3).to_string())


def _find(colmap: dict, *candidates: str):
    """Return the first original column name matching any lowercase candidate."""
    for c in candidates:
        if c in colmap:
            return colmap[c]
    return None


def _load_long_csv(csv_path: Path) -> pd.DataFrame:
    """Read one CSV and return a long frame [date, ticker, adj_close]."""
    df = pd.read_csv(csv_path)
    cmap = {c.strip().lower(): c for c in df.columns}

    date_col = _find(cmap, "date", "datetime", "timestamp")
    value_col = _find(cmap, "adj close", "adj_close", "adjclose",
                      "adjusted close", "close")
    tick_col = _find(cmap, "symbol", "ticker", "ticker symbol")

    if date_col is None or value_col is None:
        return pd.DataFrame(columns=["date", "ticker", "adj_close"])

    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col]),
        "adj_close": pd.to_numeric(df[value_col], errors="coerce"),
    })
    # If the file has no ticker column, use the filename as the ticker.
    out["ticker"] = df[tick_col].astype(str) if tick_col else csv_path.stem
    return out.dropna(subset=["date", "adj_close"])


def get_data() -> pd.DataFrame:
    """
    Return a wide DataFrame of adjusted close prices:
    index = trading date (DatetimeIndex, sorted, unique)
    columns = tickers
    """
    root = dataset_dir()
    csv_files = sorted(root.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {root}")

    long_parts = [_load_long_csv(f) for f in csv_files]
    long = pd.concat([p for p in long_parts if not p.empty], ignore_index=True)
    if long.empty:
        raise ValueError(
            "Could not find date/price columns in any CSV. "
            "Run inspect_dataset() and adapt _load_long_csv()."
        )

    wide = (
        long.pivot_table(index="date", columns="ticker",
                         values="adj_close", aggfunc="last")
            .sort_index()
    )
    wide.columns.name = None
    return wide

def load_kaggle_panel_with_benchmark(benchmark="SPY", start="2000-01-01"):
    """
    Return the Kaggle constituent panel with `benchmark` appended as a column.
    Kaggle stocks come from the local snapshot; SPY is fetched from Yahoo.
    """
    panel = get_data()                        # existing loader
    if not isinstance(panel, pd.DataFrame):
        raise TypeError(f"get_data() returned {type(panel)}, expected DataFrame")
    panel = panel.sort_index()
    panel.index = pd.to_datetime(panel.index)

    spy = yf.download(
        benchmark,
        start=panel.index.min(),
        end=panel.index.max() + pd.Timedelta(days=1),
        auto_adjust=False,
        progress=False,
    )
    if spy.empty:
        raise RuntimeError(f"Yahoo returned no rows for {benchmark}")

    # yfinance returns MultiIndex columns when start/end are both set
    if isinstance(spy.columns, pd.MultiIndex):
        spy_adj = spy["Adj Close"][benchmark]
    else:
        spy_adj = spy["Adj Close"]

    spy_adj = spy_adj.rename(benchmark)
    spy_adj.index = pd.to_datetime(spy_adj.index)

    combined = panel.join(spy_adj, how="left")
    return combined