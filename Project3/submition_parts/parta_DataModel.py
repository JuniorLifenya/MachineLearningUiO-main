import numpy as np
from pathlib import Path
import sys
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
import matplotlib.pyplot as plt
import pandas as pd 
import yfinance as yf 
from data import Kaggle_Data_download  as Kaggle_Data_download

# ===================================================================================================================
#                       Data Production: Transition-Label and Feasibility
# ===================================================================================================================

def realized_vol(returns,horizon= 5):
    """
       Calculate non-annualized realized volatility
       over a rolling horizon of daily returns.
       Conventional five-session realized volatility:
       sqrt(sum(r_t^2)) where r_t is the return of day t.
    """
    if horizon < 1:
        raise ValueError("Horizon must be positive.")

    squared_returns = returns ** 2
    rolling_variance = squared_returns.rolling(window=horizon,
                                               min_periods=horizon).sum()
    # Recall that np.sum() calculates a SINGLE sum,
    # while rolling.sum() calculates a sum for each rolling window.
    # Square first then take the square root to get realized volatility.
    # Also rolling.sum() will return NaN for the first (horizon-1) entries, which is expected.
    # It also needs pandas Series or DataFrame as input on which to move a window, not a numpy array.

    return np.sqrt(rolling_variance)

# --------------------------------- The main function to produce the transition dataset ---------------------------------

def Vol_transition_dataset(adj_close, benchmark = "SPY", horizon = 5):
    """"
        adj_close: DataFrame, one column per instrument, including benchmark.
        Returns a DataFrame with realized volatility
        including past,future and returns for each instrument and the benchmark.
        Raises if benchmark is missing - no silent fallback. 
    """
    if not isinstance(adj_close, pd.DataFrame):
        raise TypeError(
            f"adj_close must be a DataFrame, got {type(adj_close).__name__}. "
            f"If you meant to call a loader, add parentheses: loader()."
        )
    adj_close = adj_close.sort_index()                 # ← parentheses
    if benchmark not in adj_close.columns:
        raise ValueError(
            f"Benchmark {benchmark!r} not in columns. "
            f"Available: {list(adj_close.columns)[:5]} ... "
            f"({len(adj_close.columns)} total)"
        )

    # 1. Extract benchmark price series
    # Ensure it's float for calculations
    price = adj_close[benchmark].sort_index().astype(float) 

    if (price.dropna() <= 0).any():
         raise ValueError("Prices must be positive.")

    # 2. Calculate daily log returns for all instruments, including the benchmark.
    # No dropna() here, 
    # we want to keep NaNs for the first row after diff()
    # Missing observations will be handled explicitely afterwards.
    log_ret = np.log(adj_close).diff() 
    
    # 3. Calculate squared returns for all instruments, including the benchmark.
    squared_ret = log_ret.pow(2)

    # 4. Calculate rolling realized volatility for all instruments, including the benchmark.
    rv = realized_vol(log_ret, horizon=horizon)

    # 5. Align past and future volatility with date t.
    rv_past = rv.copy()
    rv_future = rv.shift(-horizon)  # Shift back by horizon to align with date t.

    # 6. Collect every intermediate variable
    steps = pd.DataFrame({
    "price":       price,
    "ret":         log_ret[benchmark],
    "squared_ret": squared_ret[benchmark],
    "rv":          rv[benchmark],
    "rv_past":     rv_past[benchmark],
    "rv_future":   rv_future[benchmark],
    })

    # 7. Keep observations where all required quantities exists
    model_data = steps.dropna(
        subset = ["price", "ret", "squared_ret", "rv", "rv_past", "rv_future"]
    ).copy()

    return steps, model_data
    

# --------------------------------- Plot every transition ---------------------------------
BENCHMARK = "SPY"
HORIZON = 5

if __name__ == "__main__":

    # 1. Acquire the raw data
    adj_close_data = Kaggle_Data_download.get_data()
    print("Raw data shape:", adj_close_data.shape)
    print("Available instruments:", adj_close_data.columns.tolist())

    # 2. Construct the volatility dataset
    steps, model_data = Vol_transition_dataset(
        adj_close=adj_close_data,
        benchmark=BENCHMARK,
        horizon=HORIZON
    )

    print("Intermediate data:", steps.shape)
    print("Complete observations:", model_data.shape)


    # 3. Construct the figure plot
    fig, axes = plt.subplots(
        5,1,figsize = (14,15),
        sharex = True
    )

    # 4. Plot the individual transformations
    axes[0].plot(steps.index, steps["price"])
    axes[0].set_title("1. Adjusted closing price")
    axes[0].set_ylabel("Price")

    axes[1].plot(steps.index, steps["ret"], linewidth=0.8)
    axes[1].set_title("2. Daily logarithmic returns")
    axes[1].set_ylabel("Log return")

    axes[2].plot(steps.index, steps["squared_ret"], linewidth=0.8)
    axes[2].set_title("3. Squared daily returns")
    axes[2].set_ylabel("Squared return")

    axes[3].plot(steps.index, steps["rv_past"])
    axes[3].set_title(
        f"4. Historical {HORIZON}-session realized volatility"
    )
    axes[3].set_ylabel("Realized volatility")

    axes[4].plot(
        steps.index,
        steps["rv_past"],
        label="Past volatility"
    )

    axes[4].plot(
        steps.index,
        steps["rv_future"],
        label="Future volatility",
        alpha=0.8
    )

    axes[4].set_title("5. Past versus future volatility")
    axes[4].set_ylabel("Realized volatility")
    axes[4].legend()

    for ax in axes:
            ax.grid(alpha=0.25)
    
    plt.tight_layout()
    plt.show()



