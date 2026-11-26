import numpy as np
import pandas as pd
from pathlib import Path
import sys
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
from data import Kaggle_Data_download
from parta_DataModel import Vol_transition_dataset


def split_with_purge(df, train_frac=0.6, val_frac=0.2, purge=5):
    n = len(df)
    i_tr = int(n * train_frac)
    i_va = int(n * (train_frac + val_frac))
    return (
        df.iloc[: i_tr - purge],
        df.iloc[i_tr : i_va - purge],
        df.iloc[i_va :],
    )


def count_episodes(split, q):
    # Keep the original timeline.
    eligible = split["rv_past"] <= q

    # Ineligible dates are False, not removed.
    positive = pd.Series(False, index=split.index)

    positive.loc[eligible] = (
        split.loc[eligible, "rv_future"] > q
    )

    # Count transitions from False to True.
    starts = positive & ~positive.shift(
        1, fill_value=False
    )

    return int(starts.sum())


def feasibility_report(model_data, q_pct=0.80):
    train, val, test = split_with_purge(model_data)
    q = train["rv_past"].quantile(q_pct)
    print(f"Threshold q_{q_pct:.2f} (train only): {q:.6f}")
    print(f"Split sizes after purge: "
          f"train={len(train)}  val={len(val)}  test={len(test)}")
    print()
    header = (f"{'split':8s} {'range':25s} "
              f"{'eligible':>9s} {'pos':>5s} {'episodes':>9s} {'rate':>8s}")
    print(header)
    print("-" * len(header))
    for name, split in [
        ("train", train),
        ("val", val),
        ("test", test)
    ]:
        elig = split[split["rv_past"] <= q]
        y = (elig["rv_future"] > q).astype(int)

        episodes = count_episodes(split, q)
        eligible_pct = 100 * len(elig) / len(split)

        print(
            f"{name:8s} "
            f"eligible={len(elig):5d} "
            f"({eligible_pct:5.1f}%) "
            f"positive={int(y.sum()):4d} "
            f"episodes={episodes:4d} "
            f"rate={y.mean():7.2%}"
        )
    return q


if __name__ == "__main__":
    adj_close_data = Kaggle_Data_download.load_kaggle_panel_with_benchmark("SPY")
    steps, model_data = Vol_transition_dataset(
        adj_close=adj_close_data,
        benchmark="SPY",
        horizon=5,
    )
    feasibility_report(model_data, q_pct=0.80)