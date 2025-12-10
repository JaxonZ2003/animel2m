import pandas as pd
import numpy as np
from pathlib import Path


BASE_DIR = Path("out")

METRIC_COLS = [
    "val_loss",
    "val_acc",
    "val_auc",
    "train_loss",
    "train_acc",
    "train_auc",
    "test_loss",
    "test_acc",
    "test_auc",
]


def load_run_metrics(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)

    row = {}
    # model name
    row["model"] = csv_path.parent.parent.name

    # record which seed/fold this run belongs to (not used but useful for debugging)
    row["run_id"] = csv_path.parents[3].name  # e.g., seed4710_fold1

    for m in METRIC_COLS:
        s = df[m].dropna()
        if not s.empty:
            row[m] = float(s.iloc[-1])
        else:
            row[m] = np.nan

    return row


def main():
    metrics_files = list(BASE_DIR.glob("seed*/logs/*/version_*/metrics.csv"))

    if not metrics_files:
        print("No metrics.csv found under", BASE_DIR)
        return

    records = [load_run_metrics(p) for p in metrics_files]
    runs_df = pd.DataFrame(records)

    # group by model and compute mean and std
    grouped = runs_df.groupby("model")[METRIC_COLS].agg(["mean", "std"])

    # concatenate mean and std into a pretty string
    pretty = pd.DataFrame(index=grouped.index)
    for m in METRIC_COLS:
        mean = grouped[(m, "mean")]
        std = grouped[(m, "std")].fillna(0.0)  # std is 0 for models with only one run
        pretty[m] = mean.round(4).astype(str) + " ± " + std.round(4).astype(str)

    # output as CSV
    csv_str = pretty.to_csv(index_label="model")
    print(csv_str)


if __name__ == "__main__":
    main()
