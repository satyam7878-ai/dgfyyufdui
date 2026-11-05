from pathlib import Path

import pandas as pd

# Relative path, so it works on any laptop
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "traffic.csv"

FEATURES = ["Junction", "hour", "day_of_week", "month"]


def load_data():
    """Read the CSV file."""
    return pd.read_csv(DATA_PATH, parse_dates=["DateTime"])


def clean_data(df):
    """Remove bad rows and make time columns from DateTime."""
    df = df.copy()
    df = df.dropna(subset=["DateTime", "Junction", "Vehicles"])
    df = df.drop_duplicates(subset=["DateTime", "Junction"])

    df["hour"] = df["DateTime"].dt.hour
    df["day_of_week"] = df["DateTime"].dt.dayofweek
    df["month"] = df["DateTime"].dt.month

    return df.sort_values("DateTime").reset_index(drop=True)


def get_train_test(test_size=0.2):
    """Older 80% of the data trains, newest 20% tests."""
    df = clean_data(load_data())

    split = int(len(df) * (1 - test_size))
    train, test = df.iloc[:split], df.iloc[split:]
    return (train[FEATURES], test[FEATURES],
            train["Vehicles"], test["Vehicles"])


def congestion_labels(X_train, y_train, X_test, y_test, q=0.75):
    """1 = congested (above the junction's limit), 0 = not congested.
    The limit is the q-th percentile of the TRAINING data for each junction."""
    limits = y_train.groupby(X_train["Junction"]).quantile(q)
    fallback = y_train.quantile(q)  # for a junction missing in training data
    train_limit = X_train["Junction"].map(limits).fillna(fallback)
    test_limit = X_test["Junction"].map(limits).fillna(fallback)
    return (y_train > train_limit).astype(int), (y_test > test_limit).astype(int)


if __name__ == "__main__":
    df = clean_data(load_data())
    print("Rows:", len(df))
    print(df.head())
    print(df["Vehicles"].describe())
