from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save plots without opening a window
import matplotlib.pyplot as plt

from data import load_data, clean_data

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def run():
    """Save three exploration plots in the outputs folder."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = clean_data(load_data())

    # 1. Average vehicles by hour, one line per junction
    plt.figure(figsize=(8, 5))
    for junction, group in df.groupby("Junction"):
        group.groupby("hour")["Vehicles"].mean().plot(label=f"Junction {junction}")
    plt.title("Average vehicles by hour of day")
    plt.xlabel("Hour")
    plt.ylabel("Average vehicles")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda_vehicles_by_hour.png")
    plt.close()

    # 2. Average vehicles by day of week (0 = Monday)
    plt.figure(figsize=(8, 5))
    df.groupby("day_of_week")["Vehicles"].mean().plot(kind="bar")
    plt.title("Average vehicles by day of week (0 = Monday)")
    plt.xlabel("Day of week")
    plt.ylabel("Average vehicles")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda_vehicles_by_day.png")
    plt.close()

    # 3. Average vehicles by junction
    plt.figure(figsize=(8, 5))
    df.groupby("Junction")["Vehicles"].mean().plot(kind="bar")
    plt.title("Average vehicles by junction")
    plt.xlabel("Junction")
    plt.ylabel("Average vehicles")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda_vehicles_by_junction.png")
    plt.close()

    print("EDA plots saved in outputs/")


if __name__ == "__main__":
    run()
