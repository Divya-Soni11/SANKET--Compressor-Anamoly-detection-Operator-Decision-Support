import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os


def load_data():
    full = pd.read_csv("data/processed/full_year.csv", parse_dates=["Timestamp"])
    scores = pd.read_csv("results/tables/anomaly_scores.csv", parse_dates=["Timestamp"])
    # Merge score and flag into the full-year data
    df = full.merge(scores, on="Timestamp", how="left")
    return df


def plot_anomaly_score(df):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df["Timestamp"], df["anomaly_score"], linewidth=0.5, color="steelblue")

    # Draw a horizontal line at the threshold where anomalies begin.
    # This is the score above which the model says "anomaly".
    threshold = df.loc[df["is_anomaly"], "anomaly_score"].min()
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1,
               label=f"Anomaly threshold = {threshold:.3f}")

    ax.set_ylabel("Anomaly score\n(higher = more anomalous)")
    ax.set_title("Anomaly Score Over the Full Year")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/plots/anomaly_score_over_year.png", dpi=100)
    plt.close()
    print("Saved: results/plots/anomaly_score_over_year.png")


def plot_speed_with_anomalies(df):
    fig, ax = plt.subplots(figsize=(14, 4))

    # Plot speed
    ax.plot(df["Timestamp"], df["speed"], linewidth=0.6, color="steelblue", label="Speed (RPM)")

    # Shade regions where the model flagged an anomaly
    _shade_anomaly_regions(ax, df)

    ax.set_ylabel("Speed (RPM)")
    ax.set_title("Speed with Anomaly Regions Highlighted (red bands)")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/plots/speed_with_anomalies.png", dpi=100)
    plt.close()
    print("Saved: results/plots/speed_with_anomalies.png")


def plot_group_with_anomalies(df, group_name, columns):
    n = len(columns)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for i, col in enumerate(columns):
        axes[i].plot(df["Timestamp"], df[col], linewidth=0.5, color="steelblue")
        _shade_anomaly_regions(axes[i], df)
        axes[i].set_ylabel(col, fontsize=9)
        axes[i].grid(alpha=0.3)

    axes[0].set_title(f"{group_name} with Anomaly Regions Highlighted", fontsize=12)
    axes[-1].set_xlabel("Time")

    plt.tight_layout()
    save_path = f"results/plots/{group_name.lower().replace(' ', '_')}_with_anomalies.png"
    plt.savefig(save_path, dpi=100)
    plt.close()
    print("Saved:", save_path)


def _shade_anomaly_regions(ax, df):
    # Draw red vertical bands where is_anomaly is True.
    # We use fill_between with alpha to make the bands semi-transparent.
    anomaly = df["is_anomaly"].values
    time = df["Timestamp"].values

    # Identify contiguous anomaly regions
    in_region = False
    start = None
    for i in range(len(anomaly)):
        if anomaly[i] and not in_region:
            start = time[i]
            in_region = True
        elif not anomaly[i] and in_region:
            ax.axvspan(start, time[i], color="red", alpha=0.15)
            in_region = False

    # Close the final region if it reaches the end of the data
    if in_region:
        ax.axvspan(start, time[-1], color="red", alpha=0.15)


def print_flagged_periods(df):
    # Print the time ranges during which anomalies were flagged,
    # merging consecutive flagged rows into one period.
    anomaly = df["is_anomaly"].values
    time = df["Timestamp"].values

    periods = []
    in_region = False
    start = None
    for i in range(len(anomaly)):
        if anomaly[i] and not in_region:
            start = time[i]
            in_region = True
        elif not anomaly[i] and in_region:
            periods.append((start, time[i]))
            in_region = False
    if in_region:
        periods.append((start, time[-1]))

    print(f"\nFlagged anomaly periods: {len(periods)}")
    for i, (s, e) in enumerate(periods):
        dur = pd.Timestamp(e) - pd.Timestamp(s)
        print(f"  {i+1}. {pd.Timestamp(s)} to {pd.Timestamp(e)}  ({dur})")


if __name__ == "__main__":
    os.makedirs("results/plots", exist_ok=True)

    df = load_data()

    plot_anomaly_score(df)
    plot_speed_with_anomalies(df)

    plot_group_with_anomalies(df, "Axial", ["axial_1", "axial_2", "axial_3", "axial_4"])
    plot_group_with_anomalies(df, "Vibration", ["vibration_1", "vibration_2", "vibration_3", "vibration_4"])
    plot_group_with_anomalies(df, "Temperature",
                              ["temp_1", "temp_2", "temp_3", "temp_4", "temp_5",
                               "temp_6", "temp_7", "temp_8", "temp_9", "temp_10", "temp_11"])
    plot_group_with_anomalies(df, "Pressure",
                              ["pressure_1", "pressure_2", "pressure_3", "pressure_4", "pressure_diff"])

    print_flagged_periods(df)