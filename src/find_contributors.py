import pandas as pd
import numpy as np
import os


sensor_columns = [
    "axial_1", "axial_2", "axial_3", "axial_4",
    "pressure_1", "pressure_2", "pressure_3", "pressure_4", "pressure_diff",
    "temp_1", "temp_2", "temp_3", "temp_4", "temp_5",
    "temp_6", "temp_7", "temp_8", "temp_9", "temp_10", "temp_11",
    "vibration_1", "vibration_2", "vibration_3", "vibration_4",
    "speed",
]


def load_data():
    train = pd.read_csv("data/processed/train_steady.csv", parse_dates=["Timestamp"])
    full = pd.read_csv("data/processed/full_year.csv", parse_dates=["Timestamp"])
    scores = pd.read_csv("results/tables/anomaly_scores.csv", parse_dates=["Timestamp"])
    full = full.merge(scores, on="Timestamp", how="left")
    return train, full


def compute_normal_stats(train_df):
    # For each sensor, compute the mean and standard deviation during steady-state operation.
    # These numbers define "normal" for every sensor.
    means = train_df[sensor_columns].mean()
    stds = train_df[sensor_columns].std()
    return means, stds


def find_anomaly_periods(df):
    # Merge consecutive flagged rows into single anomaly periods.
    anomaly = df["is_anomaly"].values
    time = df["Timestamp"].values

    periods = []
    in_region = False
    start = None
    for i in range(len(anomaly)):
        if anomaly[i] and not in_region:
            start = i
            in_region = True
        elif not anomaly[i] and in_region:
            periods.append((start, i - 1))
            in_region = False
    if in_region:
        periods.append((start, len(anomaly) - 1))

    return periods


def compute_period_z_scores(df, periods, means, stds):
    # For each anomaly period, compute the average z-score of each sensor
    # during that period. Rank the sensors by absolute z-score.
    rows = []
    for i, (start_idx, end_idx) in enumerate(periods):
        period_df = df.iloc[start_idx:end_idx + 1]
        avg_values = period_df[sensor_columns].mean()
        z_scores = (avg_values - means) / stds
        top_contributors = z_scores.abs().sort_values(ascending=False).head(5)
        rows.append({
            "period_id": i + 1,
            "start": period_df["Timestamp"].iloc[0],
            "end": period_df["Timestamp"].iloc[-1],
            "duration_hours": (period_df["Timestamp"].iloc[-1] - period_df["Timestamp"].iloc[0]).total_seconds() / 3600,
            "top_1": top_contributors.index[0],
            "top_1_z": z_scores[top_contributors.index[0]],
            "top_2": top_contributors.index[1],
            "top_2_z": z_scores[top_contributors.index[1]],
            "top_3": top_contributors.index[2],
            "top_3_z": z_scores[top_contributors.index[2]],
            "top_4": top_contributors.index[3],
            "top_4_z": z_scores[top_contributors.index[3]],
            "top_5": top_contributors.index[4],
            "top_5_z": z_scores[top_contributors.index[4]],
        })
    return pd.DataFrame(rows)


def save_contributor_table(result_df):
    os.makedirs("results/tables", exist_ok=True)
    result_df.to_csv("results/tables/contributors.csv", index=False)
    print("Saved contributors to results/tables/contributors.csv")


def print_top_contributors(result_df, top_n=10):
    # Print the top N longest anomaly periods and their contributors.
    result_df = result_df.sort_values("duration_hours", ascending=False).head(top_n)
    print(f"\nTop {top_n} longest anomaly periods and their leading contributors:\n")
    for _, row in result_df.iterrows():
        print(f"Period {row['period_id']}: {row['start']} to {row['end']} ({row['duration_hours']:.1f} h)")
        print(f"  Top 5 contributors:")
        for k in [1, 2, 3, 4, 5]:
            print(f"    {row[f'top_{k}']:20s} z-score = {row[f'top_{k}_z']:+.2f}")
        print()


if __name__ == "__main__":
    train_df, full_df = load_data()

    # Step 1: Get normal statistics from training data
    means, stds = compute_normal_stats(train_df)
    print("Normal means (first 5 sensors):")
    print(means.head())

    # Step 2: Find contiguous anomaly periods
    periods = find_anomaly_periods(full_df)
    print(f"\nTotal anomaly periods: {len(periods)}")

    # Step 3: Compute z-scores per period, rank contributors
    result_df = compute_period_z_scores(full_df, periods, means, stds)

    # Step 4: Save and print
    save_contributor_table(result_df)
    print_top_contributors(result_df, top_n=10)