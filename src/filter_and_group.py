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

# Physical groups — used to collapse 25 sensors into 5 meaningful summaries
sensor_groups = {
    "axial": ["axial_1", "axial_2", "axial_3", "axial_4"],
    "vibration": ["vibration_1", "vibration_2", "vibration_3", "vibration_4"],
    "temperature": ["temp_1", "temp_2", "temp_3", "temp_4", "temp_5",
                    "temp_6", "temp_7", "temp_8", "temp_9", "temp_10", "temp_11"],
    "pressure": ["pressure_1", "pressure_2", "pressure_3", "pressure_4", "pressure_diff"],
    "speed": ["speed"],
}


def load_all():
    train = pd.read_csv("data/processed/train_steady.csv", parse_dates=["Timestamp"])
    full = pd.read_csv("data/processed/full_year.csv", parse_dates=["Timestamp"])
    scores = pd.read_csv("results/tables/anomaly_scores.csv", parse_dates=["Timestamp"])
    full = full.merge(scores, on="Timestamp", how="left")
    return train, full


def normal_stats(train_df):
    means = train_df[sensor_columns].mean()
    stds = train_df[sensor_columns].std()
    return means, stds


def find_periods(df, min_hours=0.0):
    # Find contiguous runs of flagged rows.
    # If min_hours > 0, only keep periods whose duration is at least min_hours.
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

    if min_hours > 0:
        kept = []
        for (s, e) in periods:
            duration_hours = (time[e] - time[s]) / np.timedelta64(1, "h")
            if duration_hours >= min_hours:
                kept.append((s, e))
        periods = kept

    return periods


def period_z_scores(df, periods, means, stds):
    # For each period, compute the average z-score of every sensor.
    rows = []
    for i, (s, e) in enumerate(periods):
        period_df = df.iloc[s:e + 1]
        avg_values = period_df[sensor_columns].mean()
        z_scores = (avg_values - means) / stds

        row = {
            "period_id": i + 1,
            "start": period_df["Timestamp"].iloc[0],
            "end": period_df["Timestamp"].iloc[-1],
            "duration_hours": (period_df["Timestamp"].iloc[-1]
                               - period_df["Timestamp"].iloc[0]).total_seconds() / 3600,
        }

        # Add group-level average z-scores
        for group_name, group_cols in sensor_groups.items():
            row[f"group_{group_name}"] = z_scores[group_cols].mean()

        rows.append(row)

    return pd.DataFrame(rows)


def top_contributors(df, periods, means, stds, top_n=5):
    # Same as before: top N individual sensors per period.
    rows = []
    for i, (s, e) in enumerate(periods):
        period_df = df.iloc[s:e + 1]
        avg_values = period_df[sensor_columns].mean()
        z_scores = (avg_values - means) / stds
        top = z_scores.abs().sort_values(ascending=False).head(top_n)

        row = {"period_id": i + 1}
        for rank, sensor in enumerate(top.index, start=1):
            row[f"top_{rank}"] = sensor
            row[f"top_{rank}_z"] = z_scores[sensor]
        rows.append(row)

    return pd.DataFrame(rows)


def save_outputs(group_df, contrib_df, tag):
    os.makedirs("results/tables", exist_ok=True)
    group_df.to_csv(f"results/tables/groups_{tag}.csv", index=False)
    contrib_df.to_csv(f"results/tables/contributors_{tag}.csv", index=False)
    print(f"Saved: results/tables/groups_{tag}.csv")
    print(f"Saved: results/tables/contributors_{tag}.csv")


def print_summary(group_df, contrib_df, label):
    print(f"\n{'=' * 70}")
    print(f"{label}")
    print(f"{'=' * 70}")
    print(f"Number of periods: {len(group_df)}")
    print()
    for _, row in group_df.iterrows():
        print(f"Period {row['period_id']}: {row['start']} to {row['end']} "
              f"({row['duration_hours']:.1f} h)")
        print(f"  Group z-scores:")
        print(f"    axial       = {row['group_axial']:+.2f}")
        print(f"    vibration   = {row['group_vibration']:+.2f}")
        print(f"    temperature = {row['group_temperature']:+.2f}")
        print(f"    pressure    = {row['group_pressure']:+.2f}")
        print(f"    speed       = {row['group_speed']:+.2f}")

        # Show top 3 individual sensors for context
        c_row = contrib_df[contrib_df["period_id"] == row["period_id"]].iloc[0]
        print(f"  Top 3 individual sensors:")
        for k in [1, 2, 3]:
            print(f"    {c_row[f'top_{k}']:15s} z = {c_row[f'top_{k}_z']:+.2f}")
        print()


if __name__ == "__main__":
    train_df, full_df = load_all()
    means, stds = normal_stats(train_df)

    # ---- RAW: all 60 periods ----
    raw_periods = find_periods(full_df, min_hours=0.0)
    raw_group_df = period_z_scores(full_df, raw_periods, means, stds)
    raw_contrib_df = top_contributors(full_df, raw_periods, means, stds)
    save_outputs(raw_group_df, raw_contrib_df, tag="raw")
    print_summary(raw_group_df, raw_contrib_df, "RAW — all flagged periods")

    # ---- FILTERED: only periods lasting at least 1 hour ----
    filtered_periods = find_periods(full_df, min_hours=1.0)
    filt_group_df = period_z_scores(full_df, filtered_periods, means, stds)
    filt_contrib_df = top_contributors(full_df, filtered_periods, means, stds)
    save_outputs(filt_group_df, filt_contrib_df, tag="filtered")
    print_summary(filt_group_df, filt_contrib_df, "FILTERED — only periods >= 1 hour")