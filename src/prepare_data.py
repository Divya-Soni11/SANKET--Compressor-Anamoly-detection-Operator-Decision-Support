import pandas as pd
import numpy as np
import os


# The 25 sensor columns we will use as features
sensor_columns = [
    "axial_1", "axial_2", "axial_3", "axial_4",
    "pressure_1", "pressure_2", "pressure_3", "pressure_4", "pressure_diff",
    "temp_1", "temp_2", "temp_3", "temp_4", "temp_5",
    "temp_6", "temp_7", "temp_8", "temp_9", "temp_10", "temp_11",
    "vibration_1", "vibration_2", "vibration_3", "vibration_4",
    "speed",
]


def load_clean_data(path):
    df = pd.read_csv(path, parse_dates=["Timestamp"])
    return df


def check_speed_distribution(df):
    # Look at how speed is distributed across the year.
    # This tells us which speed range means "running".
    print("Speed statistics:")
    print(df["speed"].describe())
    print("\nSpeed percentiles:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"  {p}th percentile: {df['speed'].quantile(p / 100):.1f}")


def mark_steady_state(df):
    # A row is "steady state" if the compressor is running normally.
    # We use speed as the indicator:
    #   - speed near 0  -> stopped
    #   - speed ~6000   -> running normally
    #   - speed ~7500   -> brief overspeed during restart
    speed_low = 5500
    speed_high = 7000

    df["is_steady"] = (df["speed"] >= speed_low) & (df["speed"] <= speed_high)
    return df


def summarize_regimes(df):
    total = len(df)
    steady = df["is_steady"].sum()
    print(f"\nTotal rows: {total}")
    print(f"Steady-state rows: {steady} ({steady / total * 100:.1f}%)")
    print(f"Non-steady rows:   {total - steady} ({(total - steady) / total * 100:.1f}%)")


def split_train_test(df, min_block_hours=6):
    # Identify contiguous steady-state blocks.
    # A block is a run of consecutive rows where is_steady is True.
    steady = df[df["is_steady"]].copy()

    # New block starts whenever the time gap is larger than 20 minutes
    # (which means there was a non-steady row in between).
    steady["new_block"] = steady["Timestamp"].diff() > pd.Timedelta("20 minutes")
    steady["block_id"] = steady["new_block"].cumsum()

    # Compute duration of each block
    block_info = steady.groupby("block_id").agg(
        start=("Timestamp", "first"),
        end=("Timestamp", "last"),
    )
    block_info["hours"] = (block_info["end"] - block_info["start"]).dt.total_seconds() / 3600

    # Keep only blocks longer than the threshold
    long_blocks = block_info[block_info["hours"] >= min_block_hours].index
    train_data = steady[steady["block_id"].isin(long_blocks)].copy()

    # Drop the helper columns — the model does not need them
    train_data = train_data.drop(columns=["new_block", "block_id"])

    print(f"\nBlocks kept for training: {len(long_blocks)}")
    print(f"Training rows: {len(train_data)}")
    print(f"Training rows as % of year: {len(train_data) / len(df) * 100:.1f}%")

    # Full year is the test set (the model will see everything)
    test_data = df.copy()

    return train_data, test_data


def save_splits(train_data, test_data):
    os.makedirs("data/processed", exist_ok=True)
    train_data.to_csv("data/processed/train_steady.csv", index=False)
    test_data.to_csv("data/processed/full_year.csv", index=False)
    print("\nSaved training set:", "data/processed/train_steady.csv")
    print("Saved full year:  ", "data/processed/full_year.csv")


if __name__ == "__main__":
    df = load_clean_data("data/processed/compressor_clean.csv")

    # Step 1: Understand the speed distribution
    check_speed_distribution(df)

    # Step 2: Mark steady-state rows
    df = mark_steady_state(df)
    summarize_regimes(df)

    # Step 3: Split
    train_data, test_data = split_train_test(df)

    # Step 4: Save
    save_splits(train_data, test_data)

    # Step 5: Print the date ranges of the steady-state periods
    # This is important — we want to see when the model's training data comes from.
    print("\nSteady-state periods (contiguous blocks):")
    steady = df[df["is_steady"]].copy()
    steady["gap"] = steady["Timestamp"].diff() > pd.Timedelta("20 minutes")
    steady["block"] = steady["gap"].cumsum()

    for block_id, block in steady.groupby("block"):
        start = block["Timestamp"].iloc[0]
        end = block["Timestamp"].iloc[-1]
        hours = (end - start).total_seconds() / 3600
        print(f"  Block {block_id}: {start} to {end}  ({hours:.1f} hours, {len(block)} rows)")