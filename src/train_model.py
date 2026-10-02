import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import os


sensor_columns = [
    "axial_1", "axial_2", "axial_3", "axial_4",
    "pressure_1", "pressure_2", "pressure_3", "pressure_4", "pressure_diff",
    "temp_1", "temp_2", "temp_3", "temp_4", "temp_5",
    "temp_6", "temp_7", "temp_8", "temp_9", "temp_10", "temp_11",
    "vibration_1", "vibration_2", "vibration_3", "vibration_4",
    "speed",
]


def load_training_data(path):
    df = pd.read_csv(path, parse_dates=["Timestamp"])
    return df


def train_model(train_df):
    # Extract just the sensor columns. No Timestamp, no is_steady.
    X = train_df[sensor_columns].values

    # Build the Isolation Forest.
    # n_estimators = number of random trees
    # contamination = expected fraction of anomalies in training data
    # random_state = makes the result reproducible
    model = IsolationForest(
        n_estimators=200,
        contamination=0.01,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X)
    print("Model trained on", X.shape[0], "rows and", X.shape[1], "features.")
    return model


def score_data(model, df):
    # Compute anomaly scores for every row of the year.
    # score_samples gives negative values: lower (more negative) = more anomalous.
    # We negate so that higher score = more anomalous, which is easier to read.
    X = df[sensor_columns].values
    raw_scores = model.score_samples(X)
    df = df.copy()
    df["anomaly_score"] = -raw_scores

    # Also store the model's binary flag (-1 = anomaly, +1 = normal)
    df["is_anomaly"] = model.predict(X) == -1

    return df


def save_results(df):
    os.makedirs("results/tables", exist_ok=True)
    # Save only Timestamp, anomaly_score, and is_anomaly for review.
    # Keeping the sensor values would double the file size.
    out = df[["Timestamp", "anomaly_score", "is_anomaly"]]
    out.to_csv("results/tables/anomaly_scores.csv", index=False)
    print("Saved anomaly scores to results/tables/anomaly_scores.csv")


def print_summary(df):
    print("\nAnomaly score statistics:")
    print(df["anomaly_score"].describe())

    n_anomalies = df["is_anomaly"].sum()
    print(f"\nRows flagged as anomalies: {n_anomalies} ({n_anomalies / len(df) * 100:.2f}%)")

    # Show the timestamp distribution of flagged anomalies
    flagged = df[df["is_anomaly"]]
    if len(flagged) > 0:
        print("\nDate range of flagged anomalies:")
        print("First flagged:", flagged["Timestamp"].min())
        print("Last flagged: ", flagged["Timestamp"].max())


if __name__ == "__main__":
    train_df = load_training_data("data/processed/train_steady.csv")
    full_df = load_training_data("data/processed/full_year.csv")

    model = train_model(train_df)

    full_df = score_data(model, full_df)
    save_results(full_df)
    print_summary(full_df)