import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os


sensor_groups = {
    "axial": ["axial_1", "axial_2", "axial_3", "axial_4"],
    "vibration": ["vibration_1", "vibration_2", "vibration_3", "vibration_4"],
    "temperature": ["temp_1", "temp_2", "temp_3", "temp_4", "temp_5",
                    "temp_6", "temp_7", "temp_8", "temp_9", "temp_10", "temp_11"],
    "pressure": ["pressure_1", "pressure_2", "pressure_3", "pressure_4", "pressure_diff"],
    "speed": ["speed"],
}


def load_all():
    full = pd.read_csv("data/processed/full_year.csv", parse_dates=["Timestamp"])
    scores = pd.read_csv("results/tables/anomaly_scores.csv", parse_dates=["Timestamp"])
    groups_filtered = pd.read_csv("results/tables/groups_filtered.csv",
                                  parse_dates=["start", "end"])
    full = full.merge(scores, on="Timestamp", how="left")
    return full, groups_filtered


def plot_anomaly_score(full_df, filtered_df):
    fig, ax = plt.subplots(figsize=(14, 5))

    # The anomaly score over the whole year
    ax.plot(full_df["Timestamp"], full_df["anomaly_score"],
            linewidth=0.5, color="steelblue", label="Anomaly score")

    # Shade the filtered anomaly periods in red
    for _, row in filtered_df.iterrows():
        ax.axvspan(row["start"], row["end"], color="red", alpha=0.25)

    # Draw the anomaly threshold as a dashed line
    threshold = full_df.loc[full_df["is_anomaly"], "anomaly_score"].min()
    ax.axhline(threshold, color="darkred", linestyle="--", linewidth=1,
               label=f"Threshold = {threshold:.3f}")

    ax.set_ylabel("Anomaly score\n(higher = more anomalous)")
    ax.set_title("Anomaly Score with Filtered Anomaly Periods Highlighted")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/plots/anomaly_score_filtered.png", dpi=100)
    plt.close()
    print("Saved: results/plots/anomaly_score_filtered.png")


def plot_group_heatmap(filtered_df):
    # Build a matrix: rows = periods, columns = groups
    group_names = list(sensor_groups.keys())
    matrix = filtered_df[[f"group_{g}" for g in group_names]].values

    # Clip extreme z-scores to make the heatmap readable
    # (otherwise -80 sigma dominates the colour scale)
    matrix_clipped = np.clip(matrix, -10, 10)

    fig, ax = plt.subplots(figsize=(8, 0.35 * len(filtered_df) + 3))

    im = ax.imshow(matrix_clipped, aspect="auto", cmap="RdBu_r",
                   vmin=-10, vmax=10)

    ax.set_xticks(range(len(group_names)))
    ax.set_xticklabels(group_names, rotation=30, ha="right")

    # Label rows with period ID and date
    ylabels = []
    for _, row in filtered_df.iterrows():
        ylabels.append(f"P{int(row['period_id'])}: {row['start'].strftime('%b %d')}")
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=8)

    plt.colorbar(im, ax=ax, label="Group z-score (clipped to ±10)")
    ax.set_title("Group z-Scores per Anomaly Period")
    plt.tight_layout()
    plt.savefig("results/plots/group_heatmap.png", dpi=100)
    plt.close()
    print("Saved: results/plots/group_heatmap.png")


def plot_single_period(full_df, start, end, sensor_cols, title, filename):
    # Plot a set of sensors over a window around [start, end].
    # Shade the anomaly period itself.
    window_start = start - pd.Timedelta("5 days")
    window_end = end + pd.Timedelta("5 days")
    mask = (full_df["Timestamp"] >= window_start) & (full_df["Timestamp"] <= window_end)
    df_window = full_df[mask]

    n = len(sensor_cols)
    fig, axes = plt.subplots(n, 1, figsize=(13, 2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for i, col in enumerate(sensor_cols):
        axes[i].plot(df_window["Timestamp"], df_window[col],
                     linewidth=0.7, color="steelblue")
        axes[i].axvspan(start, end, color="red", alpha=0.2)
        axes[i].set_ylabel(col, fontsize=9)
        axes[i].grid(alpha=0.3)

    axes[0].set_title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(f"results/plots/{filename}", dpi=100)
    plt.close()
    print(f"Saved: results/plots/{filename}")


if __name__ == "__main__":
    os.makedirs("results/plots", exist_ok=True)

    full_df, filtered_df = load_all()

    # Figure 1 — overall anomaly score with filtered periods
    plot_anomaly_score(full_df, filtered_df)

    # Figure 2 — group z-score heatmap
    plot_group_heatmap(filtered_df)

    # Figure 3 — axial displacement around the June pre-shutdown drift
    # Period 15 in the filtered list: 2022-06-17 21:15 to 22:15
    period_15_start = pd.Timestamp("2022-06-17 21:15:00")
    period_15_end = pd.Timestamp("2022-06-17 22:15:00")
    plot_single_period(
        full_df,
        period_15_start,
        period_15_end,
        ["axial_1", "axial_2", "axial_3", "axial_4"],
        "Axial Displacement Around Pre-Shutdown Drift (June 17)",
        "axial_pre_shutdown.png",
    )

    # Figure 4 — pressure around the April 28 event
    # Period 10: 2022-04-28 10:45 to 12:00
    period_10_start = pd.Timestamp("2022-04-28 10:45:00")
    period_10_end = pd.Timestamp("2022-04-28 12:00:00")
    plot_single_period(
        full_df,
        period_10_start,
        period_10_end,
        ["pressure_1", "pressure_2", "pressure_3", "pressure_4", "pressure_diff"],
        "Pressure Around April 28 Event",
        "pressure_april_event.png",
    )