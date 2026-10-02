import pandas as pd
import matplotlib.pyplot as plt
import os


def plot_group_trends(df, group_name, columns, color_list=None):
    # One figure with one subplot per variable in the group.
    # x-axis is time, y-axis is the sensor value.
    n = len(columns)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for i, col in enumerate(columns):
        color = color_list[i] if color_list else "steelblue"
        axes[i].plot(df["Timestamp"], df[col], linewidth=0.6, color=color)
        axes[i].set_ylabel(col, fontsize=9)
        axes[i].grid(alpha=0.3)

    axes[0].set_title(f"{group_name} — Full Year Trend", fontsize=12)
    axes[-1].set_xlabel("Time")
    plt.tight_layout()

    save_path = f"results/plots/{group_name.lower().replace(' ', '_')}_trend.png"
    plt.savefig(save_path, dpi=100)
    plt.close()
    print("Saved:", save_path)


def plot_distributions(df, columns, group_name):
    # One histogram per variable — shows the shape of the normal data.
    n = len(columns)
    cols_per_row = 4
    rows = (n + cols_per_row - 1) // cols_per_row

    fig, axes = plt.subplots(rows, cols_per_row, figsize=(4 * cols_per_row, 3 * rows))
    axes = axes.flatten()

    for i, col in enumerate(columns):
        axes[i].hist(df[col], bins=50, color="steelblue", edgecolor="black", linewidth=0.3)
        axes[i].set_title(col, fontsize=9)
        axes[i].grid(alpha=0.3)

    # Hide unused subplots
    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"{group_name} — Distributions", fontsize=12)
    plt.tight_layout()

    save_path = f"results/plots/{group_name.lower().replace(' ', '_')}_distributions.png"
    plt.savefig(save_path, dpi=100)
    plt.close()
    print("Saved:", save_path)


def plot_correlation(df, columns):
    # Correlation matrix of all sensor variables.
    corr = df[columns].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

    ax.set_xticks(range(len(columns)))
    ax.set_yticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=90, fontsize=7)
    ax.set_yticklabels(columns, fontsize=7)

    plt.colorbar(im, ax=ax, label="Correlation")
    ax.set_title("Sensor Correlation Matrix — Full Year", fontsize=12)
    plt.tight_layout()

    save_path = "results/plots/correlation_matrix.png"
    plt.savefig(save_path, dpi=100)
    plt.close()
    print("Saved:", save_path)


if __name__ == "__main__":
    # Make sure the results folder exists
    os.makedirs("results/plots", exist_ok=True)

    df = pd.read_csv("data/processed/compressor_clean.csv", parse_dates=["Timestamp"])

    # Group the sensors by physical type
    axial_cols = ["axial_1", "axial_2", "axial_3", "axial_4"]
    vibration_cols = ["vibration_1", "vibration_2", "vibration_3", "vibration_4"]
    temp_cols = ["temp_1", "temp_2", "temp_3", "temp_4", "temp_5",
                 "temp_6", "temp_7", "temp_8", "temp_9", "temp_10", "temp_11"]
    pressure_cols = ["pressure_1", "pressure_2", "pressure_3", "pressure_4", "pressure_diff"]
    speed_col = ["speed"]

    # Trends over the full year
    plot_group_trends(df, "Axial Displacement", axial_cols)
    plot_group_trends(df, "Vibration", vibration_cols)
    plot_group_trends(df, "Temperature", temp_cols)
    plot_group_trends(df, "Pressure", pressure_cols)
    plot_group_trends(df, "Speed", speed_col)

    # Distributions
    plot_distributions(df, axial_cols, "Axial Displacement")
    plot_distributions(df, vibration_cols, "Vibration")
    plot_distributions(df, temp_cols, "Temperature")
    plot_distributions(df, pressure_cols, "Pressure")

    # Correlation matrix of all 25 sensors
    all_sensors = axial_cols + vibration_cols + temp_cols + pressure_cols + speed_col
    plot_correlation(df, all_sensors)

    print("\nAll plots saved.")