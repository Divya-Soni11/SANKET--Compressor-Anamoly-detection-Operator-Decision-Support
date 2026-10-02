import pandas as pd
import numpy as np
import os


def classify_period(row, prev_row=None):
    axial = row["group_axial"]
    vibration = row["group_vibration"]
    temperature = row["group_temperature"]
    pressure = row["group_pressure"]
    speed = row["group_speed"]

    # Rule 1: Shutdown — speed has dropped dramatically
    if speed < -10:
        return "Shutdown"

    # Rule 2: Restart transient — must follow a shutdown within the last 24 hours
    if prev_row is not None:
        hours_since_prev = (row["start"] - prev_row["end"]).total_seconds() / 3600
        if prev_row["category_placeholder"] == "Shutdown" and hours_since_prev < 24:
            return "Restart transient"

    # Rule 3: Possible thrust bearing wear — requires strong axial signal
    # and the machine to be running at normal or near-normal speed.
    if axial > 3.0 and speed > -2.0:
        return "Possible thrust bearing wear"

    # Rule 4: Pressure system event
    if pressure < -3:
        return "Pressure system event"

    # Rule 5: Thermal event — temperatures up, everything else normal
    if temperature > 2 and abs(axial) < 1.0 and abs(vibration) < 1.5:
        return "Thermal event"

    # Rule 6: Cold start / low load
    if temperature < -1.5 and speed < -2:
        return "Cold start / low load"

    return "Unclassified"

def priority_for(category):
    priorities = {
        "Shutdown": "Low",
        "Restart transient": "Low",
        "Possible thrust bearing wear": "High",
        "Pressure system event": "Medium",
        "Thermal event": "Low",
        "Cold start / low load": "Low",
        "Unclassified": "Low",
    }
    return priorities.get(category, "Low")


def suggested_action(category):
    actions = {
        "Shutdown": (
            "Compressor is stopped. Verify this shutdown was planned. "
            "No further action during the shutdown."
        ),
        "Restart transient": (
            "Compressor is warming up after restart. Monitor temperatures "
            "and vibration until they reach steady state."
        ),
        "Possible thrust bearing wear": (
            "Investigate thrust bearing. Check axial displacement trend, "
            "verify lube oil flow to thrust bearing, and schedule inspection "
            "at next opportunity."
        ),
        "Pressure system event": (
            "Check lube oil and seal oil pressure. Inspect filters for clogging. "
            "Verify control valve response on the affected pressure loop."
        ),
        "Thermal event": (
            "Check cooling water supply temperature and ambient conditions. "
            "If temperatures remain high after ambient normalises, inspect "
            "lube oil cooler."
        ),
        "Cold start / low load": (
            "Machine is running below its annual average speed and temperature. "
            "Confirm process conditions are as intended."
        ),
        "Unclassified": (
            "Pattern does not match any known failure mode. Review sensor traces "
            "manually and consider whether a new failure mode is present."
        ),
    }
    return actions.get(category, "Review manually.")


def add_decision_support(periods_df):
    periods_df = periods_df.copy().sort_values("start").reset_index(drop=True)
    categories = []

    for i, row in periods_df.iterrows():
        # Pass the previous row and its already-assigned category
        prev = periods_df.iloc[i - 1] if i > 0 else None
        if prev is not None:
            prev = prev.copy()
            prev["category_placeholder"] = categories[-1]
        cat = classify_period(row, prev)
        categories.append(cat)

    periods_df["category"] = categories
    periods_df["priority"] = periods_df["category"].apply(priority_for)
    periods_df["suggested_action"] = periods_df["category"].apply(suggested_action)
    return periods_df


if __name__ == "__main__":
    # Load the filtered anomaly periods
    df = pd.read_csv("results/tables/groups_filtered.csv",
                     parse_dates=["start", "end"])

    # Add the decision support columns
    df = add_decision_support(df)

    # Save the enriched table
    os.makedirs("results/tables", exist_ok=True)
    df.to_csv("results/tables/periods_with_actions.csv", index=False)
    print("Saved: results/tables/periods_with_actions.csv")

    # Print a clean summary
    print("\n" + "=" * 80)
    print("DECISION SUPPORT SUMMARY")
    print("=" * 80)
    for _, row in df.iterrows():
        print(f"\nPeriod {int(row['period_id'])}: "
              f"{row['start']} to {row['end']}  ({row['duration_hours']:.1f} h)")
        print(f"  Category:  {row['category']}")
        print(f"  Priority:  {row['priority']}")
        print(f"  Action:    {row['suggested_action']}")

    # Count categories
    print("\n" + "=" * 80)
    print("CATEGORY COUNTS")
    print("=" * 80)
    print(df["category"].value_counts())