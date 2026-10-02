import pandas as pd
import os


def load_raw_data(file_path):
    df = pd.read_excel(file_path)
    return df


def fix_timestamp(df):
    # Convert the Timestamp column from text to actual datetime
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df


def rename_columns(df):
    # Map the original refinery tag names to simple names.
    # The original tags follow ISA-5.1 convention:
    # ZI = axial displacement, PI = pressure, TI = temperature,
    # XI = vibration, SI = speed.
    new_names = {
        "75ZI800BA.pv": "axial_1",
        "75ZI800BB.pv": "axial_2",
        "75ZI801BA.pv": "axial_3",
        "75ZI801BB.pv": "axial_4",
        "75PI808.pv": "pressure_1",
        "75PI823.pv": "pressure_2",
        "75PI870.pv": "pressure_3",
        "75PI845.pv": "pressure_4",
        "75PDI853.pv": "pressure_diff",
        "75TI821.pv": "temp_1",
        "75TI822.pv": "temp_2",
        "75TI827.pv": "temp_3",
        "75TI828.pv": "temp_4",
        "75TI824.pv": "temp_5",
        "75TI829.pv": "temp_6",
        "75TI830.pv": "temp_7",
        "75TI831.pv": "temp_8",
        "75TI832.pv": "temp_9",
        "75TI836.pv": "temp_10",
        "75TI834.pv": "temp_11",
        "75XI821BX.pv": "vibration_1",
        "75XI822BY.pv": "vibration_2",
        "75XI823BX.pv": "vibration_3",
        "75XI824BX.pv": "vibration_4",
        "75SI865R.pv": "speed",
    }
    df = df.rename(columns=new_names)
    return df


def check_sampling(df):
    # Check the time gap between consecutive rows
    time_diff = df["Timestamp"].diff().dropna()
    print("Most common gap between rows:", time_diff.mode()[0])
    print("Smallest gap:", time_diff.min())
    print("Largest gap:", time_diff.max())
    return time_diff


def save_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)
    print("\nSaved cleaned data to:", output_path)


if __name__ == "__main__":
    input_path = "data/raw/Centrifugal compressor _2022.xlsx"
    output_path = "data/processed/compressor_clean.csv"

    # Create the processed folder if it does not exist yet
    os.makedirs("data/processed", exist_ok=True)

    df = load_raw_data(input_path)
    df = fix_timestamp(df)
    df = rename_columns(df)

    # Quick sanity checks
    print("Shape after cleaning:", df.shape)
    print("\nColumn names after renaming:")
    print(list(df.columns))
    print("\nFirst 5 rows:")
    print(df.head())

    check_sampling(df)

    save_cleaned_data(df, output_path)