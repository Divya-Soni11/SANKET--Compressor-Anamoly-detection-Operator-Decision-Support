import pandas as pd

# Path to the Excel file we downloaded
file_path = "data/raw/Centrifugal compressor _2022.xlsx"

# Load the Excel file into a pandas DataFrame
df = pd.read_excel(file_path)

# Print the basic shape of the data
print("Shape of data:", df.shape)

# Print all column names so we can see what sensors we have
print("\nColumn names:")
for col in df.columns:
    print(col)

# Print the first 5 rows so we can see what the values look like
print("\nFirst 5 rows:")
print(df.head())

# Print the data type of each column
print("\nData types:")
print(df.dtypes)

# Check for missing values in each column
print("\nMissing values per column:")
print(df.isnull().sum())