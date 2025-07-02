# process_dataset.py

import pandas as pd


# Drop columns except column in exclude_columns
def drop_columns_except(df: pd.DataFrame, exclude_columns: list) -> pd.DataFrame:
    """
    Drop all columns except those specified in exclude_columns.

    Parameters:
    df (pd.DataFrame): The DataFrame from which to drop columns.
    exclude_columns (list): List of column names to keep.

    Returns:
    pd.DataFrame: DataFrame with only the specified columns.
    """
    return df[exclude_columns]


# Rename columns in a DataFrame
def rename_columns(df: pd.DataFrame, column_mapping: dict) -> pd.DataFrame:
    """
    Rename columns in a DataFrame based on a mapping.

    Parameters:
    df (pd.DataFrame): The DataFrame with columns to rename.
    column_mapping (dict): A dictionary mapping old column names to new column names.

    Returns:
    pd.DataFrame: DataFrame with renamed columns.
    """
    return df.rename(columns=column_mapping)


df = pd.read_csv("data/datasets/save/binance_BTCUSDT_20170817_20250630_1d_dataset.csv")

# Define columns to keep
exclude_columns = ["datetime", "open_1d", "high_1d", "low_1d", "close_1d", "volume_1d"]

# Drop columns except those specified
df = drop_columns_except(df, exclude_columns)

# Define column mapping for renaming
column_mapping = {
    "open_1d": "Open",
    "high_1d": "High",
    "low_1d": "Low",
    "close_1d": "Close",
    "volume_1d": "Volume",
}

# Rename columns
df = rename_columns(df, column_mapping)
# Save the processed DataFrame to a new CSV file
df.to_csv(
    "data/datasets/save/processed_binance_BTCUSDT_20170817_20250630_1d_dataset.csv",
    index=False,
)
