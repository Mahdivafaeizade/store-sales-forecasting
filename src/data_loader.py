"""Load raw CSV files for the store sales forecasting project."""

import pandas as pd

DATA_DIR = "data"


def load_csv(file_name: str, parse_dates: bool = True) -> pd.DataFrame:
    """
    Load a CSV file from the data directory.

    Args:
        file_name (str): The name of the CSV file to load.
        parse_dates (bool): Whether the file has a 'date' column to parse.

    Returns:
        pd.DataFrame: The loaded data.
    """
    path = f"{DATA_DIR}/{file_name}"
    if parse_dates:
        return pd.read_csv(path, parse_dates=["date"])
    return pd.read_csv(path)