"""Feature engineering for the store sales forecasting project."""

import pandas as pd

CATEGORICAL_COLS = ["family", "city", "state", "type"]


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add day-of-week, month, year, and weekend flag from the date column."""
    df = df.copy()
    df["day_of_week_num"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["is_weekend"] = df["day_of_week_num"].isin([5, 6]).astype(int)
    return df


def add_holiday_flag(df: pd.DataFrame, holidays_df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows that fall on a national, non-transferred holiday."""
    df = df.copy()
    national_holidays = holidays_df[
        (holidays_df["locale"] == "National") & (holidays_df["transferred"] == False)
    ]["date"].unique()
    df["is_holiday"] = df["date"].isin(national_holidays).astype(int)
    return df


def merge_oil(df: pd.DataFrame, oil_df: pd.DataFrame) -> pd.DataFrame:
    """Attach daily oil price, filling gaps from weekends/holidays."""
    df = df.merge(oil_df, on="date", how="left")
    df["dcoilwtico"] = df["dcoilwtico"].ffill().bfill()
    return df


def merge_stores(df: pd.DataFrame, stores_df: pd.DataFrame) -> pd.DataFrame:
    """Attach store metadata: city, state, type, cluster."""
    return df.merge(stores_df, on="store_nbr", how="left")


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 7-day lag and a 7-day rolling mean of sales, per store+family."""
    df = df.sort_values(["store_nbr", "family", "date"]).copy()
    grouped = df.groupby(["store_nbr", "family"], observed=True)["sales"]
    df["sales_lag_7"] = grouped.shift(7).fillna(0)
    df["sales_rolling_mean_7"] = (
        grouped.transform(lambda s: s.shift(1).rolling(window=7).mean()).fillna(0)
    )
    return df


def set_categorical_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert text columns to pandas 'category' dtype for LightGBM."""
    df = df.copy()
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")
    return df