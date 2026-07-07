"""Unit tests for feature engineering functions."""

import pandas as pd

from src.features import add_date_features, add_holiday_flag, add_lag_and_rolling_features


def test_add_date_features_flags_weekend():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-06", "2024-01-07"]),  # Mon, Sat, Sun
    })

    result = add_date_features(df)

    assert result["is_weekend"].tolist() == [0, 1, 1]
    assert result["day_of_week_num"].tolist() == [0, 5, 6]


def test_add_holiday_flag_matches_national_non_transferred():
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-02"])})
    holidays_df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "locale": ["National", "National"],
        "transferred": [False, True],
    })

    result = add_holiday_flag(df, holidays_df)

    assert result["is_holiday"].tolist() == [1, 0]


def test_lag_and_rolling_features_respect_group_boundaries():
    df = pd.DataFrame({
        "store_nbr": [1, 1, 2],
        "family": ["A", "A", "A"],
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01"]),
        "sales": [10.0, 20.0, 999.0],
    })

    result = add_lag_and_rolling_features(df)
    store_2_row = result[result["store_nbr"] == 2].iloc[0]

    assert store_2_row["sales_lag_7"] == 0
    assert store_2_row["sales_rolling_mean_7"] == 0