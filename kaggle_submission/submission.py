"""
Standalone Kaggle kernel script for Store Sales - Time Series Forecasting.

Self-contained version of the pipeline in src/ (data_loader.py, features.py,
train.py) so it can run inside Kaggle's isolated notebook environment,
which cannot import from an external GitHub repo.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import root_mean_squared_log_error

INPUT_DIR = "/kaggle/input/competitions/store-sales-time-series-forecasting"
CATEGORICAL_COLS = ["family", "city", "state", "type"]
FEATURE_COLS = [
    "store_nbr", "family", "onpromotion", "day_of_week_num", "month", "year",
    "is_weekend", "is_holiday", "dcoilwtico", "city", "state", "type", "cluster",
    "sales_lag_7", "sales_rolling_mean_7",
]


def add_date_features(df):
    df = df.copy()
    df["day_of_week_num"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["is_weekend"] = df["day_of_week_num"].isin([5, 6]).astype(int)
    return df


def add_holiday_flag(df, holidays_df):
    df = df.copy()
    national_holidays = holidays_df[
        (holidays_df["locale"] == "National") & (holidays_df["transferred"] == False)
    ]["date"].unique()
    df["is_holiday"] = df["date"].isin(national_holidays).astype(int)
    return df


def merge_oil(df, oil_df):
    df = df.merge(oil_df, on="date", how="left")
    df["dcoilwtico"] = df["dcoilwtico"].ffill().bfill()
    return df


def merge_stores(df, stores_df):
    return df.merge(stores_df, on="store_nbr", how="left")


def add_lag_and_rolling_features(df):
    df = df.sort_values(["store_nbr", "family", "date"]).copy()
    grouped = df.groupby(["store_nbr", "family"], observed=True)["sales"]
    df["sales_lag_7"] = grouped.shift(7).fillna(0)
    df["sales_rolling_mean_7"] = (
        grouped.transform(lambda s: s.shift(1).rolling(window=7).mean()).fillna(0)
    )
    return df


def set_categorical_dtypes(df):
    df = df.copy()
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")
    return df


def build_features(df, holidays_df, oil_df, stores_df):
    df = add_date_features(df)
    df = add_holiday_flag(df, holidays_df)
    df = merge_oil(df, oil_df)
    df = merge_stores(df, stores_df)
    df = set_categorical_dtypes(df)
    return df


def main():
    train_df = pd.read_csv(f"{INPUT_DIR}/train.csv", parse_dates=["date"])
    test_df = pd.read_csv(f"{INPUT_DIR}/test.csv", parse_dates=["date"])
    holidays_df = pd.read_csv(f"{INPUT_DIR}/holidays_events.csv", parse_dates=["date"])
    oil_df = pd.read_csv(f"{INPUT_DIR}/oil.csv", parse_dates=["date"])
    stores_df = pd.read_csv(f"{INPUT_DIR}/stores.csv")

    train_df = build_features(train_df, holidays_df, oil_df, stores_df)
    train_df = add_lag_and_rolling_features(train_df)

    cutoff_date = train_df["date"].max() - pd.Timedelta(days=15)
    train_split = train_df[train_df["date"] <= cutoff_date]
    val_split = train_df[train_df["date"] > cutoff_date]

    X_train, y_train = train_split[FEATURE_COLS], train_split["sales"]
    X_val, y_val = val_split[FEATURE_COLS], val_split["sales"]

    model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
    model.fit(X_train, np.log1p(y_train), categorical_feature=CATEGORICAL_COLS)

    val_preds = np.clip(np.expm1(model.predict(X_val)), 0, None)
    rmsle = root_mean_squared_log_error(y_val, val_preds)
    print(f"Validation RMSLE: {rmsle:.4f}")

    test_df = build_features(test_df, holidays_df, oil_df, stores_df)
    last_features = (
        train_df.sort_values("date")
        .groupby(["store_nbr", "family"], observed=True)
        .tail(1)[["store_nbr", "family", "sales_lag_7", "sales_rolling_mean_7"]]
    )
    test_df = test_df.merge(last_features, on=["store_nbr", "family"], how="left")

    test_preds = np.clip(np.expm1(model.predict(test_df[FEATURE_COLS])), 0, None)
    submission = pd.DataFrame({"id": test_df["id"], "sales": test_preds})
    submission.to_csv("submission.csv", index=False)
    print("Saved submission.csv")


if __name__ == "__main__":
    main()
