"""Train a LightGBM model to forecast store sales and save it to disk."""

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_log_error

from src.data_loader import load_csv
from src.features import (
    add_date_features,
    add_holiday_flag,
    add_lag_and_rolling_features,
    merge_oil,
    merge_stores,
    set_categorical_dtypes,
    CATEGORICAL_COLS,
)

FEATURE_COLS = [
    "store_nbr", "family", "onpromotion", "day_of_week_num", "month", "year",
    "is_weekend", "is_holiday", "dcoilwtico", "city", "state", "type", "cluster",
    "sales_lag_7", "sales_rolling_mean_7",
]
TARGET_COL = "sales"


def build_features(df: pd.DataFrame, holidays_df, oil_df, stores_df) -> pd.DataFrame:
    """Run the full feature engineering pipeline on a raw dataframe."""
    df = add_date_features(df)
    df = add_holiday_flag(df, holidays_df)
    df = merge_oil(df, oil_df)
    df = merge_stores(df, stores_df)
    df = set_categorical_dtypes(df)
    return df


def main():
    train_df = load_csv("train.csv")
    holidays_df = load_csv("holidays_events.csv")
    oil_df = load_csv("oil.csv")
    stores_df = load_csv("stores.csv", parse_dates=False)

    train_df = build_features(train_df, holidays_df, oil_df, stores_df)
    train_df = add_lag_and_rolling_features(train_df)

    reference = (
        train_df.sort_values("date")
        .groupby(["store_nbr", "family"], observed=True)
        .tail(1)[["store_nbr", "family", "sales_lag_7", "sales_rolling_mean_7"]]
    )
    reference.to_csv("models/reference_features.csv", index=False)

    latest_oil_price = oil_df["dcoilwtico"].ffill().iloc[-1]
    with open("models/latest_oil_price.txt", "w") as f:
        f.write(str(latest_oil_price))

    cutoff_date = train_df["date"].max() - pd.Timedelta(days=15)
    train_split = train_df[train_df["date"] <= cutoff_date]
    val_split = train_df[train_df["date"] > cutoff_date]

    X_train, y_train = train_split[FEATURE_COLS], train_split[TARGET_COL]
    X_val, y_val = val_split[FEATURE_COLS], val_split[TARGET_COL]

    model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
    model.fit(X_train, np.log1p(y_train), categorical_feature=CATEGORICAL_COLS)

    val_preds = np.clip(np.expm1(model.predict(X_val)), 0, None)
    rmsle = root_mean_squared_log_error(y_val, val_preds)
    print(f"Validation RMSLE: {rmsle:.4f}")

    joblib.dump(model, "models/lgbm_model.joblib")
    print("Model saved to models/lgbm_model.joblib")


if __name__ == "__main__":
    main()