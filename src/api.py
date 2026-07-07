"""FastAPI service for serving store sales predictions."""

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.data_loader import load_csv
from src.train import FEATURE_COLS

app = FastAPI(title="Store Sales Forecasting API")

model = joblib.load("models/lgbm_model.joblib")
reference_features = pd.read_csv("models/reference_features.csv")
stores_df = load_csv("stores.csv", parse_dates=False)
holidays_df = load_csv("holidays_events.csv")

with open("models/latest_oil_price.txt") as f:
    LATEST_OIL_PRICE = float(f.read())

NATIONAL_HOLIDAYS = set(
    holidays_df[
        (holidays_df["locale"] == "National") & (holidays_df["transferred"] == False)
    ]["date"].dt.strftime("%Y-%m-%d")
)


class PredictionRequest(BaseModel):
    store_nbr: int
    family: str
    date: str
    onpromotion: int = 0


class PredictionResponse(BaseModel):
    predicted_sales: float


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    date = pd.Timestamp(request.date)

    store_row = stores_df[stores_df["store_nbr"] == request.store_nbr]
    if store_row.empty:
        raise HTTPException(status_code=404, detail=f"Unknown store_nbr: {request.store_nbr}")

    ref_row = reference_features[
        (reference_features["store_nbr"] == request.store_nbr)
        & (reference_features["family"] == request.family)
    ]
    if ref_row.empty:
        raise HTTPException(status_code=404, detail=f"Unknown family: {request.family}")

    row = {
        "store_nbr": request.store_nbr,
        "family": request.family,
        "onpromotion": request.onpromotion,
        "day_of_week_num": date.dayofweek,
        "month": date.month,
        "year": date.year,
        "is_weekend": int(date.dayofweek in (5, 6)),
        "is_holiday": int(date.strftime("%Y-%m-%d") in NATIONAL_HOLIDAYS),
        "dcoilwtico": LATEST_OIL_PRICE,
        "city": store_row["city"].iloc[0],
        "state": store_row["state"].iloc[0],
        "type": store_row["type"].iloc[0],
        "cluster": store_row["cluster"].iloc[0],
        "sales_lag_7": ref_row["sales_lag_7"].iloc[0],
        "sales_rolling_mean_7": ref_row["sales_rolling_mean_7"].iloc[0],
    }

    X = pd.DataFrame([row])[FEATURE_COLS]
    for col in ["family", "city", "state", "type"]:
        X[col] = X[col].astype("category")

    pred_log = model.predict(X)[0]
    predicted_sales = float(np.clip(np.expm1(pred_log), 0, None))

    return PredictionResponse(predicted_sales=predicted_sales)


@app.get("/")
def root():
    return {"message": "Store Sales Forecasting API. POST to /predict."}