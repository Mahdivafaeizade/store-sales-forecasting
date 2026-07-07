---
title: Store Sales Forecasting API
emoji: 📈
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Store Sales Forecasting

Forecasting daily grocery sales for Corporación Favorita, a large Ecuadorian retailer, using historical sales, promotions, holidays, and oil price data. Built for [Kaggle's Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) competition.

## Problem

Retailers have to decide how much stock to buy without knowing future demand. Overstock leads to waste (especially for perishables); understock leads to lost sales and unhappy customers. This project predicts daily unit sales per store and product category, 15 days into the future, so those buying decisions can be made with actual data instead of guesswork.

## Approach

1. **EDA** — found three dominant patterns in the data: a clear upward sales trend over 2013-2017, a repeating drop to zero every New Year's Day, and a strong weekly cycle (Saturday/Sunday sales ~60% higher than the weekly low on Thursday).
2. **Feature engineering** — date parts (day of week, month, year), a national holiday flag, daily oil price (Ecuador's economy is oil-dependent), store metadata (city, state, type, cluster), and time series lag/rolling-average features (7-day lag, 7-day rolling mean of sales) computed per store+product to avoid leaking information across unrelated series.
3. **Model** — LightGBM gradient boosting, trained on `log1p(sales)` to directly optimize for the competition's evaluation metric.
4. **Validation** — time-based split (last 15 days held out, matching the real competition test window). Random/shuffled splits are invalid for time series since they'd leak future information into training.

## Results

| Model | Validation RMSLE |
|---|---|
| Naive baseline (repeat last week's sales) | 0.5690 |
| LightGBM (lag feature only) | 0.4775 |
| LightGBM (+ rolling mean feature) | **0.4101** |

The final model reduces error by ~28% versus the naive baseline. Feature importance shows product category, the 7-day rolling mean, and day-of-week as the strongest predictors — consistent with what the EDA found by hand.

## Known limitation

The competition's test set covers 15 future days with no real sales data. Lag/rolling features for that window are approximated by freezing each store+product's last known values from the end of the training period, rather than a full recursive day-by-day forecast. A future improvement would be iterative multi-step forecasting, re-predicting one day at a time and feeding each prediction into the next day's lag features.

## Project structure

```
store-sales-forecasting/
├── data/                    # raw Kaggle CSVs (not tracked in git — see Setup)
├── notebooks/
│   └── 01_data_exploration.ipynb   # EDA and initial prototyping
├── src/
│   ├── data_loader.py       # CSV loading
│   ├── features.py          # feature engineering functions
│   ├── train.py             # trains the model, saves model + serving artifacts
│   └── api.py                # FastAPI prediction service
├── models/                  # trained model + reference data for serving
├── outputs/
│   └── submission.csv       # Kaggle competition submission
└── tests/
    └── test_features.py     # unit tests for feature engineering
```

## Setup

```bash
pip install -r requirements.txt
```

Download the competition data from [Kaggle](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data) into `data/` (requires a Kaggle account and API key).

## Usage

Train the model (also regenerates `models/` artifacts):
```bash
python3 -m src.train
```

Run the tests:
```bash
python3 -m pytest tests/ -v
```

Run the API:
```bash
python3 -m uvicorn src.api:app --reload
```

Example request:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"store_nbr": 1, "family": "BEVERAGES", "date": "2017-08-20", "onpromotion": 5}'
```

## Tech stack

Python, pandas, LightGBM, scikit-learn, FastAPI, pytest.
