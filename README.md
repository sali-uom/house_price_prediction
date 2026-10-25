# Sri Lanka House Price Estimator

Predicts the listing price of a Sri Lankan house, with a calibrated price range, using NGBoost. Explains each prediction with SHAP and LIME, and serves the model through a Streamlit app with an MLOps pipeline (GitHub, DagsHub, Azure).

Setup and run instructions are in [RUN.md](RUN.md).

### The app is live at https://housepriceprediction-cdcyexdqhzf0g2e4.eastasia-01.azurewebsites.net/

## Problem
House prices in Sri Lanka are set by seller and broker judgement, so buyers can't easily tell whether an asking price is fair. This regression model estimates a fair listing price and explains what drives it.

## Data
- **Source:** Sri Lanka Property Ads [Dataset](https://www.kaggle.com/datasets/ivantha/sri-lanka-property-ads-dataset?select=properties.csv) (Kaggle), ikman.lk listings, Sep 2022 – Apr 2023
- **Rows:** 14,396 houses for sale → 12,807 after cleaning
- **Features:** bedrooms, bathrooms, house size, land size, district, city, plus storeys, road width and 8 amenity keywords extracted from the ad text
- **Split:** grouped by seller (10,092 train / 2,715 test), so one agent's near-identical ads can't leak into the test set

## Key EDA findings
- Price is right-skewed, so the target is `log(price)`.
- Bathrooms (r = 0.72), house size (0.63) and bedrooms (0.58) correlate most with log price.
- Land size barely correlates (0.05); its effect depends on location, which suits tree models.
- Most listings are from Colombo and Gampaha.

## Algorithm: NGBoost
NGBoost (Natural Gradient Boosting) predicts a full price distribution for each house instead of a single number, so every estimate comes with its own uncertainty range. It is gradient boosting applied to the mean and spread of a normal distribution on log price, trained with the natural gradient. Random Forest and XGBoost are tuned baselines. All models are tuned with Optuna using cross validation grouped by seller.

The price ranges are calibrated on the cross validation folds, so an 80% range contains close to 80% of actual test prices.

| Model | Test RMSE (log) | R² (log) | MAPE (%) |
|---|---|---|---|
| XGBoost (baseline) | 0.370 | 0.794 | 25.3 |
| **NGBoost** | **0.379** | **0.784** | **26.3** |
| Random Forest (baseline) | 0.408 | 0.750 | 27.5 |

NGBoost is close to XGBoost on accuracy and is the only model with a range per house: after calibration, its 80% range contains 84.9% of actual test prices and its 95% range 95.7%. Its uncertainty is about 65% higher outside the Western Province, where it has less data.

## What drives price and uncertainty (SHAP)
SHAP explains both outputs of NGBoost: what pushes the price up or down (house size, bathrooms, location) and what makes the model less certain (rare locations and unusual properties). The model is noticeably less certain outside the Western Province, where it has less data.

## Limitations
Asking prices, not sale prices; data concentrated in the Western Province; road access and plot shape are not captured.

## MLOps
| Tool | Role |
|---|---|
| GitHub + Actions | Code versioning; tests and quality gates (MAPE < 40%, calibrated ranges) run before every deploy |
| DagsHub | MLflow experiment tracking and DVC data versioning |
| Azure App Service | Hosts the live Streamlit app |

## Repository
```
app.py              Streamlit front end
src/features.py     Feature extraction and cleaning
src/train.py        Tuning, evaluation, range calibration, MLflow logging
tests/              Unit tests and quality gate
models/             Saved model and metadata
notebooks/          Full analysis (EDA, SHAP, LIME)
```