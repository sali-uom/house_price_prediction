# Sri Lanka House Price Estimator

Predicts the listing price of a Sri Lankan house from its features, explains each prediction with SHAP and LIME, and serves the model through a Streamlit app with an MLOps pipeline (GitHub, DagsHub, Azure).

Local setup and run instructions are in [RUN.md](RUN.md).

The app is live at https://housepriceprediction-cdcyexdqhzf0g2e4.eastasia-01.azurewebsites.net/

## Problem
House prices in Sri Lanka are set by seller and broker judgement, so buyers can't easily tell whether an asking price is fair. This regression model estimates a fair listing price and explains what drives it.

## Data
- **Source:** Sri Lanka Property Ads [Dataset](https://www.kaggle.com/datasets/ivantha/sri-lanka-property-ads-dataset?select=properties.csv), ikman.lk listings, Sep 2022 – Apr 2023
- **Rows:** 14,396 houses for sale → 12,807 after cleaning
- **Features:** bedrooms, bathrooms, house size, land size, district, city, plus storeys, road width and 8 amenity keywords extracted from the ad text
- **Split:** grouped by seller (10,092 train / 2,715 test), so one agent's near-identical ads can't leak into the test set

## Key EDA findings
- Price is right-skewed, so the target is `log(price)`.
- Bathrooms (r = 0.72), house size (0.63) and bedrooms (0.58) correlate most with log price.
- Land size barely correlates (0.05); its effect depends on location, which suits tree models.
- Most listings are from Colombo and Gampaha.

## Models and results
Random Forest (baseline), XGBoost, LightGBM, CatBoost and a stacking ensemble, all tuned with Optuna using cross validation grouped by seller.

| Model | RMSE (log) | R² (log) | MAPE (%) |
|---|---|---|---|
| **Stacking** | **0.364** | **0.800** | **25.3** |
| LightGBM | 0.368 | 0.796 | 25.7 |
| XGBoost | 0.370 | 0.794 | 25.3 |
| CatBoost | 0.371 | 0.793 | 26.3 |
| Random Forest | 0.408 | 0.750 | 27.5 |

**Stacking was selected:** it combines XGBoost, LightGBM and CatBoost through a RidgeCV combiner and had the lowest test RMSE and highest R². XGBoost (weight 0.47) and CatBoost (0.44) carry most of the stack. CatBoost and stacking are the new methods not covered in lectures.

## What drives price (SHAP)
1. House size and bathrooms are the strongest single features.
2. Location (city + district) has the largest combined effect.
3. A "luxury" mention and land size add a premium.
4. Storeys and road width matter little, since most ads omit them.

LIME explanations for individual houses agree with SHAP on these main drivers.

## Limitations
Asking prices, not sale prices; data concentrated in the Western Province; road access and plot shape are not captured.

## MLOps
| Tool | Role |
|---|---|
| GitHub + Actions | Code versioning; tests and a quality gate (MAPE < 40%) run before every deploy |
| DagsHub | MLflow experiment tracking and DVC data versioning |
| Azure App Service | Hosts the live Streamlit app |

## Repository
```
app.py              Streamlit front end
src/features.py     Feature extraction and cleaning
src/train.py        Tuning, evaluation, model selection, MLflow logging
tests/              Unit tests and quality gate
models/             Saved model and metadata
notebooks/          Full analysis (EDA, SHAP, LIME)
```
