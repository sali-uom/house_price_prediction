# Sri Lanka House Price Estimator

Predicts the listing price of a Sri Lankan house from its features, explains each prediction with SHAP and LIME, and serves the model through a Streamlit app with an MLOps pipeline (GitHub, DagsHub, Azure).
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

## Models and results
Random Forest (baseline), XGBoost, LightGBM, CatBoost and a stacking ensemble, all tuned with Optuna using seller-grouped cross-validation.

| Model | RMSE (log) | R² (log) | MAPE (%) |
|---|---|---|---|
| **CatBoost** | **0.365** | **0.800** | **25.8** |
| Stacking | 0.366 | 0.799 | 25.9 |
| LightGBM | 0.376 | 0.787 | 26.5 |
| XGBoost | 0.386 | 0.776 | 27.4 |
| Random Forest | 0.410 | 0.748 | 27.6 |

**CatBoost was selected:** it had the lowest test and cross-validation error and handles the city and district categories natively. Stacking tied but is more complex.

## What drives price (SHAP)
1. Location (city + district) has the largest combined effect.
2. Bathrooms and house size are the main structural drivers.
3. Land size and a "luxury" mention in the ad add a premium.
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
