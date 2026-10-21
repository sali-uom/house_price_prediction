"""Training pipeline: features -> tuning -> evaluation -> best model, all tracked in MLflow (DagsHub).

Usage:  python -m src.train            (N_TRIALS env var controls tuning effort, default 25)
"""
import json
import os
from pathlib import Path

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor

from src.features import FEATURES, FEATURES_CAT, FEATURES_NUM, KEYWORDS, build_features, clean

DATA_PATH = os.getenv("DATA_PATH", "data/houses_sample.csv")
MODEL_DIR = Path("models")
N_TRIALS = int(os.getenv("N_TRIALS", 25))
RANDOM_STATE = 42
CV_FOLDS = 3
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------- data ----------
data = clean(build_features(pd.read_csv(DATA_PATH)))
X, y, groups = data[FEATURES], np.log1p(data["price"]), data["user"]
train_idx, test_idx = next(GroupShuffleSplit(1, test_size=0.2, random_state=RANDOM_STATE).split(X, y, groups))
X_train, X_test, y_train, y_test = X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
cv_splits = list(GroupKFold(CV_FOLDS).split(X_train, y_train, groups.iloc[train_idx]))

# ---------- models ----------
preprocess = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), FEATURES_NUM),
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), FEATURES_CAT),
], verbose_feature_names_out=False).set_output(transform="pandas")


def pipe(model):
    return Pipeline([("prep", preprocess), ("model", model)])


SPACES = {
    "RandomForest": lambda t: pipe(RandomForestRegressor(
        n_estimators=t.suggest_int("n_estimators", 200, 600, step=100), max_depth=t.suggest_int("max_depth", 8, 30),
        min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 10), max_features=t.suggest_float("max_features", 0.3, 1.0),
        random_state=RANDOM_STATE, n_jobs=-1)),
    "XGBoost": lambda t: pipe(XGBRegressor(
        n_estimators=t.suggest_int("n_estimators", 300, 1500, step=100),
        learning_rate=t.suggest_float("learning_rate", 0.01, 0.2, log=True), max_depth=t.suggest_int("max_depth", 3, 10),
        min_child_weight=t.suggest_float("min_child_weight", 1, 20, log=True), subsample=t.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0), reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10, log=True),
        random_state=RANDOM_STATE, n_jobs=-1)),
    "LightGBM": lambda t: pipe(LGBMRegressor(
        n_estimators=t.suggest_int("n_estimators", 300, 1500, step=100),
        learning_rate=t.suggest_float("learning_rate", 0.01, 0.2, log=True), num_leaves=t.suggest_int("num_leaves", 15, 127),
        min_child_samples=t.suggest_int("min_child_samples", 5, 60), subsample=t.suggest_float("subsample", 0.6, 1.0), subsample_freq=1,
        colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0), reg_lambda=t.suggest_float("reg_lambda", 1e-3, 10, log=True),
        random_state=RANDOM_STATE, verbose=-1)),
    "CatBoost": lambda t: CatBoostRegressor(
        iterations=t.suggest_int("iterations", 300, 1500, step=100), learning_rate=t.suggest_float("learning_rate", 0.02, 0.2, log=True),
        depth=t.suggest_int("depth", 4, 10), l2_leaf_reg=t.suggest_float("l2_leaf_reg", 1, 20, log=True),
        cat_features=tuple(FEATURES_CAT), random_seed=RANDOM_STATE, verbose=0, thread_count=-1),
}
START_POINTS = {
    "RandomForest": {"n_estimators": 300, "max_depth": 30, "min_samples_leaf": 1, "max_features": 1.0},
    "XGBoost": {"n_estimators": 300, "learning_rate": 0.1, "max_depth": 6, "min_child_weight": 1, "subsample": 1.0, "colsample_bytree": 1.0, "reg_lambda": 1.0},
    "LightGBM": {"n_estimators": 300, "learning_rate": 0.1, "num_leaves": 31, "min_child_samples": 20, "subsample": 1.0, "colsample_bytree": 1.0, "reg_lambda": 0.001},
    "CatBoost": {"iterations": 1000, "learning_rate": 0.06, "depth": 6, "l2_leaf_reg": 3},
}


def cv_rmse(model):
    return -cross_val_score(model, X_train, y_train, cv=cv_splits, scoring="neg_root_mean_squared_error").mean()


def test_metrics(model):
    pred_log = model.predict(X_test)
    actual, pred = np.expm1(y_test), np.expm1(pred_log)
    return {
        "test_rmse_log": float(np.sqrt(mean_squared_error(y_test, pred_log))),
        "test_r2_log": float(r2_score(y_test, pred_log)),
        "test_mae_rs_m": float(mean_absolute_error(actual, pred) / 1e6),
        "test_mape_pct": float(mean_absolute_percentage_error(actual, pred) * 100),
    }


# ---------- training run ----------
mlflow.set_experiment("sl-house-price")
with mlflow.start_run(run_name=f"train-{N_TRIALS}-trials"):
    mlflow.log_params({"n_trials": N_TRIALS, "cv_folds": CV_FOLDS, "rows": len(data),
                       "train_rows": len(X_train), "test_rows": len(X_test), "data_path": DATA_PATH})
    tuned, results = {}, {}

    for name, space in SPACES.items():
        with mlflow.start_run(run_name=name, nested=True):
            study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
            study.enqueue_trial(START_POINTS[name])
            study.optimize(lambda t: cv_rmse(space(t)), n_trials=N_TRIALS)
            tuned[name] = space(optuna.trial.FixedTrial(study.best_params)).fit(X_train, y_train)
            results[name] = {"cv_rmse_log": study.best_value, **test_metrics(tuned[name])}
            mlflow.log_params(study.best_params)
            mlflow.log_metrics(results[name])
            print(f"{name:12s} {results[name]}")

    with mlflow.start_run(run_name="Stacking", nested=True):
        tuned["Stacking"] = StackingRegressor(
            [("xgb", tuned["XGBoost"]), ("lgbm", tuned["LightGBM"]), ("cat", tuned["CatBoost"])],
            final_estimator=RidgeCV(), cv=cv_splits).fit(X_train, y_train)
        results["Stacking"] = test_metrics(tuned["Stacking"])
        mlflow.log_metrics(results["Stacking"])
        print(f"{'Stacking':12s} {results['Stacking']}")

    best = min(results, key=lambda n: results[n]["test_rmse_log"])
    final_model = tuned[best]

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(final_model, MODEL_DIR / "house_price_model.pkl")
    meta = {
        "model_name": best,
        "features_num": FEATURES_NUM,
        "features_cat": FEATURES_CAT,
        "keywords": list(KEYWORDS),
        "districts": sorted(data["district"].unique().tolist()),
        "cities_by_district": {d: sorted(g["city"].unique().tolist()) for d, g in data.groupby("district")},
        "test_mape": round(results[best]["test_mape_pct"], 1),
    }
    (MODEL_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2))
    pd.DataFrame(results).T.to_csv(MODEL_DIR / "model_comparison.csv")

    mlflow.set_tag("best_model", best)
    mlflow.log_metrics({f"best_{k}": v for k, v in results[best].items()})
    mlflow.log_artifacts(str(MODEL_DIR), artifact_path="model")
    print(f"\nBest model: {best} -> saved to {MODEL_DIR}/")
