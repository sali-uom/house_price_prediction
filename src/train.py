"""Training pipeline: features -> tuning -> evaluation -> NGBoost final model, tracked in MLflow (DagsHub).

Usage:  python -m src.train
Env vars: N_TRIALS (baselines, default 25), NGB_TRIALS (NGBoost, default 15), DATA_PATH.
"""
import json
import os
from pathlib import Path

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
import sklearn
from ngboost import NGBRegressor
from ngboost.distns import Normal
from scipy.stats import norm
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.features import FEATURES, FEATURES_CAT, FEATURES_NUM, KEYWORDS, build_features, clean

# Lets scikit-learn pipelines recognise a fitted NGBoost model
NGBRegressor.__sklearn_is_fitted__ = lambda self: bool(getattr(self, "base_models", None))

DATA_PATH = os.getenv("DATA_PATH", "data/houses_sample.csv")
MODEL_DIR = Path("models")
N_TRIALS = int(os.getenv("N_TRIALS", 25))
NGB_TRIALS = int(os.getenv("NGB_TRIALS", 15))
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


def ngb_model(n_estimators=500, learning_rate=0.01, max_depth=3, min_samples_leaf=1, minibatch_frac=1.0, col_sample=1.0):
    base = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=RANDOM_STATE)
    return pipe(NGBRegressor(Dist=Normal, Base=base, n_estimators=n_estimators, learning_rate=learning_rate,
                             minibatch_frac=minibatch_frac, col_sample=col_sample, random_state=RANDOM_STATE, verbose=False))


def ngb_dist(model, X):
    params = model.named_steps["model"].pred_dist(model.named_steps["prep"].transform(X)).params
    return params["loc"], params["scale"]


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
    "NGBoost": lambda t: ngb_model(
        n_estimators=t.suggest_int("n_estimators", 300, 1000, step=100),
        learning_rate=t.suggest_float("learning_rate", 0.01, 0.1, log=True), max_depth=t.suggest_int("max_depth", 3, 7),
        min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 20), minibatch_frac=t.suggest_float("minibatch_frac", 0.5, 1.0),
        col_sample=t.suggest_float("col_sample", 0.5, 1.0)),
}
START_POINTS = {
    "RandomForest": {"n_estimators": 300, "max_depth": 30, "min_samples_leaf": 1, "max_features": 1.0},
    "XGBoost": {"n_estimators": 300, "learning_rate": 0.1, "max_depth": 6, "min_child_weight": 1, "subsample": 1.0, "colsample_bytree": 1.0, "reg_lambda": 1.0},
    "NGBoost": {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 5, "min_samples_leaf": 5, "minibatch_frac": 0.8, "col_sample": 0.8},
}
TRIALS = {"NGBoost": NGB_TRIALS}


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
with mlflow.start_run(run_name=f"train-ngboost-{NGB_TRIALS}-trials"):
    mlflow.log_params({"n_trials": N_TRIALS, "ngb_trials": NGB_TRIALS, "cv_folds": CV_FOLDS, "rows": len(data),
                       "train_rows": len(X_train), "test_rows": len(X_test), "data_path": DATA_PATH})
    tuned, results = {}, {}

    for name, space in SPACES.items():
        with mlflow.start_run(run_name=name, nested=True):
            study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
            study.enqueue_trial(START_POINTS[name])
            study.optimize(lambda t: cv_rmse(space(t)), n_trials=TRIALS.get(name, N_TRIALS))
            tuned[name] = space(optuna.trial.FixedTrial(study.best_params)).fit(X_train, y_train)
            results[name] = {"cv_rmse_log": study.best_value, **test_metrics(tuned[name])}
            mlflow.log_params(study.best_params)
            mlflow.log_metrics(results[name])
            print(f"{name:12s} {results[name]}")

    # NGBoost is the final model: the new algorithm, with a calibrated uncertainty range per house
    final_model = tuned["NGBoost"]
    z_scores = []
    for tr, va in cv_splits:
        m = clone(final_model).fit(X_train.iloc[tr], y_train.iloc[tr])
        loc, scale = ngb_dist(m, X_train.iloc[va])
        z_scores.append(np.abs(y_train.iloc[va].values - loc) / scale)
    interval_scale = float(np.quantile(np.concatenate(z_scores), 0.8) / norm.ppf(0.9))

    loc, scale = ngb_dist(final_model, X_test)
    z80 = norm.ppf(0.9) * scale * interval_scale
    coverage_80 = float(np.mean((y_test.values >= loc - z80) & (y_test.values <= loc + z80)) * 100)
    nll = float(-norm.logpdf(y_test.values, loc, scale * interval_scale).mean())
    results["NGBoost"].update({"interval_scale": interval_scale, "test_coverage_80_pct": coverage_80, "test_nll": nll})

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(final_model, MODEL_DIR / "house_price_model.pkl")
    meta = {
        "model_name": "NGBoost",
        "features_num": FEATURES_NUM,
        "features_cat": FEATURES_CAT,
        "keywords": list(KEYWORDS),
        "districts": sorted(data["district"].unique().tolist()),
        "cities_by_district": {d: sorted(g["city"].unique().tolist()) for d, g in data.groupby("district")},
        "test_mape": round(results["NGBoost"]["test_mape_pct"], 1),
        "interval_scale": round(interval_scale, 4),
        "test_coverage_80": round(coverage_80, 1),
        "sklearn_version": sklearn.__version__,
    }
    (MODEL_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2))
    pd.DataFrame(results).T.to_csv(MODEL_DIR / "model_comparison.csv")

    mlflow.set_tag("final_model", "NGBoost")
    mlflow.log_metrics({f"final_{k}": v for k, v in results["NGBoost"].items()})
    mlflow.log_artifacts(str(MODEL_DIR), artifact_path="model")
    print(f"\nNGBoost: 80% range coverage {coverage_80:.1f}%, calibration factor {interval_scale:.2f}. Saved to {MODEL_DIR}/")