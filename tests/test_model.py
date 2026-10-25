import json

import joblib
import numpy as np
import pandas as pd
import sklearn
from ngboost import NGBRegressor

# Lets scikit-learn pipelines recognise a fitted NGBoost model
NGBRegressor.__sklearn_is_fitted__ = lambda self: bool(getattr(self, "base_models", None))

model = joblib.load("models/house_price_model.pkl")
meta = json.load(open("models/model_meta.json"))
FEATURES = meta["features_num"] + meta["features_cat"]


def make_row(**overrides):
    row = {f: 0 for f in meta["keywords"]}
    row.update({"bedrooms": 3, "bathrooms": 2, "house_size_sqft": 2000, "land_size_perches": 10,
                "storeys": 2, "road_width_ft": np.nan, "district": "Colombo", "city": meta["cities_by_district"]["Colombo"][0]})
    row.update(overrides)
    return pd.DataFrame([row])[FEATURES]


def test_prediction_is_plausible():
    price = float(np.expm1(model.predict(make_row())[0]))
    assert 2e6 < price < 1.5e9


def test_bigger_house_costs_more():
    small = model.predict(make_row(house_size_sqft=1000, bathrooms=1))[0]
    large = model.predict(make_row(house_size_sqft=5000, bathrooms=5))[0]
    assert large > small


def test_range_contains_estimate():
    x = model.named_steps["prep"].transform(make_row())
    params = model.named_steps["model"].pred_dist(x).params
    loc, scale = params["loc"][0], params["scale"][0] * meta["interval_scale"]
    assert scale > 0
    assert loc - 1.28 * scale < loc < loc + 1.28 * scale


def test_quality_gate():
    assert meta["test_mape"] < 40, "Model error too high to deploy"


def test_ranges_are_calibrated():
    if "test_coverage_80" in meta:
        assert 70 <= meta["test_coverage_80"] <= 92, "80% price ranges are badly calibrated"


def test_sklearn_version_matches_training():
    if "sklearn_version" in meta:
        assert meta["sklearn_version"] == sklearn.__version__, (
            f"Model trained with scikit-learn {meta['sklearn_version']}, but {sklearn.__version__} is installed.")