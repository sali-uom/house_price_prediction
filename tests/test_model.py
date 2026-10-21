import json

import joblib
import numpy as np
import pandas as pd

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


def test_quality_gate():
    assert meta["test_mape"] < 40, "Model error too high to deploy"
