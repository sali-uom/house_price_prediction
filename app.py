import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from ngboost import NGBRegressor
from scipy.stats import norm

# Lets scikit-learn pipelines recognise a fitted NGBoost model
NGBRegressor.__sklearn_is_fitted__ = lambda self: bool(getattr(self, "base_models", None))

st.set_page_config(page_title="Sri Lanka House Price Estimator", page_icon="🏠", layout="centered")


@st.cache_resource
def load():
    model = joblib.load("models/house_price_model.pkl")
    with open("models/model_meta.json") as f:
        meta = json.load(f)
    return model, meta


model, meta = load()
FEATURES = meta["features_num"] + meta["features_cat"]
prep, ngb = model.named_steps["prep"], model.named_steps["model"]

KEYWORD_LABELS = {
    "kw_luxury": "Luxury", "kw_brand_new": "Brand new", "kw_clear_deed": "Clear deed",
    "kw_pool": "Swimming pool", "kw_ac": "Air conditioning", "kw_solar": "Solar power",
    "kw_parking": "Parking / garage", "kw_main_road": "Facing main road",
}

st.title("🏠 Sri Lanka House Price Estimator")
st.caption("NGBoost model trained on ikman.lk house listings (2022 to 2023). Every estimate comes with its own price range.")

col1, col2 = st.columns(2)
district = col1.selectbox("District", meta["districts"], index=meta["districts"].index("Colombo"))
city = col2.selectbox("City", meta["cities_by_district"][district])

col1, col2 = st.columns(2)
bedrooms = col1.number_input("Bedrooms", 1, 10, 3)
bathrooms = col2.number_input("Bathrooms", 1, 10, 2)
house_size = col1.number_input("House size (sqft)", 300, 15000, 2000, step=100)
land_size = col2.number_input("Land size (perches)", 2.0, 160.0, 10.0, step=0.5)
storeys = col1.selectbox("Storeys", ["Unknown", 1, 2, 3, 4])
road_width = col2.number_input("Road width (ft, 0 = unknown)", 0, 60, 0)
level = st.select_slider("Range confidence", options=[0.8, 0.9, 0.95], value=0.8, format_func=lambda v: f"{v:.0%}")

st.write("**Features**")
cols = st.columns(4)
flags = {kw: int(cols[i % 4].checkbox(label)) for i, (kw, label) in enumerate(KEYWORD_LABELS.items())}

row = pd.DataFrame([{
    "bedrooms": bedrooms, "bathrooms": bathrooms,
    "house_size_sqft": house_size, "land_size_perches": land_size,
    "storeys": np.nan if storeys == "Unknown" else storeys,
    "road_width_ft": np.nan if road_width == 0 else road_width,
    **flags, "district": district, "city": city,
}])[FEATURES]

if st.button("Estimate price", type="primary", use_container_width=True):
    x = prep.transform(row)
    params = ngb.pred_dist(x).params
    loc, scale = float(params["loc"][0]), float(params["scale"][0]) * meta["interval_scale"]
    z = norm.ppf(0.5 + level / 2) * scale
    price, low, high = np.expm1(loc), np.expm1(loc - z), np.expm1(loc + z)

    st.metric("Estimated price", f"Rs {price:,.0f}")
    st.write(f"{level:.0%} price range: **Rs {low:,.0f} to Rs {high:,.0f}**")
    width = (high - low) / price
    if width > 1.5:
        st.warning("The model is unsure about this house, usually because few similar houses were in the training data.")
    else:
        st.info("The range reflects how sure the model is about this specific house.")

    with st.expander("Why this price? (SHAP)", expanded=True):
        values = shap.TreeExplainer(ngb, model_output=0)(x)
        fig = plt.figure()
        shap.plots.waterfall(values[0], max_display=10, show=False)
        st.pyplot(fig, clear_figure=True)
        st.caption("Bars show how each feature pushes the estimate above or below the average house (log price scale).")

    with st.expander("Why this range? (SHAP for uncertainty)"):
        values = shap.TreeExplainer(ngb, model_output=1)(x)
        fig = plt.figure()
        shap.plots.waterfall(values[0], max_display=10, show=False)
        st.pyplot(fig, clear_figure=True)
        st.caption("Positive bars make the model less certain and the range wider; negative bars make it more certain.")