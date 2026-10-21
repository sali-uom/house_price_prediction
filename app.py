import json
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Sri Lanka House Price Estimator", page_icon="🏠", layout="centered")


@st.cache_resource
def load():
    model = joblib.load("models/house_price_model.pkl")
    with open("models/model_meta.json") as f:
        meta = json.load(f)
    return model, meta


model, meta = load()
FEATURES = meta["features_num"] + meta["features_cat"]

KEYWORD_LABELS = {
    "kw_luxury": "Luxury", "kw_brand_new": "Brand new", "kw_clear_deed": "Clear deed",
    "kw_pool": "Swimming pool", "kw_ac": "Air conditioning", "kw_solar": "Solar power",
    "kw_parking": "Parking / garage", "kw_main_road": "Facing main road",
}

st.title("🏠 Sri Lanka House Price Estimator")
st.caption(f"Model: {meta['model_name']} · trained on ikman.lk house listings (2022–23) · typical error ±{meta['test_mape']}%")

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


def shap_explanation(m, x):
    """Return SHAP values for the tree model inside the saved pipeline."""
    if hasattr(m, "named_estimators_"):          # stacking -> explain its CatBoost component
        m = m.named_estimators_["cat"]
    if hasattr(m, "named_steps"):                # sklearn pipeline -> transform, then explain
        x = m.named_steps["prep"].transform(x)
        m = m.named_steps["model"]
    return shap.TreeExplainer(m)(x)


if st.button("Estimate price", type="primary", use_container_width=True):
    price = float(np.expm1(model.predict(row)[0]))
    margin = meta["test_mape"] / 100
    st.metric("Estimated price", f"Rs {price:,.0f}")
    st.write(f"Likely range: **Rs {price * (1 - margin):,.0f} – Rs {price * (1 + margin):,.0f}**")

    with st.expander("Why this price? (SHAP)", expanded=True):
        values = shap_explanation(model, row)
        fig = plt.figure()
        shap.plots.waterfall(values[0], max_display=10, show=False)
        st.pyplot(fig, clear_figure=True)
        st.caption("Bars show how each feature pushes the estimate above or below the average house (log-price scale).")
