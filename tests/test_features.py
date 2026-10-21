import numpy as np
import pandas as pd

from src.features import FEATURES, build_features, clean, count_value, extract_road_width, extract_storeys, land_to_perches, to_number


def test_parsers():
    assert to_number("Rs 28,500,000") == 28_500_000
    assert count_value("10+") == 10
    assert land_to_perches("1 acres") == 160
    assert extract_storeys("lovely two story house") == 2
    assert extract_road_width("20 ft road access") == 20
    assert np.isnan(extract_storeys("no info here"))


def test_build_and_clean():
    raw = pd.DataFrame([{
        "category": "Houses For Sale", "price": "Rs 30,000,000", "title": "Luxury house",
        "ad_description": "two story, 20 ft road, swimming pool",
        "properties": "{'Bedrooms': '4', 'Bathrooms': '3', 'House size': '2,500.0 sqft', 'Land size': '10 perches'}",
        "area": "{'id': 1, 'name': 'Colombo'}", "location": "Nugegoda", "user": "u1",
    }, {
        "category": "House Rentals", "price": "Rs 100,000", "title": "Rent", "ad_description": "",
        "properties": "{}", "area": "{'id': 1, 'name': 'Colombo'}", "location": "Nugegoda", "user": "u2",
    }])
    data = clean(build_features(raw), min_city_ads=1)
    assert len(data) == 1
    row = data.iloc[0]
    assert row["kw_luxury"] == 1 and row["kw_pool"] == 1 and row["storeys"] == 2
    assert set(FEATURES) <= set(data.columns)
