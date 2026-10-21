"""Feature extraction and cleaning shared by training and tests."""
import ast
import re

import numpy as np
import pandas as pd

FEATURES_NUM_BASE = ["bedrooms", "bathrooms", "house_size_sqft", "land_size_perches", "storeys", "road_width_ft"]
FEATURES_CAT = ["district", "city"]

KEYWORDS = {
    "kw_luxury": r"luxur",
    "kw_brand_new": r"brand\s*new|newly built",
    "kw_clear_deed": r"clear\s*(?:deed|title)",
    "kw_pool": r"swimming|pool",
    "kw_ac": r"a/c|air\s*condition|\bac\b",
    "kw_solar": r"solar",
    "kw_parking": r"parking|garage|car\s*park|carport",
    "kw_main_road": r"main\s*road",
}
FEATURES_NUM = FEATURES_NUM_BASE + list(KEYWORDS)
FEATURES = FEATURES_NUM + FEATURES_CAT

BOUNDS = {
    "price": (2_000_000, 1_500_000_000),
    "house_size_sqft": (300, 15_000),
    "land_size_perches": (2, 160),
    "bedrooms": (1, 10),
    "bathrooms": (1, 10),
}
WORD_NUM = {"one": 1, "single": 1, "two": 2, "double": 2, "three": 3, "four": 4}


def to_number(s):
    if pd.isna(s):
        return np.nan
    digits = re.sub(r"[^0-9.]", "", str(s))
    return float(digits) if digits else np.nan


def count_value(s):
    if pd.isna(s):
        return np.nan
    s = str(s).replace("+", "")
    return float(s) if s.replace(".", "").isdigit() else np.nan


def land_to_perches(s):
    if pd.isna(s):
        return np.nan
    value = to_number(s)
    return value * 160 if "acre" in str(s).lower() else value


def extract_storeys(text):
    m = re.search(r"\b(\d|one|single|two|double|three|four)\s*-?\s*(?:story|storey|stories|storeys|floor)", text)
    if not m:
        return np.nan
    v = m.group(1)
    v = int(v) if v.isdigit() else WORD_NUM[v]
    return v if 1 <= v <= 5 else np.nan


def extract_road_width(text):
    m = re.search(r"(\d{1,2})\s*(?:ft|feet|foot)\s*(?:wide\s*)?(?:road|rd|carpet|access)", text)
    if not m:
        return np.nan
    v = int(m.group(1))
    return v if 6 <= v <= 60 else np.nan


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw ikman.lk ads -> one row per house for sale with model features."""
    df = raw[raw["category"] == "Houses For Sale"].copy()
    props = df["properties"].apply(ast.literal_eval).apply(pd.Series)
    text = (df["title"].fillna("") + " " + df["ad_description"].fillna("")).str.lower()

    data = pd.DataFrame({
        "price": df["price"].apply(to_number),
        "bedrooms": props["Bedrooms"].apply(count_value),
        "bathrooms": props["Bathrooms"].apply(count_value),
        "house_size_sqft": props["House size"].apply(to_number),
        "land_size_perches": props["Land size"].apply(land_to_perches),
        "storeys": text.apply(extract_storeys),
        "road_width_ft": text.apply(extract_road_width),
        "district": df["area"].apply(lambda s: ast.literal_eval(s)["name"]),
        "city": df["location"],
        "user": df["user"],
        "title": df["title"],
    }, index=df.index)
    for col, pattern in KEYWORDS.items():
        data[col] = text.str.contains(pattern, regex=True).astype(int)
    return data


def clean(data: pd.DataFrame, min_city_ads: int = 15) -> pd.DataFrame:
    """Drop missing prices, duplicates and impossible values; group rare cities."""
    data = data.dropna(subset=["price"]).drop_duplicates(subset=["title", "price", "city"])
    for col, (lo, hi) in BOUNDS.items():
        data = data[data[col].between(lo, hi)]
    counts = data["city"].value_counts()
    return data.assign(city=data["city"].where(data["city"].map(counts) >= min_city_ads, "Other"))
