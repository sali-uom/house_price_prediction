import pandas as pd

chunks = pd.read_csv("../data/properties.csv", chunksize=50_000)
houses = pd.concat(
    c[c.apply(lambda r: r.astype(str).str.contains("House", case=False).any(), axis=1)]
    for c in chunks
)
houses.sample(n=min(20_000, len(houses)), random_state=42).to_csv("../data/houses_sample.csv", index=False)