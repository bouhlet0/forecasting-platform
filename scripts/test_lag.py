import polars as pl
from src.features.lag import add_lag_features

# load a single store for testing purposes
lf = pl.scan_parquet("data/processed/long_by_store/CA_1.parquet")

result = add_lag_features(lf).collect()

print(result.shape)
print(result.select(["id", "date", "sales"] + [f"lag_{lw}" for lw in [1,2,3,7,14,28]]).head(10))

# inspect a single known series
series = result.filter(pl.col("id") == "FOODS_3_090_CA_1_evaluation")


# sanity check: lag_1 on row N should equal sales on row N-1
print("\nLag correctness check (lag_1 should equal previous day sales):")
print(series.select(["date", "sales", "lag_1"]).head(10))

# null counts - first N rows per series should be null
print("\nNull counts:")
print(result.select([f"lag_{lw}" for lw in [1,2,3,7,14,28]]).null_count())