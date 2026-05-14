import polars as pl
from src.features.demand import add_demand_features

lf = pl.scan_parquet("data/processed/long_by_store/CA_1.parquet")
result = add_demand_features(lf).collect()

print(f"Shape: {result.shape}")

feature_cols = ["zero_frac_30d", "days_since_last_sale"]
print("\nNull counts:")
print(result.select(feature_cols).null_count())

print("\nFeature stats:")
print(result.select(feature_cols).describe())

# check on intermittent item
intermittent = result.filter(pl.col("id") == "HOUSEHOLD_2_101_CA_1_evaluation")
print("\nIntermittent item sample:")
print(intermittent.select(["date", "sales", "zero_frac_30d",
                           "days_since_last_sale"]).tail(30))

# check on high volume item
high_vol = result.filter(pl.col("id") == "FOODS_3_090_CA_1_evaluation")
print("\nHigh volume item sample:")
print(high_vol.select(["date", "sales", "zero_frac_30d",
                       "days_since_last_sale"]).tail(30))