import polars as pl
from src.features.demand import add_demand_features

lf = pl.scan_parquet("data/processed/long_by_store/CA_1.parquet")
result = add_demand_features(lf).collect()

print(f"Shape: {result.shape}")

feature_cols = ["zero_frac_30d", "days_since_last_sale", "days_since_first_sale"]
print("\nNull counts:")
print(result.select(feature_cols).null_count())

print("\nFeature stats:")
print(result.select(feature_cols).describe())

# days_since_last_sale should have no nulls (filled with 9999)
print(f"\ndays_since_last_sale nulls: {result['days_since_last_sale'].null_count()}")
print(f"days_since_last_sale max: {result['days_since_last_sale'].max()}")

# days_since_first_sale should have no nulls
print(f"\ndays_since_first_sale nulls: {result['days_since_first_sale'].null_count()}")

# verify on intermittent item - should show large days_since_last_sale values
print("\nIntermittent item tail:")
print(result.filter(pl.col("id") == "HOUSEHOLD_2_101_CA_1_evaluation")
    .select(["date", "sales", "days_since_last_sale", "days_since_first_sale"])
    .tail(30))

# verify on high volume item - days_since_last_sale should mostly be 0
print("\nHigh volume item tail:")
print(result.filter(pl.col("id") == "FOODS_3_090_CA_1_evaluation")
    .select(["date", "sales", "days_since_last_sale", "days_since_first_sale"])
    .tail(10))