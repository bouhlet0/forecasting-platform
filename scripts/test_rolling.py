import polars as pl
from src.features.rolling import add_rolling_features

lf = pl.scan_parquet("data/processed/long_by_store/CA_1.parquet")
result = add_rolling_features(lf).collect()

feature_cols = [
    "roll_mean_7", "roll_mean_28", "roll_mean_90",
    "roll_std_7", "roll_std_28", "roll_std_90",
    "roll_sum_7", "roll_sum_28", "roll_sum_90",
    "roll_max_28", "roll_median_28",
]

print(f"Shape: {result.shape}")

# check nulls
print("\nNull counts:")
print(result.select(feature_cols).null_count())

# check single known high volume series
series = result.filter(pl.col("id") == "FOODS_3_090_CA_1_evaluation")
print("\nSample series:")
print(series.select(["date", "sales", "roll_mean_7", "roll_mean_28", "roll_sum_7"]).tail(35))

# leakage check: roll_mean_7 on row N should not include sales on row N
# verify by checking row 7 manually: mean of rows 0-6 shifted = rows 1-7
print("\nLeakage check (roll_mean_7 on day 8 should be mean of days 1-7):")
check = series.select(["date", "sales", "roll_mean_7"]).head(10)
print(check)
manual_mean = series["sales"].head(7).mean()
print(f"Manual mean of first 7 days: {manual_mean:.4f}")
print(f"roll_mean_7 on day 8: {check['roll_mean_7'][7]:.4f}")

print("\nTrend ratio null counts:")
print(result.select(["trend_ratio_7_28", "trend_ratio_28_90"]).null_count())

print("\nTrend ratio stats:")
print(result.select(["trend_ratio_7_28", "trend_ratio_28_90"]).describe())

print("\nTrend ratio sample (high volume item):")
print(result.filter(pl.col("id") == "FOODS_3_090_CA_1_evaluation")
    .select(["date", "sales", "roll_mean_7", "roll_mean_28", "roll_mean_90",
             "trend_ratio_7_28", "trend_ratio_28_90"])
    .tail(10))