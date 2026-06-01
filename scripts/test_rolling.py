import polars as pl
from src.features.rolling import add_rolling_features

lf = pl.scan_parquet("data/processed/long_by_store/CA_1.parquet")
result = add_rolling_features(lf).collect()

feature_cols = [
    "roll_mean_7", "roll_mean_28", "roll_mean_90",
    "roll_std_7", "roll_std_28", "roll_std_90",
    "roll_sum_7", "roll_sum_28", "roll_sum_90",
    "roll_max_28", "roll_median_28",
    "roll_mean_7_lag_28", "roll_mean_28_lag_28", "roll_std_28_lag_28"
]

print(f"Shape: {result.shape}")

# check nulls (Expect high leading null counts for lag_28 due to structural 28 + window offset)
print("\nNull counts:")
print(result.select(feature_cols).null_count())

# check single known high volume series
series = result.filter(pl.col("id") == "FOODS_3_090_CA_1_evaluation").sort("date")
print("\nSample series tail:")
print(series.select(["date", "sales", "roll_mean_7", "roll_mean_7_lag_28", "roll_mean_28_lag_28"]).tail(15))

# leakage check: roll_mean_7 on row N should not include sales on row N
print("\nLeakage check (roll_mean_7 on day 8 should be mean of days 1-7):")
check = series.select(["date", "sales", "roll_mean_7"]).head(10)
print(check)
manual_mean = series["sales"].head(7).mean()
print(f"Manual mean of first 7 days: {manual_mean:.4f}")
print(f"roll_mean_7 on day 8: {check['roll_mean_7'][7]:.4f}")

print("\nAnchor Pillar Alignment Check:")

# move our inspection window deep enough into the timeline to clear leading nulls
base_idx = 30 
day_30_standard = series["roll_mean_7"][base_idx]
day_58_anchor = series["roll_mean_7_lag_28"][base_idx + 28]

if day_30_standard is None or day_58_anchor is None:
    print("Inspection points still contain null values. Check series length or introduction dates.")
else:
    print(f"Standard roll_mean_7 on Day 31: {day_30_standard:.4f}")
    print(f"Anchor roll_mean_7_lag_28 on Day 59 (28 days later): {day_58_anchor:.4f}")
    
    if abs(day_30_standard - day_58_anchor) < 1e-5:
        print("Anchor Pillar tracking check passed! Shift offsets align perfectly.")
    else:
        print("Alignment discrepancy detected. Check your shift bounds.")

print("\nTrend ratio null counts:")
print(result.select(["trend_ratio_7_28", "trend_ratio_28_90"]).null_count())