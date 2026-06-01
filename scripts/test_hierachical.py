import polars as pl
from src.features.hierarchical import add_hierarchical_features, build_group_aggregates
from pathlib import Path

AGG_PATH = Path("data/processed/group_aggregates.parquet")

if not AGG_PATH.exists():
    print("Building group aggregates...")
    build_group_aggregates(AGG_PATH)

lf = pl.scan_parquet("data/processed/long_by_store/CA_1.parquet")
result = add_hierarchical_features(lf).collect()

print(f"Shape: {result.shape}")

feature_cols = [
    "dept_sales_lag1", "dept_sales_roll_7", "dept_sales_roll_28",
    "cat_sales_lag1", "cat_sales_roll_7", "cat_sales_roll_28",
    "item_share_dept", "item_share_cat",
]

print("\nNull counts:")
print(result.select(feature_cols).null_count())

print("\nFeature stats:")
print(result.select(feature_cols).describe())

# Verify share features are bounded between 0 and 1
print(f"\nitem_share_dept range: {result['item_share_dept'].fill_nan(None).min():.4f} - {result['item_share_dept'].fill_nan(None).max():.4f}")
print(f"item_share_cat range: {result['item_share_cat'].fill_nan(None).min():.4f} - {result['item_share_cat'].fill_nan(None).max():.4f}")
# Sample on a known high-volume item to visually check feature alignment
print("\nHigh volume item sample:")
print(result.filter(pl.col("id") == "FOODS_3_090_CA_1_evaluation")
    .select(["date", "sales"] + feature_cols)
    .tail(10))

# Cleaned up duplicate block: Compute correlation matrix cleanly dropping row-wise nulls
print("\nCorrelation between hierarchical features (dropping nulls):")
print(result.select(feature_cols).drop_nulls().corr())

# --- STRUCTURAL INTEGRITY VALIDATION ---
print("\nSanity Check: Validating that item_share is strictly less than or equal to 1.0...")
out_of_bounds_dept = result.filter(pl.col("item_share_dept") > 1.0001).shape[0]
out_of_bounds_cat = result.filter(pl.col("item_share_cat") > 1.0001).shape[0]

if out_of_bounds_dept == 0 and out_of_bounds_cat == 0:
    print(" Hierarchical tracking check passed! Item sales do not exceed group totals.")
else:
    print(f" Discrepancy found: {out_of_bounds_dept} rows have dept share > 100%. Check sorting order.")