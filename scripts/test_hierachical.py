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
    "dept_sales_lag1", "dept_sales_roll_28",
    "cat_sales_lag1", "cat_sales_roll_28",
    "item_share_dept", "item_share_cat",
]

print("\nNull counts:")
print(result.select(feature_cols).null_count())

print("\nFeature stats:")
print(result.select(feature_cols).describe())

# verify features differ from each other
print("\nCorrelation between hierarchical features:")
print(result.select(feature_cols).corr())

# verify share features are between 0 and 1
print(f"\nitem_share_dept range: {result['item_share_dept'].min():.4f} - {result['item_share_dept'].max():.4f}")
print(f"item_share_cat range: {result['item_share_cat'].min():.4f} - {result['item_share_cat'].max():.4f}")

# sample on high volume item
print("\nHigh volume item sample:")
print(result.filter(pl.col("id") == "FOODS_3_090_CA_1_evaluation")
    .select(["date", "sales"] + feature_cols)
    .tail(10))

print("\nCorrelation between hierarchical features:")
print(result.select(feature_cols).drop_nulls().corr())