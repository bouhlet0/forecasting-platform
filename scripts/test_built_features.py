import polars as pl
from pathlib import Path

FEATURES_DIR = Path("data/processed/features")

store_ids = [
    "CA_1", "CA_2", "CA_3", "CA_4",
    "TX_1", "TX_2", "TX_3",
    "WI_1", "WI_2", "WI_3",
]

# check all files exist
for store_id in store_ids:
    path = FEATURES_DIR / f"{store_id}.parquet"
    assert path.exists(), f"Missing feature file: {store_id}"
print("All feature files present.")

# scan all stores lazily
lf = pl.scan_parquet(FEATURES_DIR / "*.parquet")

# row count
total_rows = lf.select(pl.len()).collect().item()
print(f"Total rows: {total_rows:,}")
assert total_rows == 59181090, f"Expected 59,181,090 rows, got {total_rows:,}"

# rows per store should be equal
store_counts = lf.group_by("store_id").len().sort("store_id").collect()
print("\nRows per store:")
print(store_counts)
assert (store_counts["len"] == 5918109).all(), "Unequal row counts across stores"

# no nulls in critical columns across all stores
critical_cols = [
    "sales", "snap", "days_since_last_sale", "days_since_first_sale",
    "has_price", "item_share_dept", "item_share_cat",
]
null_counts = lf.select([
    pl.col(c).is_null().sum().alias(c) for c in critical_cols
]).collect()
print("\nNull counts in critical columns (all stores):")
print(null_counts)
for col in critical_cols:
    assert null_counts[col][0] == 0, f"{col} has nulls across full dataset"

# date range
date_range = lf.select([
    pl.col("date").min().alias("min_date"),
    pl.col("date").max().alias("max_date"),
]).collect()
print(f"\nDate range: {date_range['min_date'][0]} to {date_range['max_date'][0]}")
assert str(date_range["min_date"][0]) == "2011-01-29", "Unexpected min date"
assert str(date_range["max_date"][0]) == "2016-05-22", "Unexpected max date"

print("\nAll post-build checks passed.")