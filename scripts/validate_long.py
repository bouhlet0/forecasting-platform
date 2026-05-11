import polars as pl
from pathlib import Path

OUTPUT_DIR = Path("data/processed/long_by_store")

df = pl.scan_parquet(OUTPUT_DIR / "*.parquet")

# 1. row count per store should be equal
print("=== Rows per store ===")
print(df.group_by("store_id").len().sort("store_id").collect())

# 2. date range
print("\n=== Date range ===")
print(df.select([
    pl.col("date").min().alias("min_date"),
    pl.col("date").max().alias("max_date"),
]).collect())

# 3. null counts
print("\n=== Null counts ===")
print(df.select(pl.all().is_null().sum()).collect())

# 4. sales sanity - no negative values
print("\n=== Negative sales ===")
print(df.filter(pl.col("sales") < 0).select(pl.len()).collect())

# 5. sell_price nulls by store (expected for early periods)
print("\n=== Sell price nulls per store ===")
print(
    df.filter(pl.col("sell_price").is_null())
    .group_by("store_id")
    .len()
    .sort("store_id")
    .collect()
)