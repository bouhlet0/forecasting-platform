import polars as pl
from src.features.price import add_price_features

lf = pl.scan_parquet("data/processed/long_by_store/CA_1.parquet")

# Safety feature check: price_vs_market_avg requires item_id node.
# Extract it from 'id' string if it doesn't already exist.
if "item_id" not in lf.collect_schema().names():
    lf = lf.with_columns(
        pl.col("id").str.extract(r"^([A-Z]+_\d+)_[A-Z]+_\d+").alias("item_id")
    )

result = add_price_features(lf).collect()
print(f"Shape: {result.shape}")

# Added the 2 new competitor metrics to feature columns
feature_cols = [
    "has_price", "sell_price", "price_change_w",
    "price_rel_mean_90", "price_std_30", "is_on_promotion",
    "price_momentum_w", "price_vs_market_avg"
]
print("\nNull counts:")
print(result.select(feature_cols).null_count())

# price null analysis
print(f"\nSell price nulls after forward fill: {result['sell_price'].null_count()}")

null_summary = result.group_by("id").agg(
    pl.col("sell_price").is_null().mean().alias("null_rate"),
)
fully_null = null_summary.filter(pl.col("null_rate") == 1.0)
partial_null = null_summary.filter(
    (pl.col("null_rate") > 0) & (pl.col("null_rate") < 1.0)
)
print(f"Items with 100% null prices: {len(fully_null)}")
print(f"Items with partial nulls: {len(partial_null)}")

# has_price flag
print("\nhas_price distribution:")
print(result["has_price"].value_counts())

# promotion analysis
print(f"\nPromotion rate: {result['is_on_promotion'].mean()*100:.1f}% of days")

# Validate market comparison distribution
print("\nMarket Position distribution (should hover tightly around 1.0):")
print(result.select("price_vs_market_avg").describe())

# validate on price-varied item
price_varied = (
    result.group_by("id")
    .agg(pl.col("sell_price").n_unique().alias("n_prices"))
    .filter(pl.col("n_prices") > 3)
    .sort("n_prices", descending=True)
    .head(1)["id"][0]
)
print(f"\nMost price-varied item: {price_varied}")
print(result.filter(pl.col("id") == price_varied)
    .select(["date", "sell_price", "price_change_w", "price_rel_mean_90", 
             "price_momentum_w", "price_vs_market_avg", "is_on_promotion"])
    .filter(pl.col("is_on_promotion") == 1)
    .head(10))