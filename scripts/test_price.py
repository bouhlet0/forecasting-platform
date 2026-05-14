import polars as pl
from src.features.price import add_price_features

lf = pl.scan_parquet("data/processed/long_by_store/CA_1.parquet")
result = add_price_features(lf).collect()
print(f"Shape: {result.shape}")

# null counts
feature_cols = ["has_price", "sell_price", "price_change_w",
                "price_rel_mean_90", "price_std_30", "is_on_promotion"]
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
print("\nTop 10 highest null rate items:")
print(
    result.group_by("id").agg(
        pl.col("sell_price").is_null().sum().alias("null_count"),
        pl.col("sell_price").is_null().mean().alias("null_rate"),
        pl.len().alias("total_days"),
    )
    .filter(pl.col("null_count") > 0)
    .sort("null_rate", descending=True)
    .head(10)
)

# has_price flag
print("\nhas_price distribution:")
print(result["has_price"].value_counts())

# promotion analysis
print(f"\nPromotion rate: {result['is_on_promotion'].mean()*100:.1f}% of days")
print("\nPromotion rate by threshold (price_rel_mean_90):")
for threshold in [0.99, 0.97, 0.95, 0.90]:
    rate = (result["price_rel_mean_90"] < threshold).mean()
    print(f"  {threshold}: {rate*100:.1f}%")

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
    .select(["date", "sell_price", "price_change_w",
             "price_rel_mean_90", "is_on_promotion"])
    .filter(pl.col("is_on_promotion") == 1)
    .head(10))

# validate has_price on late-introduced item
high_null_item = "FOODS_3_595_CA_1_evaluation"
print(f"\nhas_price=0 sample ({high_null_item}):")
print(result.filter(pl.col("id") == high_null_item)
    .select(["date", "sell_price", "has_price"])
    .filter(pl.col("has_price") == 0)
    .sample(10, seed=42))
print(f"\nhas_price=1 sample ({high_null_item}):")
print(result.filter(pl.col("id") == high_null_item)
    .select(["date", "sell_price", "has_price"])
    .filter(pl.col("has_price") == 1)
    .sample(10, seed=42))