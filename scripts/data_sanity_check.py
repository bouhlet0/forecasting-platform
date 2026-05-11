import polars as pl

calendar = pl.read_parquet("data/parquet/calendar.parquet")
prices = pl.read_parquet("data/parquet/sell_prices.parquet")

print("=== CALENDAR ===")
print(calendar.shape)
print(calendar.schema)
print(calendar.head(3))

print("\n=== PRICES ===")
print(prices.shape)
print(prices.schema)
print(prices.head(3))