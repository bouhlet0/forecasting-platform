import polars as pl

sales = pl.read_parquet("data/parquet/sales_train_validation.parquet")

print(sales.shape)
print(sales.schema)
print(sales.head())