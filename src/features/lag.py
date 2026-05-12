import polars as pl

LAG_DAYS = [1, 2, 3, 7, 14, 28]

def add_lag_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    lag_exprs = [
        pl.col("sales")
        .shift(lag)
        .over("id")
        .alias(f"lag_{lag}")
        for lag in LAG_DAYS
    ]
    return lf.with_columns(lag_exprs)