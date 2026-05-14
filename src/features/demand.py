import polars as pl


def add_demand_features(lf: pl.LazyFrame) -> pl.LazyFrame:

    return (
        lf.sort(["id", "date"])
        .with_columns(
            # lag sales by 1 day to avoid leakage
            sales_lag1=pl.col("sales").shift(1).over("id"),

            # stable per-series row number
            row_nr=pl.col("sales").cum_count().over("id"),
        )
        .with_columns(
            had_sale=(pl.col("sales_lag1") > 0),
            zero_frac_30d=(
                (
                    (pl.col("sales_lag1") <= 0)
                    .cast(pl.UInt8)
                    .rolling_mean(window_size=30)
                    .over("id")
                )
                .cast(pl.Float32)
            ),
        )
        .with_columns(
            last_sale_row=(
                pl.when(pl.col("had_sale"))
                .then(pl.col("row_nr"))
                .otherwise(None)
                .forward_fill()
                .over("id")
            ),
        )
        .with_columns(
            # exact days since last sale
            (
                pl.col("row_nr") - pl.col("last_sale_row")
            )
            .cast(pl.Int16)
            .alias("days_since_last_sale")
        )
        .drop([
            "sales_lag1",
            "row_nr",
            "had_sale",
            "last_sale_row",
        ])
    )