import polars as pl


def add_demand_features(lf: pl.LazyFrame) -> pl.LazyFrame:

    return (
        lf.sort(["id", "date"])
        .with_columns(
            sales_lag1=pl.col("sales").shift(1).over("id"),
            row_nr=pl.col("sales").cum_count().over("id"),
        )
        .with_columns(
            had_sale=(pl.col("sales_lag1") > 0),
            zero_frac_30d=(
                (pl.col("sales_lag1") <= 0)
                .cast(pl.UInt8)
                .rolling_mean(window_size=30)
                .over("id")
                .cast(pl.Float32)
            ),
        )
        .with_columns(
            last_sale_row=(
                pl.when(pl.col("had_sale"))
                .then(pl.col("row_nr").cast(pl.Int32))
                .otherwise(None)
                .forward_fill()
                .over("id")
            ),
            first_sale_row=(
                pl.when(pl.col("had_sale"))
                .then(pl.col("row_nr").cast(pl.Int32))
                .otherwise(None)
                .min()
                .over("id")
            ),
        )
        .with_columns(
            (
                pl.col("row_nr").cast(pl.Int32) - pl.col("last_sale_row").cast(pl.Int32)
            )
            .cast(pl.Int16)
            .alias("days_since_last_sale"),

            (
                pl.col("row_nr").cast(pl.Int32) - pl.col("first_sale_row").cast(pl.Int32)
            )
            .clip(lower_bound=0)
            .fill_null(0)
            .cast(pl.Int16)
            .alias("days_since_first_sale"),
        )
        .with_columns(
            pl.col("days_since_last_sale")
            .fill_null(9999)
            .cast(pl.Int16)
            .alias("days_since_last_sale"),
        )
        .drop([
            "sales_lag1",
            "row_nr",
            "had_sale",
            "last_sale_row",
            "first_sale_row",
        ])
    )