import polars as pl


def add_price_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    
    lf = lf.with_columns([
        pl.col("sell_price").is_not_null().cast(pl.Int8).alias("has_price"),
    ])
    # forward-fill sell_price within each series to handle early nulls
    lf = lf.with_columns([
        pl.col("sell_price")
        .forward_fill()
        .over("id")
        .alias("sell_price"),
    ])

    lf = lf.with_columns([
        # week-over-week price change as ratio
        (
            pl.col("sell_price") / pl.col("sell_price").shift(7).over("id") - 1
        ).alias("price_change_w"),

        # price relative to item's rolling 90-day mean (promotion signal)
        (
            pl.col("sell_price") / pl.col("sell_price").rolling_mean(window_size=90).over("id")
        ).alias("price_rel_mean_90"),

        # price volatility: std of price over last 30 days (much faster to run approximation)
        pl.col("sell_price")
        .rolling_std(window_size=30)
        .over("id")
        .alias("price_std_30"),

        # on promotion flag: price more than 5% below 90-day rolling mean
        (
            pl.col("sell_price") < (pl.col("sell_price").rolling_mean(window_size=90).over("id") * 0.95)
        ).cast(pl.Int8).alias("is_on_promotion"),
    ])

    return lf