import polars as pl

ROLLING_WINDOWS = [7, 28, 90]


def add_rolling_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    rolling_exprs = []

    for window in ROLLING_WINDOWS:
        shifted = pl.col("sales").shift(1).over("id")

        rolling_exprs.extend([
            shifted
            .rolling_mean(window_size=window)
            .over("id")
            .alias(f"roll_mean_{window}"),

            shifted
            .rolling_std(window_size=window)
            .over("id")
            .alias(f"roll_std_{window}"),

            shifted
            .rolling_sum(window_size=window)
            .over("id")
            .alias(f"roll_sum_{window}"),
        ])

    rolling_exprs.append(
        pl.col("sales").shift(1).over("id")
        .rolling_max(window_size=28)
        .over("id")
        .alias("roll_max_28")
    )

    rolling_exprs.append(
        pl.col("sales").shift(1).over("id")
        .rolling_median(window_size=28)
        .over("id")
        .alias("roll_median_28")
    )
    
    shifted_anchor = pl.col("sales").shift(28).over("id")

    rolling_exprs.extend([
        shifted_anchor
        .rolling_mean(window_size=7)
        .over("id")
        .alias("roll_mean_7_lag_28"),

        shifted_anchor
        .rolling_mean(window_size=28)
        .over("id")
        .alias("roll_mean_28_lag_28"),

        shifted_anchor
        .rolling_std(window_size=28)
        .over("id")
        .alias("roll_std_28_lag_28"),
    ])

    lf = lf.with_columns(rolling_exprs)

    lf = lf.with_columns([
        (
            pl.col("roll_mean_7") /
            pl.col("roll_mean_28").clip(lower_bound=1)
        ).cast(pl.Float32).alias("trend_ratio_7_28"),

        (
            pl.col("roll_mean_28") /
            pl.col("roll_mean_90").clip(lower_bound=1)
        ).cast(pl.Float32).alias("trend_ratio_28_90"),
    ])

    return lf