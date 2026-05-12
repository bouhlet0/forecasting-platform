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

    # rolling max only for 28
    rolling_exprs.append(
        pl.col("sales").shift(1).over("id")
        .rolling_max(window_size=28)
        .over("id")
        .alias("roll_max_28")
    )

    # rolling median only for 28
    rolling_exprs.append(
        pl.col("sales").shift(1).over("id")
        .rolling_median(window_size=28)
        .over("id")
        .alias("roll_median_28")
    )

    return lf.with_columns(rolling_exprs)