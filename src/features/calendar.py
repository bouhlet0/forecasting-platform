import polars as pl

CALENDAR_PATH = "data/parquet/calendar.parquet"


def build_calendar_features() -> pl.LazyFrame:
    lf = pl.scan_parquet(CALENDAR_PATH)

    is_national = (
        (pl.col("event_type_1") == "National") |
        (pl.col("event_type_2") == "National")
    ).cast(pl.Int8).fill_null(0)

    is_sporting = (
    (pl.col("event_type_1") == "Sporting") |
    (pl.col("event_type_2") == "Sporting")
    ).cast(pl.Int8).fill_null(0)
    
    return (
        lf.with_columns([
            pl.col("date").str.to_date("%Y-%m-%d"),
            pl.col("event_name_1").fill_null("none"),
            pl.col("event_type_1").fill_null("none"),
            pl.col("event_name_2").fill_null("none"),
            pl.col("event_type_2").fill_null("none"),
            is_national.alias("is_national_holiday"),
            is_sporting.alias("is_sporting_event"),
            pl.col("wday").is_in([1, 2]).cast(pl.Int8).alias("is_weekend"),
        ])
        .with_columns([
            (
                (pl.col("date").dt.month() == 12) &
                (pl.col("date").dt.day() == 25)
            ).cast(pl.Int8).alias("is_christmas"),
        ])
        .sort("date")
        .with_columns([
            pl.col("is_national_holiday").shift(1).fill_null(0).alias("national_lag_1"),
            pl.col("is_national_holiday").shift(2).fill_null(0).alias("national_lag_2"),
            pl.col("is_national_holiday").shift(-1).fill_null(0).alias("national_lead_1"),
            pl.col("is_national_holiday").shift(-2).fill_null(0).alias("national_lead_2"),
        ])
    )


def add_snap_feature(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns([
        pl.when(pl.col("state_id") == "CA").then(pl.col("snap_CA"))
        .when(pl.col("state_id") == "TX").then(pl.col("snap_TX"))
        .when(pl.col("state_id") == "WI").then(pl.col("snap_WI"))
        .otherwise(0)
        .cast(pl.Int8)
        .alias("snap"),
    ]).drop(["snap_CA", "snap_TX", "snap_WI"])