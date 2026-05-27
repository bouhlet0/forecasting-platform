import polars as pl
from pathlib import Path

CALENDAR_PATH = Path(__file__).resolve().parents[2] / "data" / "parquet" / "calendar.parquet"


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
            pl.col("event_name_1").fill_null("0"),
            pl.col("event_type_1").fill_null("0"),
            pl.col("event_name_2").fill_null("0"),
            pl.col("event_type_2").fill_null("0"),
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
            # Original National Holiday leads/lags
            pl.col("is_national_holiday").shift(1).fill_null(0).alias("national_lag_1"),
            pl.col("is_national_holiday").shift(2).fill_null(0).alias("national_lag_2"),
            pl.col("is_national_holiday").shift(-1).fill_null(0).alias("national_lead_1"),
            pl.col("is_national_holiday").shift(-2).fill_null(0).alias("national_lead_2"),
            
            # Universal Macro Event Window (Any holiday type, e.g., Sporting, Cultural)
            pl.when(
                (pl.col("event_name_1") != "0") & (pl.col("event_name_1").is_not_null())
            ).then(1).otherwise(0).cast(pl.Int8).alias("is_event_day")
        ])
        .with_columns([
            # Deeper Lead/Lag effects for the macro event day indicator
            pl.col("is_event_day").shift(-1).fill_null(0).alias("event_lead_1"),
            pl.col("is_event_day").shift(-2).fill_null(0).alias("event_lead_2"),
            pl.col("is_event_day").shift(-3).fill_null(0).alias("event_lead_3"),
            pl.col("is_event_day").shift(1).fill_null(0).alias("event_lag_1"),
            pl.col("is_event_day").shift(2).fill_null(0).alias("event_lag_2"),
        ])
    )


def add_snap_feature(lf: pl.LazyFrame) -> pl.LazyFrame:
    return (
        lf.with_columns([
            pl.when(pl.col("state_id") == "CA").then(pl.col("snap_CA"))
            .when(pl.col("state_id") == "TX").then(pl.col("snap_TX"))
            .when(pl.col("state_id") == "WI").then(pl.col("snap_WI"))
            .otherwise(0)
            .cast(pl.Int8)
            .alias("snap"),
            
            # Safe local fallback for weekend check
            pl.col("wday").is_in([1, 2]).cast(pl.Int8).alias("_local_is_weekend")
        ])
        .with_columns([
            # NEW: Cross-product interactions using our safe fallback indicator
            (pl.col("snap") * pl.col("_local_is_weekend")).cast(pl.Int8).alias("snap_x_weekend"),
            
            pl.when(pl.col("snap") == 1)
            .then(pl.col("wday"))
            .otherwise(0)
            .cast(pl.Int8)
            .alias("snap_wday_interaction")
        ])
        .drop(["snap_CA", "snap_TX", "snap_WI", "_local_is_weekend"])
    )