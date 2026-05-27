import numpy as np
import polars as pl
import lightgbm as lgb
from pathlib import Path
from datetime import date, timedelta

from src.training.dataset import get_feature_cols

FEATURES_DIR = Path("data/processed/features")

def prepare_recursive_inputs(
    lf: pl.LazyFrame,
    forecast_start: date,
    history_days: int = 118,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    history_start = forecast_start - timedelta(days=history_days)

    historical_records = (
        lf.filter(
            (pl.col("date") >= history_start) &
            (pl.col("date") < forecast_start)
        )
        .select(["id", "date", "sales"])
        .collect()
    )
    
    context = (
        historical_records
        .with_columns(pl.col("sales").cast(pl.Float32))
        .sort(["id", "date"])
        .group_by("id")
        .tail(118)
    )

    day1_parquet_data = (
        lf.filter(pl.col("date") == forecast_start)
        .collect()
    )

    cols_to_drop = ["sales", "date", "wday", "month", "year", "is_weekend"]
    available_drops = [c for c in cols_to_drop if c in day1_parquet_data.columns]
    
    static_features = day1_parquet_data.drop(available_drops)

    from src.features.calendar import build_calendar_features
    forecast_end = forecast_start + timedelta(days=27)
    cal = build_calendar_features()

    calendar_period = (
        cal.filter(
            (pl.col("date") >= forecast_start) &
            (pl.col("date") <= forecast_end)
        )
        .with_columns([
            pl.col("event_name_1").fill_null("0"),
            pl.col("event_type_1").fill_null("0"),
            pl.col("event_name_2").fill_null("0"),
            pl.col("event_type_2").fill_null("0"),
        ])
        .collect()
    )

    return context, static_features, calendar_period


def compute_lag_rolling(
    context: pl.DataFrame,
    forecast_date: date,
) -> pl.DataFrame:
    sorted_ctx = context.sort(["id", "date"])

    features = sorted_ctx.group_by("id").agg([
        pl.col("sales").last().alias("lag_1"),
        pl.col("sales").reverse().get(1).alias("lag_2"),
        pl.col("sales").reverse().get(2).alias("lag_3"),
        pl.col("sales").reverse().get(6).alias("lag_7"),
        pl.col("sales").reverse().get(13).alias("lag_14"),
        pl.col("sales").reverse().get(27).alias("lag_28"),

        pl.col("sales").tail(7).mean().cast(pl.Float32).alias("roll_mean_7"),
        pl.col("sales").tail(28).mean().cast(pl.Float32).alias("roll_mean_28"),
        pl.col("sales").tail(90).mean().cast(pl.Float32).alias("roll_mean_90"),

        pl.col("sales").tail(7).std().cast(pl.Float32).alias("roll_std_7"),
        pl.col("sales").tail(28).std().cast(pl.Float32).alias("roll_std_28"),
        pl.col("sales").tail(90).std().cast(pl.Float32).alias("roll_std_90"),

        # CLEANUP: Dropped roll_sum_7 and roll_sum_28 to perfectly match training features schema
        pl.col("sales").tail(90).sum().cast(pl.Int32).alias("roll_sum_90"),

        pl.col("sales").tail(28).max().cast(pl.Int32).alias("roll_max_28"),
        pl.col("sales").tail(28).median().cast(pl.Float32).alias("roll_median_28"),

        (pl.col("sales").tail(7).mean() /
         pl.col("sales").tail(28).mean().clip(lower_bound=1))
        .cast(pl.Float32).alias("trend_ratio_7_28"),

        (pl.col("sales").tail(28).mean() /
         pl.col("sales").tail(90).mean().clip(lower_bound=1))
        .cast(pl.Float32).alias("trend_ratio_28_90"),

        (pl.col("sales") == 0).tail(30).mean()
        .cast(pl.Float32).alias("zero_frac_30d"),
    ])

    return features.with_columns(pl.lit(forecast_date).alias("date"))


def predict_recursive(
    model: lgb.Booster,
    lf: pl.LazyFrame,
    forecast_start: date,
    forecast_horizon: int = 28,
) -> pl.DataFrame:
    """
    Generate recursive 28-day forecasts.
    """
    context, static_features, calendar_period = prepare_recursive_inputs(
        lf, forecast_start
    )

    feature_cols = get_feature_cols(
        pl.scan_parquet(FEATURES_DIR / "*.parquet").collect_schema().names()
    )

    dsls_tracker = static_features.select(["id", "days_since_last_sale"]).with_columns(
        pl.col("days_since_last_sale").fill_null(0).cast(pl.Int16)
    )
    
    static_features = static_features.drop("days_since_last_sale")

    all_predictions = []

    for step in range(forecast_horizon):
        forecast_date = forecast_start + timedelta(days=step)

        lag_features = compute_lag_rolling(context, forecast_date)

        cal_row = calendar_period.filter(pl.col("date") == forecast_date)

        feature_row = (
            lag_features
            .join(static_features, on="id", how="left")
            .join(dsls_tracker, on="id", how="left")
            .join(cal_row, on="date", how="left")
        )

        feature_row = feature_row.with_columns([
            (pl.col("days_since_first_sale") + step).alias("days_since_first_sale"),
            
            pl.when(pl.col("state_id") == "CA").then(pl.col("snap_CA"))
            .when(pl.col("state_id") == "TX").then(pl.col("snap_TX"))
            .when(pl.col("state_id") == "WI").then(pl.col("snap_WI"))
            .otherwise(0)
            .cast(pl.Int8)
            .alias("snap"),
            
            pl.col("wday").is_in([1, 2]).cast(pl.Int8).alias("_local_is_weekend")
        ]).with_columns([
            (pl.col("snap") * pl.col("_local_is_weekend")).cast(pl.Int8).alias("snap_x_weekend"),
            
            pl.when(pl.col("snap") == 1)
            .then(pl.col("wday"))
            .otherwise(0)
            .cast(pl.Int8)
            .alias("snap_wday_interaction")
        ]).drop(["_local_is_weekend"])

        STR_CAT_FEATURES = ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]
        with pl.StringCache():
            for col in STR_CAT_FEATURES:
                if col in feature_row.columns:
                    feature_row = feature_row.with_columns(
                        pl.col(col).cast(pl.Categorical).to_physical().cast(pl.Int32)
                    )
                    
        if "snap_wday_interaction" in feature_row.columns:
            feature_row = feature_row.with_columns(
                pl.col("snap_wday_interaction").cast(pl.Int32)
            )

        X = np.asarray(
            feature_row.select(feature_cols).to_numpy(),
            dtype=np.float32
        )

        preds = np.clip(model.predict(X), 0, None)

        pred_df = feature_row.select(["id"]).with_columns([
            pl.lit(forecast_date).alias("date"),
            pl.Series("sales_pred", preds.astype(np.float32)),
        ])
        all_predictions.append(pred_df)

        new_rows = pl.DataFrame({
            "id": feature_row["id"].to_list(),
            "date": [forecast_date] * len(preds),
            "sales": preds.astype(np.float32),
        })
        context = (
            pl.concat([context, new_rows])
            .sort(["id", "date"])
            .group_by("id")
            .tail(118)
        )

        dsls_tracker = (
            feature_row.select(["id"])
            .with_columns(pl.Series("preds", preds))
            .join(dsls_tracker, on="id", how="left")
            .select([
                "id",
                pl.when(pl.col("preds") > 0.5)
                .then(pl.lit(0))
                .otherwise(
                    (pl.col("days_since_last_sale") + 1).clip(0, 9999)
                )
                .cast(pl.Int16)
                .alias("days_since_last_sale")
            ])
        )

    return pl.concat(all_predictions).sort(["id", "date"])