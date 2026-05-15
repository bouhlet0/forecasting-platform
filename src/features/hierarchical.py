import polars as pl
from pathlib import Path

PROCESSED_DIR = Path("data/processed")


def build_group_aggregates(
    output_path: Path = PROCESSED_DIR / "group_aggregates.parquet",
) -> None:
    """
    Pre-compute dept and cat daily sales totals across all stores.
    Must be run once before per-store feature generation.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lf = pl.scan_parquet(PROCESSED_DIR / "long_by_store" / "*.parquet")

    group_aggs = (
        lf.group_by(["store_id", "dept_id", "cat_id", "date"])
        .agg(pl.col("sales").sum().alias("dept_sales"))
        .sort(["store_id", "dept_id", "date"])
        .with_columns([
            pl.col("dept_sales").shift(1)
            .over(["store_id", "dept_id"])
            .alias("dept_sales_lag1"),

            pl.col("dept_sales").shift(1)
            .rolling_mean(window_size=28)
            .over(["store_id", "dept_id"])
            .cast(pl.Float32)
            .alias("dept_sales_roll_28"),
        ])
        .rename({"dept_sales": "_dept_sales_raw"})
    )

    cat_aggs = (
        lf.group_by(["store_id", "cat_id", "date"])
        .agg(pl.col("sales").sum().alias("cat_sales"))
        .sort(["store_id", "cat_id", "date"])
        .with_columns([
            pl.col("cat_sales").shift(1)
            .over(["store_id", "cat_id"])
            .alias("cat_sales_lag1"),

            pl.col("cat_sales").shift(1)
            .rolling_mean(window_size=28)
            .over(["store_id", "cat_id"])
            .cast(pl.Float32)
            .alias("cat_sales_roll_28"),
        ])
        .rename({"cat_sales": "_cat_sales_raw"})
    )

    result = group_aggs.join(
        cat_aggs,
        on=["store_id", "cat_id", "date"],
        how="left",
    ).select([
        "store_id", "dept_id", "cat_id", "date",
        "_dept_sales_raw", "dept_sales_lag1", "dept_sales_roll_28",
        "_cat_sales_raw", "cat_sales_lag1", "cat_sales_roll_28",
    ])

    result.sink_parquet(output_path)
    print(f"Written to {output_path}")


def add_hierarchical_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    group_aggs = pl.scan_parquet(
        PROCESSED_DIR / "group_aggregates.parquet"
    )

    lf = lf.join(
        group_aggs,
        on=["store_id", "dept_id", "cat_id", "date"],
        how="left",
    )

    # relative position features
    lf = lf.with_columns([
        (pl.col("sales") / pl.col("_dept_sales_raw").clip(lower_bound=1))
        .cast(pl.Float32)
        .alias("item_share_dept"),

        (pl.col("sales") / pl.col("_cat_sales_raw").clip(lower_bound=1))
        .cast(pl.Float32)
        .alias("item_share_cat"),
    ])

    return lf.drop(["_dept_sales_raw", "_cat_sales_raw"])