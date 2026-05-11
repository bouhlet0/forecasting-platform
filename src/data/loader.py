from pathlib import Path
import polars as pl
import gc

PARQUET_DIR = Path("data/parquet")
PROCESSED_DIR = Path("data/processed")


def build_long_chunked(
    output_dir: Path = PROCESSED_DIR / "long_by_store",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    sales = pl.read_parquet(PARQUET_DIR / "sales_train_evaluation.parquet")
    calendar = (
        pl.read_parquet(PARQUET_DIR / "calendar.parquet")
        .with_columns([
            pl.col("date").str.to_date("%Y-%m-%d"),
            pl.col("event_name_1").fill_null("none"),
            pl.col("event_type_1").fill_null("none"),
            pl.col("event_name_2").fill_null("none"),
            pl.col("event_type_2").fill_null("none"),
        ])
        .select([
            "d", "date", "wm_yr_wk", "wday", "month", "year",
            "event_name_1", "event_type_1",
            "event_name_2", "event_type_2",
            "snap_CA", "snap_TX", "snap_WI",
        ])
    )
    prices = pl.read_parquet(PARQUET_DIR / "sell_prices.parquet")

    store_ids = sorted(sales["store_id"].unique().to_list())
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

    for i, store_id in enumerate(store_ids):
        out_file = output_dir / f"{store_id}.parquet"
        if out_file.exists():
            print(f"[{i+1}/{len(store_ids)}] {store_id} already exists, skipping.")
            continue

        print(f"[{i+1}/{len(store_ids)}] Processing {store_id}...")

        chunk = (
            sales
            .filter(pl.col("store_id") == store_id)
            .unpivot(index=id_cols, variable_name="d", value_name="sales")
            .join(calendar, on="d", how="left")
            .join(
                prices.filter(pl.col("store_id") == store_id),
                on=["store_id", "item_id", "wm_yr_wk"],
                how="left",
            )
            .with_columns([
                pl.col("sales").cast(pl.Int32),
                pl.col("wday").cast(pl.Int8),
                pl.col("month").cast(pl.Int8),
                pl.col("year").cast(pl.Int16),
                pl.col("snap_CA").cast(pl.Int8),
                pl.col("snap_TX").cast(pl.Int8),
                pl.col("snap_WI").cast(pl.Int8),
                pl.col("sell_price").cast(pl.Float32),
            ])
            .sort(["id", "date"])
        )

        chunk.write_parquet(out_file)
        del chunk
        gc.collect()
        print(f"    Saved {out_file.name}")

    print(f"Done. Files in {output_dir}")