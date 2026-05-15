from pathlib import Path
import polars as pl
import gc

from src.features.lag import add_lag_features
from src.features.rolling import add_rolling_features
from src.features.calendar import add_snap_feature, build_calendar_features
from src.features.price import add_price_features
from src.features.demand import add_demand_features
from src.features.hierarchical import add_hierarchical_features, build_group_aggregates

PROCESSED_DIR = Path("data/processed")
AGG_PATH = PROCESSED_DIR / "group_aggregates.parquet"

CALENDAR_FEATURE_COLS = [
    "d",
    "is_national_holiday", "is_christmas", "is_weekend", "is_sporting_event",
    "national_lag_1", "national_lag_2",
    "national_lead_1", "national_lead_2",
]


def build_features_for_store(
    store_id: str,
    calendar_features: pl.LazyFrame,
) -> pl.DataFrame:
    """
    Build full feature set for a single store.
    Args:
        store_id: e.g. "CA_1"
        calendar_features: output of build_calendar_features(),
                           provides event/holiday flags not in long format
    """
    lf = pl.scan_parquet(
        PROCESSED_DIR / "long_by_store" / f"{store_id}.parquet"
    )

    # join calendar-derived event features
    lf = lf.join(
        calendar_features.select(CALENDAR_FEATURE_COLS),
        on="d",
        how="left",
    )

    # state-matched snap (drops snap_CA, snap_TX, snap_WI)
    lf = add_snap_feature(lf)

    # feature modules - order matters, demand needs sales, price needs sell_price
    lf = add_lag_features(lf)
    lf = add_rolling_features(lf)
    lf = add_price_features(lf)
    lf = add_demand_features(lf)
    lf = add_hierarchical_features(lf)

    return lf.collect()


def build_all_features(
    output_dir: Path = PROCESSED_DIR / "features",
) -> None:
    """
    Build and save features for all stores.
    Skips stores already processed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if not AGG_PATH.exists():
        print("Building group aggregates...")
        build_group_aggregates(AGG_PATH)

    calendar_features = build_calendar_features()

    store_ids = [
        "CA_1", "CA_2", "CA_3", "CA_4",
        "TX_1", "TX_2", "TX_3",
        "WI_1", "WI_2", "WI_3",
    ]

    for i, store_id in enumerate(store_ids):
        out_path = output_dir / f"{store_id}.parquet"
        if out_path.exists():
            print(f"[{i+1}/{len(store_ids)}] {store_id} already exists, skipping.")
            continue

        print(f"[{i+1}/{len(store_ids)}] Building features for {store_id}...")
        df = build_features_for_store(store_id, calendar_features)
        df.write_parquet(out_path)
        print(f"    Shape: {df.shape}: saved to {out_path.name}")
        del df
        gc.collect()

    print("Done.")