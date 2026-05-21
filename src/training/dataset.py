import polars as pl
import numpy as np
import gc
from src.training.splits import Fold, get_fold_data

ID_COLS = [
    "id", "item_id", "dept_id", "cat_id",
    "store_id", "state_id", "d", "date", "wm_yr_wk",
]

TARGET = "sales"

CAT_FEATURES = [
    "event_name_1", "event_type_1",
    "event_name_2", "event_type_2",
]


def get_feature_cols(df_cols: list[str]) -> list[str]:
    exclude = set(ID_COLS + [TARGET])
    return [c for c in df_cols if c not in exclude]


def prepare_fold(
    lf: pl.LazyFrame,
    fold: Fold,
    sample_frac: float = 0.3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    train_lf, val_lf = get_fold_data(lf, fold)
    train = train_lf.collect()

    if sample_frac < 1.0:
        sampled_ids = (
            train.select("id")
            .unique()
            .sample(fraction=sample_frac, seed=seed, shuffle=True)
        )
        train = train.join(sampled_ids, on="id", how="inner")

    val = val_lf.collect()

    feature_cols = get_feature_cols(train.columns)

    for col in CAT_FEATURES:
        train = train.with_columns(
            pl.col(col).cast(pl.Categorical).to_physical().cast(pl.Int32)
        )
        val = val.with_columns(
            pl.col(col).cast(pl.Categorical).to_physical().cast(pl.Int32)
        )

    train_ids = train.select("id").to_numpy().ravel()
    val_ids = val.select("id").to_numpy().ravel()

    X_train = np.asarray(train.select(feature_cols).to_numpy(), dtype=np.float32)
    y_train = np.asarray(train.select(TARGET).to_numpy().ravel(), dtype=np.float32)

    del train
    gc.collect()

    X_val = np.asarray(val.select(feature_cols).to_numpy(), dtype=np.float32)
    y_val = np.asarray(val.select(TARGET).to_numpy().ravel(), dtype=np.float32)

    del val
    gc.collect()

    cat_col_indices = [feature_cols.index(c) for c in CAT_FEATURES]

    return X_train, y_train, train_ids, X_val, y_val, val_ids, cat_col_indices