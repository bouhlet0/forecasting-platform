import polars as pl
import pandas as pd
from src.training.splits import Fold, get_fold_data

# columns that identify the series but are not model features
ID_COLS = [
    "id", "item_id", "dept_id", "cat_id",
    "store_id", "state_id", "d", "date", "wm_yr_wk",
]

TARGET = "sales"

CAT_FEATURES = [
    "event_name_1", "event_type_1",
    "event_name_2", "event_type_2",
]

# all feature columns = everything except ID_COLS and TARGET
def get_feature_cols(df_cols: list[str]) -> list[str]:
    exclude = set(ID_COLS + [TARGET])
    return [c for c in df_cols if c not in exclude]


def prepare_fold(
    lf: pl.LazyFrame,
    fold: Fold,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train_lf, val_lf = get_fold_data(lf, fold)

    train = train_lf.collect()
    val = val_lf.collect()

    feature_cols = get_feature_cols(train.columns)

    for col in CAT_FEATURES:
        train = train.with_columns(pl.col(col).cast(pl.Categorical))
        val = val.with_columns(pl.col(col).cast(pl.Categorical))

    train_pd = train.to_pandas()
    val_pd = val.to_pandas()

    X_train = train_pd[feature_cols]
    y_train = train_pd[TARGET]
    X_val = val_pd[feature_cols]
    y_val = val_pd[TARGET]

    return X_train, y_train, X_val, y_val