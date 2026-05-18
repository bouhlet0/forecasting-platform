import polars as pl
from src.training.splits import make_backtest_folds, get_fold_data

lf = pl.scan_parquet("data/processed/features/CA_1.parquet")
dates = lf.select("date").collect()["date"].unique().to_list()

folds = make_backtest_folds(dates)

for fold in folds:
    print(fold)
    train_lf, val_lf = get_fold_data(lf, fold)
    train = train_lf.collect()
    val = val_lf.collect()
    print(f"  Train rows: {len(train):,}  Val rows: {len(val):,}")

    assert len(train) > 0, f"Empty train set on fold {fold.fold_idx}"
    assert len(val) > 0, f"Empty val set on fold {fold.fold_idx}"
    assert train["date"].max() < val["date"].min(), f"Train/val overlap on fold {fold.fold_idx}"
    gap_days = (val["date"].min() - train["date"].max()).days
    assert gap_days >= 28, f"Gap too small on fold {fold.fold_idx}: {gap_days} days"
    val_days = (val["date"].max() - val["date"].min()).days + 1
    assert val_days == 28, f"Val window wrong size on fold {fold.fold_idx}: {val_days} days"
    print(f"  Gap: {gap_days} days")