from dataclasses import dataclass
from datetime import date, timedelta
import polars as pl
from pathlib import Path

FORECAST_HORIZON = 28
N_FOLDS = 3
GAP_DAYS = 28

FEATURES_DIR = Path("data/processed/features")


@dataclass
class Fold:
    fold_idx: int
    train_start: date
    train_end: date
    val_start: date
    val_end: date

    def __repr__(self) -> str:
        return (
            f"Fold {self.fold_idx}: "
            f"train [{self.train_start} : {self.train_end}] "
            f"gap {GAP_DAYS}d "
            f"val [{self.val_start} : {self.val_end}]"
        )


def make_backtest_folds(
    dates: list[date],
    n_folds: int = N_FOLDS,
    horizon: int = FORECAST_HORIZON,
    gap: int = GAP_DAYS,
) -> list[Fold]:
    dates = sorted(dates)
    n_days = len(dates)

    folds = []
    for i in range(n_folds):
        val_end_idx = n_days - 1 - (n_folds - 1 - i) * horizon
        val_end = dates[val_end_idx]
        val_start = val_end - timedelta(days=horizon - 1)
        train_end = val_start - timedelta(days=gap + 1)
        train_start = dates[0]

        folds.append(Fold(
            fold_idx=i + 1,
            train_start=train_start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
        ))

    return folds


def get_fold_data(
    lf: pl.LazyFrame,
    fold: Fold,
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    train = lf.filter(
        (pl.col("date") >= fold.train_start) &
        (pl.col("date") <= fold.train_end)
    )
    val = lf.filter(
        (pl.col("date") >= fold.val_start) &
        (pl.col("date") <= fold.val_end)
    )
    return train, val