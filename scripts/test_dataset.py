import polars as pl
from src.training.splits import make_backtest_folds
from src.training.dataset import prepare_fold, CAT_FEATURES, ID_COLS, TARGET

lf = pl.scan_parquet("data/processed/features/CA_1.parquet")
dates = lf.select("date").collect()["date"].unique().to_list()
folds = make_backtest_folds(dates)

fold = folds[0]
X_train, y_train, X_val, y_val = prepare_fold(lf, fold)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_val shape:   {X_val.shape}")
print(f"y_val shape:   {y_val.shape}")

print(f"\nFeature count: {X_train.shape[1]}")
print(f"Categorical features: {CAT_FEATURES}")

# check categoricals are correctly typed
for col in CAT_FEATURES:
    assert X_train[col].dtype.name == "category", f"{col} is not category dtype in X_train"
    assert X_val[col].dtype.name == "category", f"{col} is not category dtype in X_val"
print(" Categorical dtypes correct")

# check no target leakage
assert TARGET not in X_train.columns, "Target in X_train"
assert TARGET not in X_val.columns, "Target in X_val"
print(" No target leakage")

# check no id columns in features
for col in ID_COLS:
    assert col not in X_train.columns, f"ID column {col} in X_train"
print(" No ID columns in features")

print("\nSample X_train:")
print(X_train.head(3))