import polars as pl
import numpy as np
from src.training.splits import make_backtest_folds
from src.training.dataset import prepare_fold, CAT_FEATURES, ID_COLS, TARGET, get_feature_cols

lf = pl.scan_parquet("data/processed/features/CA_1.parquet")
dates = lf.select("date").collect()["date"].unique().sort().to_list()
folds = make_backtest_folds(dates)

fold = folds[0]
X_train, y_train, train_ids, X_val, y_val, val_ids, cat_col_indices = prepare_fold(lf, fold)

print("--- Array Structural Metrics ---")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_val shape:   {X_val.shape}")
print(f"y_val shape:   {y_val.shape}")

print(f"\nFeature count: {X_train.shape[1]}")
print(f"Categorical feature names: {CAT_FEATURES}")
print(f"Categorical column matrix indices: {cat_col_indices}")

# Re-verify columns using the helper logic since X is now a NumPy array
raw_columns = lf.collect_schema().names()
feature_cols = get_feature_cols(raw_columns)

# Check categorical index integrity
print("\n--- Integrity Validations ---")
for idx, name in zip(cat_col_indices, CAT_FEATURES):
    assert feature_cols[idx] == name, f"Index mismatch! Index {idx} points to {feature_cols[idx]}, expected {name}"
print(" Categorical physical index alignment verified.")

# Check for target leakage
assert TARGET not in feature_cols, f" Leakage Warning: Target column '{TARGET}' found inside feature columns!"
print(" No target leakage found in feature list.")

# Check for id columns in features
for col in ID_COLS:
    assert col not in feature_cols, f" Structural Error: ID column '{col}' leaked into features!"
print(" No ID tracking columns found in feature list.")

# Check that data types are fully numeric arrays (LightGBM ready)
assert X_train.dtype == np.float32, "X_train must be float32 array"
assert X_val.dtype == np.float32, "X_val must be float32 array"
print(" Array memory dtypes correct (np.float32).")

print("\n Matrix Sample (First row feature values):")
print(X_train[0, :])