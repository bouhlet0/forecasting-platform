import polars as pl
import lightgbm as lgb
import numpy as np
from pathlib import Path

from src.training.splits import make_backtest_folds, get_fold_data
from src.training.predict import predict_recursive
from src.training.metrics import compute_fold_weights_and_scales, calculate_sampled_wrmsse
from src.training.dataset import get_feature_cols, TARGET
import src.training.predict as pred_mod

FEATURES_DIR = Path("data/processed/features")
MODELS_DIR = Path("models")
POST_PROCESSING_MULTIPLIER = 0.975

# load model and data
model = lgb.Booster(model_file=str(MODELS_DIR / "lgbm_fold_3.txt"))
lf = pl.scan_parquet(FEATURES_DIR / "*.parquet")

dates = lf.select("date").collect()["date"].unique().sort().to_list()
folds = make_backtest_folds(dates)
fold3 = folds[2]

print(f"Fold 3: {fold3}")
print(f"Forecast start: {fold3.val_start}")

# generate recursive predictions
print("\nGenerating recursive predictions...")
preds_df = predict_recursive(
    model=model,
    lf=lf,
    forecast_start=fold3.val_start,
    forecast_horizon=28,
)

# Scale down the recursive predictions by your winning multiplier
preds_df = preds_df.with_columns(
    (pl.col("sales_pred") * POST_PROCESSING_MULTIPLIER).alias("sales_pred")
)

print(f"Predictions shape: {preds_df.shape}")
print(f"Date range: {preds_df['date'].min()} to {preds_df['date'].max()}")
print(f"Series count: {preds_df['id'].n_unique()}")
print(f"Pred range: [{preds_df['sales_pred'].min():.4f}, {preds_df['sales_pred'].max():.4f}]")

# get actual values for comparison
_, val_lf = get_fold_data(lf, fold3)
val_df = val_lf.collect()

# join predictions with actuals
comparison = val_df.select(["id", "date", "sales"]).join(
    preds_df,
    on=["id", "date"],
    how="inner",
)

print(f"\nComparison rows: {len(comparison):,}")

y_true = comparison["sales"].to_numpy()
y_pred_rec = comparison["sales_pred"].to_numpy()
val_ids = comparison["id"].to_numpy()

# compute WRMSSE
df_weight_ref = (
    lf.filter(pl.col("date") <= fold3.train_end)
    .select(["id", "date", "sales", "sell_price"])
    .collect()
)
m5_ref = compute_fold_weights_and_scales(df_weight_ref, fold3.train_end)

recursive_wrmsse = calculate_sampled_wrmsse(y_true, y_pred_rec, val_ids, m5_ref)
rmse = np.sqrt(np.mean((y_true - y_pred_rec) ** 2))
mae = np.mean(np.abs(y_true - y_pred_rec))

print("\n--- Recursive Forecast Results ---")
print(f"RMSE  : {rmse:.4f}")
print(f"MAE   : {mae:.4f}")
print(f"WRMSSE: {recursive_wrmsse:.4f}")


feature_cols = get_feature_cols(val_df.columns)

STR_CAT_FEATURES = ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]

with pl.StringCache():
    for col in STR_CAT_FEATURES:
        if col in val_df.columns:
            val_df = val_df.with_columns(
                pl.col(col).cast(pl.Categorical).to_physical().cast(pl.Int32)
            )

if "snap_wday_interaction" in val_df.columns:
    val_df = val_df.with_columns(
        pl.col("snap_wday_interaction").cast(pl.Int32)
    )
    
X_val = np.asarray(val_df.select(feature_cols).to_numpy(), dtype=np.float32)
y_val = val_df.select(TARGET).to_numpy().ravel()
val_ids_direct = val_df.select("id").to_numpy().ravel()

y_pred_direct = np.clip(model.predict(X_val), 0, None) * POST_PROCESSING_MULTIPLIER
direct_wrmsse = calculate_sampled_wrmsse(y_val, y_pred_direct, val_ids_direct, m5_ref)

print("\n--- Direct Forecast (baseline) ---")
print(f"WRMSSE: {direct_wrmsse:.4f}")
print(f"\nRecursive vs Direct: {recursive_wrmsse - direct_wrmsse:+.4f}")
print("(negative means recursive is better)")

# Create a strict Day 1 evaluation mask
day1_date = fold3.val_start  # 2016-04-25

# Filter Recursive Results to Day 1
comp_day1 = comparison.filter(pl.col("date") == day1_date)
y_true_d1 = comp_day1["sales"].to_numpy()
y_pred_rec_d1 = comp_day1["sales_pred"].to_numpy()

rec_rmse_d1 = np.sqrt(np.mean((y_true_d1 - y_pred_rec_d1) ** 2))
print("\n--- Day 1 Strict Comparison ---")
print(f"Recursive Day 1 RMSE: {rec_rmse_d1:.4f}")

# Filter Direct Baseline Data to Day 1 
val_df_d1 = val_df.filter(pl.col("date") == day1_date)
X_val_d1 = np.asarray(val_df_d1.select(feature_cols).to_numpy(), dtype=np.float32)
y_true_direct_d1 = val_df_d1.select(TARGET).to_numpy().ravel()

# Apply the multiplier to the Day 1 direct baseline prediction
y_pred_direct_d1 = np.clip(model.predict(X_val_d1), 0, None) * POST_PROCESSING_MULTIPLIER
direct_rmse_d1 = np.sqrt(np.mean((y_true_direct_d1 - y_pred_direct_d1) ** 2))
print(f"Direct Baseline Day 1 RMSE: {direct_rmse_d1:.4f}")

print(f"Day 1 Diff (RMSE): {rec_rmse_d1 - direct_rmse_d1:+.4f}")

# --- TRACKING DIAGNOSTIC ---
sample_id = val_df_d1["id"].to_list()[0]

feat_direct = val_df_d1.filter(pl.col("id") == sample_id).select(feature_cols)

context, static_features, calendar_period = pred_mod.prepare_recursive_inputs(lf, fold3.val_start)
lag_features = pred_mod.compute_lag_rolling(context, fold3.val_start)
cal_row = calendar_period.filter(pl.col("date") == fold3.val_start)

dsls_tracker = context.sort(["id", "date"]).group_by("id").agg([
    pl.col("sales").reverse().arg_max().cast(pl.Int16).alias("days_since_last_sale")
])

feat_recursive = (
    lag_features
    .join(static_features, on="id", how="left")
    .join(dsls_tracker, on="id", how="left")
    .join(cal_row, on="date", how="left")
)
feat_recursive = feat_recursive.with_columns([
    pl.when(pl.col("state_id") == "CA").then(pl.col("snap_CA"))
    .when(pl.col("state_id") == "TX").then(pl.col("snap_TX"))
    .when(pl.col("state_id") == "WI").then(pl.col("snap_WI"))
    .otherwise(0).cast(pl.Int8).alias("snap"),
    pl.col("wday").is_in([1, 2]).cast(pl.Int8).alias("_local_is_weekend")
]).with_columns([
    (pl.col("snap") * pl.col("_local_is_weekend")).cast(pl.Int8).alias("snap_x_weekend"),
    pl.when(pl.col("snap") == 1).then(pl.col("wday")).otherwise(0).cast(pl.Int8).alias("snap_wday_interaction")
]).drop(["_local_is_weekend"])

# Ensure types match the direct validation dataframe for the check loop
if "snap_wday_interaction" in feat_recursive.columns:
    feat_recursive = feat_recursive.with_columns(pl.col("snap_wday_interaction").cast(pl.Int32))

feat_recursive = feat_recursive.filter(pl.col("id") == sample_id).select(feature_cols)

print("\n=== COLUMN VALUES COMPARISON FOR FIRST ITEM ===")
for col in feature_cols:
    val_dir = feat_direct[col][0]
    val_rec = feat_recursive[col][0]
    if val_dir != val_rec:
        print(f"X MISMATCH -> {col}: Direct={val_dir} | Recursive={val_rec}")
    else:
        print(f"V MATCH    -> {col}: {val_dir}")