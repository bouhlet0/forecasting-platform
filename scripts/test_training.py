import polars as pl
import mlflow
from src.training.train_lightgbm import train_fold, LGBM_PARAMS
from src.training.splits import make_backtest_folds

mlflow.set_experiment("m5-forecasting-lgbm-test")

lf = pl.scan_parquet("data/processed/features/CA_1.parquet")
dates = lf.select("date").collect()["date"].unique().to_list()
folds = make_backtest_folds(dates)

fold = folds[-1]  # use last fold (most training data)
print(f"Testing on {fold}")

with mlflow.start_run(run_name="single_fold_test"):
    model, metrics, y_pred = train_fold(lf, fold, LGBM_PARAMS)
    mlflow.log_metrics(metrics)
    print(f"\nMetrics: {metrics}")
    print(f"Predictions sample: {y_pred[:10]}")