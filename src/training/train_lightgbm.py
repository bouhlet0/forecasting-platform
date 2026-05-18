import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import polars as pl
from pathlib import Path

from src.training.dataset import prepare_fold, CAT_FEATURES
from src.training.splits import make_backtest_folds

FEATURES_DIR = Path("data/processed/features")
MLFLOW_EXPERIMENT = "m5-forecasting-lgbm"

# based on top-performing public M5 solutions
LGBM_PARAMS = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.1,
    "metric": "rmse",
    "num_leaves": 511,
    "min_child_samples": 20,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 0.1,
    "n_estimators": 1500,
    "early_stopping_rounds": 50,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}


def rmsse(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
    y_train = np.asarray(y_train)

    scale = np.mean(np.diff(y_train) ** 2)
    scale = scale if scale > 1e-8 else 1.0

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2) / scale))


def wrmsse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    weights: np.ndarray,
) -> float:
    n_series = weights.shape[0]
    series_len = len(y_true) // n_series
    total = 0.0
    for i in range(n_series):
        start = i * series_len
        end = start + series_len
        r = rmsse(y_true[start:end], y_pred[start:end], y_train)
        total += weights[i] * r
    return total


def train_fold(
    lf: pl.LazyFrame,
    fold,
    params: dict,
) -> tuple[lgb.Booster, dict]:
    X_train, y_train, X_val, y_val = prepare_fold(lf, fold)

    dtrain = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=CAT_FEATURES,
        free_raw_data=False,
    )
    dval = lgb.Dataset(
        X_val,
        label=y_val,
        reference=dtrain,
        categorical_feature=CAT_FEATURES,
        free_raw_data=False,
    )

    callbacks = [
        lgb.early_stopping(params["early_stopping_rounds"], verbose=False),
        lgb.log_evaluation(period=100),
    ]

    fit_params = {k: v for k, v in params.items()
                  if k not in ("early_stopping_rounds",)}

    model = lgb.train(
        fit_params,
        dtrain,
        valid_sets=[dval],
        callbacks=callbacks,
    )

    y_pred = model.predict(X_val)
    y_pred = np.clip(y_pred, 0, None)  # no negative forecasts

    y_val_arr = np.asarray(y_val)

    metrics = {
        "rmse": float(np.sqrt(np.mean((y_val_arr - y_pred) ** 2))),
        "mae": float(np.mean(np.abs(y_val_arr - y_pred))),
        "best_iteration": model.best_iteration,
    }

    return model, metrics, y_pred


def run_cv(
    store_ids: list[str] | None = None,
    params: dict = LGBM_PARAMS,
    experiment_name: str = MLFLOW_EXPERIMENT,
) -> None:
    if store_ids is None:
        store_ids = [
            "CA_1", "CA_2", "CA_3", "CA_4",
            "TX_1", "TX_2", "TX_3",
            "WI_1", "WI_2", "WI_3",
        ]

    mlflow.set_experiment(experiment_name)

    # get fold dates from first store
    lf_sample = pl.scan_parquet(FEATURES_DIR / f"{store_ids[0]}.parquet")
    dates = lf_sample.select("date").collect()["date"].unique().to_list()
    folds = make_backtest_folds(dates)

    with mlflow.start_run(run_name="lgbm_baseline"):
        mlflow.log_params({k: v for k, v in params.items()
                          if k != "verbose"})

        all_fold_metrics = []

        for fold in folds:
            print(f"\n{'='*50}")
            print(f"Fold {fold.fold_idx}: {fold}")
            print(f"{'='*50}")

            fold_metrics = {"fold": fold.fold_idx, "stores": {}}

            for store_id in store_ids:
                lf = pl.scan_parquet(FEATURES_DIR / f"{store_id}.parquet")
                model, metrics, _ = train_fold(lf, fold, params)

                fold_metrics["stores"][store_id] = metrics
                print(f"  {store_id}: RMSE={metrics['rmse']:.4f} "
                      f"MAE={metrics['mae']:.4f} "
                      f"best_iter={metrics['best_iteration']}")

                mlflow.log_metrics({
                    f"fold{fold.fold_idx}_{store_id}_rmse": metrics["rmse"],
                    f"fold{fold.fold_idx}_{store_id}_mae": metrics["mae"],
                }, step=fold.fold_idx)

            # aggregate across stores for this fold
            fold_rmse = np.mean([
                m["rmse"] for m in fold_metrics["stores"].values()
            ])
            fold_mae = np.mean([
                m["mae"] for m in fold_metrics["stores"].values()
            ])
            print(f"\n  Fold {fold.fold_idx} mean | "
                  f"RMSE: {fold_rmse:.4f}  MAE: {fold_mae:.4f}")

            mlflow.log_metrics({
                f"fold{fold.fold_idx}_mean_rmse": fold_rmse,
                f"fold{fold.fold_idx}_mean_mae": fold_mae,
            }, step=fold.fold_idx)

            all_fold_metrics.append(fold_metrics)

        # overall CV metrics
        all_rmse = [
            m["rmse"]
            for fold_m in all_fold_metrics
            for m in fold_m["stores"].values()
        ]
        cv_rmse = float(np.mean(all_rmse))
        cv_rmse_std = float(np.std(all_rmse))

        mlflow.log_metrics({
            "cv_mean_rmse": cv_rmse,
            "cv_std_rmse": cv_rmse_std,
        })

        print(f"\n{'='*50}")
        print(f"CV Results: RMSE={cv_rmse:.4f} +- {cv_rmse_std:.4f}")
        print(f"{'='*50}")