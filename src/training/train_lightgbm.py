import gc
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import polars as pl
from pathlib import Path

from src.training.dataset import prepare_fold
from src.training.splits import make_backtest_folds
from src.training.metrics import compute_fold_weights_and_scales, calculate_sampled_wrmsse


FEATURES_DIR = Path("data/processed/features")
MLFLOW_EXPERIMENT = "m5-forecasting-lgbm"

LGBM_PARAMS = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.15,
    "metric": "tweedie",
    "num_leaves": 255,
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
    "force_col_wise": True,
    "max_bin": 127, 
    "seed": 42,
}

def train_fold(
    lf: pl.LazyFrame,
    df_weight_ref: pl.DataFrame,
    fold,
    params: dict,
    sample_frac: float = 0.3,
    seed: int = 42,
) -> tuple[lgb.Booster, dict, np.ndarray]:
    X_train, y_train, train_ids, X_val, y_val, val_ids, cat_col_indices = prepare_fold(lf, fold, sample_frac=sample_frac, seed=seed)
    
    m5_reference = compute_fold_weights_and_scales(df_weight_ref, fold.train_end)


    dtrain = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=cat_col_indices,
        free_raw_data=True,
    )

    dval = lgb.Dataset(
        X_val,
        label=y_val,
        reference=dtrain,
        categorical_feature=cat_col_indices,
        free_raw_data=True,
    )

    del X_train
    gc.collect()

    callbacks = [
        lgb.early_stopping(params["early_stopping_rounds"], verbose=False),
        lgb.log_evaluation(period=100),
    ]

    fit_params = {
        k: v for k, v in params.items()
        if k not in ("early_stopping_rounds", "n_estimators")
    }

    model = lgb.train(
        fit_params,
        dtrain,
        num_boost_round=params["n_estimators"],
        valid_sets=[dval],
        callbacks=callbacks,
    )

    del dtrain, dval
    gc.collect()

    best_iter = model.best_iteration
    y_pred = np.clip(model.predict(X_val, num_iteration=best_iter), 0, None)
    
    POST_PROCESSING_MULTIPLIER = 0.975
    y_pred_scaled = y_pred * POST_PROCESSING_MULTIPLIER

    metrics = {
        "rmse": float(np.sqrt(np.mean((y_val - y_pred_scaled) ** 2))),
        "mae": float(np.mean(np.abs(y_val - y_pred_scaled))),
        "wrmsse": calculate_sampled_wrmsse(y_val, y_pred_scaled, val_ids, m5_reference),
        "best_iteration": int(best_iter),
    }

    del X_val, y_train, y_val, train_ids, val_ids
    gc.collect()

    Path("models").mkdir(exist_ok=True)
    model.save_model(
        f"models/lgbm_fold_{fold.fold_idx}.txt",
        num_iteration=best_iter,
    )

    # Possibly temporary removal before optuna pass
    # mlflow.lightgbm.log_model(
    #     model,
    #     name=f"model_fold_{fold.fold_idx}",
    # )

    return model, metrics, y_pred_scaled


def run_cv(
    params: dict = LGBM_PARAMS,
    experiment_name: str = MLFLOW_EXPERIMENT,
    sample_frac: float = 0.3,
) -> None:
    mlflow.set_experiment(experiment_name)

    lf = pl.scan_parquet(FEATURES_DIR / "*.parquet")
    print("Collecting weight reference data...")
    df_weight_ref = lf.select(["id", "date", "sales", "sell_price"]).collect()
    print(f"Reference data shape: {df_weight_ref.shape}")

    dates = (
        lf.select("date")
        .collect()["date"]
        .unique()
        .sort()
        .to_list()
    )

    folds = make_backtest_folds(dates)

    with mlflow.start_run(run_name="lgbm_baseline"):
        mlflow.log_params({k: v for k, v in params.items() if k != "verbose"})
        mlflow.log_param("sample_frac", sample_frac)

        all_fold_metrics = []

        for fold in folds:
            print(f"\n{'=' * 60}")
            print(f"Fold {fold.fold_idx}: {fold}")
            print(f"{'=' * 60}")

            model, metrics, _ = train_fold(
                lf, df_weight_ref, fold, params,
                sample_frac=sample_frac,
            )

            print(
                f"RMSE={metrics['rmse']:.4f} | "
                f"MAE={metrics['mae']:.4f} | "
                f"WRMSSE={metrics['wrmsse']:.4f} | "
                f"best_iter={metrics['best_iteration']}"
            )

            mlflow.log_metrics({
                f"fold{fold.fold_idx}_rmse": metrics["rmse"],
                f"fold{fold.fold_idx}_mae": metrics["mae"],
                f"fold{fold.fold_idx}_wrmsse": metrics["wrmsse"],
            }, step=fold.fold_idx)

            importance = model.feature_importance(importance_type="gain")
            feature_names = model.feature_name()
            top_features = sorted(
                zip(feature_names, importance),
                key=lambda x: x[1],
                reverse=True,
            )[:20]

            mlflow.log_dict(
                {feat: float(gain) for feat, gain in top_features},
                f"feature_importance/fold_{fold.fold_idx}.json",
            )

            for feat, gain in top_features[:5]:
                mlflow.log_metric(f"importance_top_{feat}", float(gain))

            all_fold_metrics.append(metrics)

            del model
            gc.collect()

        cv_rmse = float(np.mean([m["rmse"] for m in all_fold_metrics]))
        cv_rmse_std = float(np.std([m["rmse"] for m in all_fold_metrics]))
        cv_mae = float(np.mean([m["mae"] for m in all_fold_metrics]))
        cv_wrmsse = float(np.mean([m["wrmsse"] for m in all_fold_metrics]))

        mlflow.log_metrics({
            "cv_mean_rmse": cv_rmse,
            "cv_std_rmse": cv_rmse_std,
            "cv_mean_mae": cv_mae,
            "cv_mean_wrmsse": cv_wrmsse,
        })

        print(f"\n{'=' * 60}")
        print("CV RESULTS")
        print(f"{'=' * 60}")
        print(f"RMSE  : {cv_rmse:.4f} +- {cv_rmse_std:.4f}")
        print(f"MAE   : {cv_mae:.4f}")
        print(f"WRMSSE : {cv_wrmsse:.4f}")
        print(f"{'=' * 60}")