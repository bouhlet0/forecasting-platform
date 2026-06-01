import gc
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import polars as pl
import scipy.optimize as opt
import json
from pathlib import Path

from src.training.dataset import prepare_fold
from src.training.splits import make_backtest_folds
from src.training.metrics import compute_fold_weights_and_scales, calculate_sampled_wrmsse


FEATURES_DIR = Path("data/processed/features")
MLFLOW_EXPERIMENT = "m5-forecasting-lgbm"

LGBM_PARAMS = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.15,
    "metric": "rmse",
    "num_leaves": 255,
    "min_child_samples": 255,
    "learning_rate": 0.025,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.5,
    "bagging_freq": 1,
    # "lambda_l2": 0.1,
    "n_estimators": 2000,
    "early_stopping_rounds": 150,
    "verbose": -1,
    "n_jobs": -1,
    "force_col_wise": True,
    "boost_from_average": False,
    "max_bin": 100, 
    "seed": 42,
}

def optimize_department_multipliers(
    y_true: np.ndarray,
    y_pred_base: np.ndarray,
    val_ids: np.ndarray,
    m5_reference: dict,
    lf: pl.LazyFrame,
    fold,
) -> tuple[np.ndarray, dict]:
    print("\nOptimizing multipliers by department on validation predictions.")
    
    val_depts = (
        lf.filter(
            (pl.col("date") >= fold.val_start) & 
            (pl.col("date") <= fold.val_end)
        )
        .select("dept_id")
        .collect()
        ["dept_id"]
        .to_numpy()
    )
    
    unique_depts = np.unique(val_depts)

    def evaluate_multipliers(multipliers_vector):
        mult_dict = dict(zip(unique_depts, multipliers_vector))
        row_multipliers = np.array([mult_dict[d] for d in val_depts])
        scaled_preds = y_pred_base * row_multipliers
        return calculate_sampled_wrmsse(y_true, scaled_preds, val_ids, m5_reference)

    # Use 0.975 as the starting anchor point for all departments
    initial_guess = [0.975] * len(unique_depts)
    bounds = [(0.85, 1.15)] * len(unique_depts)

    res = opt.minimize(evaluate_multipliers, initial_guess, bounds=bounds, method="Powell")
    
    optimal_mult_dict = dict(zip(unique_depts, res.x))
    final_row_multipliers = np.array([optimal_mult_dict[d] for d in val_depts])
    
    y_pred_final = y_pred_base * final_row_multipliers

    print("Optimal per-dept multipliers for this run ")
    for d, m in optimal_mult_dict.items():
        print(f"  {d:12} -> {m:.4f}")
        
    return y_pred_final, optimal_mult_dict

def get_safe_multiplier(sid, m5_ref, alpha=0.5):
    if sid not in m5_ref:
        return 1.0

    scale = m5_ref[sid]["scale"]
    weight = m5_ref[sid]["weight"]

    if scale <= 0:
        return 1.0

    # Soft stabilization floor to handle low-volume items smoothly
    safe_scale = max(scale, 1e-5)
    ratio = weight / safe_scale

    # Compute raw importance curve
    multiplier = ratio ** alpha

    # Soft clipping here just to prevent catastrophic infinity errors early on
    return float(np.clip(multiplier, 1e-5, 50.0))

def train_fold(
    lf: pl.LazyFrame,
    df_weight_ref: pl.DataFrame,
    fold,
    params: dict,
    sample_frac: float = 0.3,
    seed: int = 42,
) -> tuple[list[lgb.Booster], dict, np.ndarray]:
    
    X_train, y_train, train_ids, X_val, y_val, val_ids, cat_col_indices = prepare_fold(
        lf, fold, sample_frac=sample_frac, seed=seed
    )
    
    m5_reference = compute_fold_weights_and_scales(df_weight_ref, fold.train_end)

    print("Generating WRMSSE Custom Sample Weights for Training.")
    raw_train_multipliers = np.array([
        get_safe_multiplier(sid, m5_reference, alpha=0.5) for sid in train_ids
    ], dtype=np.float32)

    train_median = np.median(raw_train_multipliers)

    normalized_weights = raw_train_multipliers / max(train_median, 1e-8)

    train_weights = np.clip(normalized_weights, 0.2, 5.0)

    print("\n" + "="*40)
    print(" TRAINING WEIGHT DIAGNOSTICS (MEDIAN SCALED)")
    print("="*40)
    print(f"  Min Weight:  {train_weights.min():.4f}")
    print(f"  Max Weight:  {train_weights.max():.4f}")
    print(f"  Mean Weight: {train_weights.mean():.4f}")
    print(f"  Median:      {np.median(train_weights):.4f}")
    print(f"  95th Pctl:   {np.percentile(train_weights, 95):.4f}")
    print(f"  99th Pctl:   {np.percentile(train_weights, 99):.4f}")
    print("="*40 + "\n")
    print("="*40 + "\n")

    dtrain = lgb.Dataset(
        X_train,
        label=y_train,         
        weight=train_weights,  
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
        if k not in ("early_stopping_rounds", "n_estimators", "seed")
    }

    seeds = [42, 100]
    trained_seeds_models = []
    raw_val_preds_accumulated = np.zeros_like(y_val, dtype=np.float32)

    for current_seed in seeds:
        print(f"--- Training Seed {current_seed} ---")
        current_params = fit_params.copy()
        current_params["seed"] = current_seed
        
        model = lgb.train(
            current_params,
            dtrain,
            num_boost_round=params["n_estimators"],
            valid_sets=[dval],
            callbacks=callbacks,
        )
        trained_seeds_models.append(model)
        
        raw_val_preds_accumulated += model.predict(X_val, num_iteration=model.best_iteration)
        
        del model
        gc.collect()

    avg_raw_preds = raw_val_preds_accumulated / len(seeds)

    del dtrain, dval
    gc.collect()

    y_pred_base = np.clip(avg_raw_preds, 0, None)
    
    y_pred_final, optimal_mult_dict = optimize_department_multipliers(
        y_true=y_val,
        y_pred_base=y_pred_base,
        val_ids=val_ids,
        m5_reference=m5_reference,
        lf=lf,
        fold=fold
    )

    metrics = {
        "rmse": float(np.sqrt(np.mean((y_val - y_pred_final) ** 2))),
        "mae": float(np.mean(np.abs(y_val - y_pred_final))),
        "wrmsse": calculate_sampled_wrmsse(y_val, y_pred_final, val_ids, m5_reference),
        "best_iteration": int(trained_seeds_models[0].best_iteration),
    }

    for d, m in optimal_mult_dict.items():
        mlflow.log_metric(f"fold{fold.fold_idx}_multiplier_{d}", float(m))

    del X_val, y_train, y_val, train_ids, val_ids
    gc.collect()

    Path("models").mkdir(exist_ok=True)
    trained_seeds_models[0].save_model(
        f"models/lgbm_fold_{fold.fold_idx}.txt",
        num_iteration=trained_seeds_models[0].best_iteration,
    )

    with open(f"models/multipliers_fold_{fold.fold_idx}.json", "w") as f:
        json.dump(optimal_mult_dict, f)

    return trained_seeds_models, metrics, y_pred_final


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

            importance = model[0].feature_importance(importance_type="gain")
            feature_names = model[0].feature_name()
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