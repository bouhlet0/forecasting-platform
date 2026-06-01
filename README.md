# M5 Forecasting: Hierarchical Time Series with LightGBM

An end-to-end forecasting platform for the M5 competition dataset, implementing a symmetrical global LightGBM model with hierarchical features, recursive multi-horizon prediction, and robust post-hoc calibration. The system achieves a highly stable **0.7777 WRMSSE** utilizing an unweighted early-stopping evaluation setup.

## Table of Contents

- [M5 Forecasting: Hierarchical Time Series with LightGBM](#m5-forecasting-hierarchical-time-series-with-lightgbm)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Project Structure](#project-structure)
  - [Setup \& Installation](#setup--installation)
  - [Data Preparation](#data-preparation)
    - [Execution Flow](#execution-flow)
  - [Feature Engineering](#feature-engineering)
    - [Compute Features](#compute-features)
  - [Model Training \& Weights Geometry](#model-training--weights-geometry)
    - [1. Robust Median Normalization](#1-robust-median-normalization)
    - [2. Symmetrical Boosting Parameters](#2-symmetrical-boosting-parameters)
    - [3. Cross-Validation Configuration](#3-cross-validation-configuration)
  - [Prediction (Recursive)](#prediction-recursive)
  - [Evaluation Results](#evaluation-results)
    - [Cross-Validation Summary (Out-of-Sample Performance)](#cross-validation-summary-out-of-sample-performance)
    - [Fold 3 Granular Tracking Metrics](#fold-3-granular-tracking-metrics)
    - [Top 10 Feature Attributes by Split Gain (Fold 3)](#top-10-feature-attributes-by-split-gain-fold-3)
  - [Reproducing Results](#reproducing-results)
    - [Testing and Sanity Validation Utilities](#testing-and-sanity-validation-utilities)
  - [Experiment Tracking with MLflow](#experiment-tracking-with-mlflow)
  - [Future Improvements](#future-improvements)
    - [1. Architectural Scaling \& Compute Optimization](#1-architectural-scaling--compute-optimization)
    - [2. Automated Hyperparameter Tuning (Optuna Optimization Loop)](#2-automated-hyperparameter-tuning-optuna-optimization-loop)
    - [3. Hierarchical Model Ensembling (Top Competition Strategies)](#3-hierarchical-model-ensembling-top-competition-strategies)
  - [Dependencies](#dependencies)
  - [License](#license)

---

## Overview

This repository provides a complete platform for the **M5 forecasting** task (Kaggle competition). The goal is to forecast daily unit sales for 3,049 items across 10 stores in three US states, respecting a strict hierarchical structure (item -> department -> category -> store -> state).

**Key architectural pillars:**

- **Symmetrical Tree Topology:** Designed to prevent overfitting on highly volatile, intermittent outliers by matching leaf thresholds directly with leaf sizing.
- **Median-Normalized Training Weights:** Robust sample weighting applied exclusively to training rows to mirror the metric priorities of the WRMSSE evaluation layout without destabilizing gradient steps.
- **Unbiased Metric Decoupling:** Employs a Tweedie loss objective for gradient/Hessian calculation, but points early-stopping to raw validation RMSE, directly aligning the tree cutoff with squared-error competition scoring.
- **Recursive 28-Day Horizons:** Recomputes dynamic lags and sparse demand tracking iteratively to avoid input feature degradation over the prediction window.
- **Powell Multiplier Calibration:** Department-level post-processing calibration executed on clean out-of-sample estimates.

---

## Project Structure

```

.
├── .gitignore
├── LICENSE
├── README.md
├── data
│   ├── parquet
│   │   ├── calendar.parquet
│   │   ├── sales_train_evaluation.parquet
│   │   ├── sales_train_validation.parquet
│   │   ├── sample_submission.parquet
│   │   └── sell_prices.parquet
│   ├── processed
│   │   ├── features
│   │   │   ├── CA_1.parquet
│   │   │   ├── CA_2.parquet
│   │   │   ├── CA_3.parquet
│   │   │   ├── CA_4.parquet
│   │   │   ├── TX_1.parquet
│   │   │   ├── TX_2.parquet
│   │   │   ├── TX_3.parquet
│   │   │   ├── WI_1.parquet
│   │   │   ├── WI_2.parquet
│   │   │   └── WI_3.parquet
│   │   ├── group_aggregates.parquet
│   │   └── long_by_store
│   │       ├── CA_1.parquet
│   │       ├── CA_2.parquet
│   │       ├── CA_3.parquet
│   │       ├── CA_4.parquet
│   │       ├── TX_1.parquet
│   │       ├── TX_2.parquet
│   │       ├── TX_3.parquet
│   │       ├── WI_1.parquet
│   │       ├── WI_2.parquet
│   │       └── WI_3.parquet
│   └── raw
│       ├── calendar.csv
│       ├── sales_train_evaluation.csv
│       ├── sales_train_validation.csv
│       ├── sample_submission.csv
│       └── sell_prices.csv
├── mlflow.db
├── models
│   ├── lgbm_fold_1.txt
│   ├── lgbm_fold_2.txt
│   ├── lgbm_fold_3.txt
│   ├── multipliers_fold_1.json
│   ├── multipliers_fold_2.json
│   └── multipliers_fold_3.json
├── notebooks
│   ├── eda.ipynb
│   ├── forecast_visualization.ipynb
│   └── wrmsse_debug.ipynb
├── pyproject.toml
├── scripts
│   ├── build_long.py
│   ├── build_pipeline.py
│   ├── convert_to_parquet.py
│   ├── data_sanity_check.py
│   ├── inspect_data.py
│   ├── random_tests.py
│   ├── run_training.py
│   ├── test_built_features.py
│   ├── test_calendar.py
│   ├── test_dataset.py
│   ├── test_demand.py
│   ├── test_hierachical.py
│   ├── test_lag.py
│   ├── test_pipeline.py
│   ├── test_predict.py
│   ├── test_price.py
│   ├── test_rolling.py
│   ├── test_splits.py
│   ├── test_training.py
│   └── validate_long.py
└── src
    ├── __init__.py
    ├── data
    │   ├── __init__.py
    │   └── loader.py
    ├── features
    │   ├── __init__.py
    │   ├── calendar.py
    │   ├── demand.py
    │   ├── hierarchical.py
    │   ├── lag.py
    │   ├── pipeline.py
    │   ├── price.py
    │   └── rolling.py
    └── training
        ├── __init__.py
        ├── dataset.py
        ├── metrics.py
        ├── predict.py
        ├── splits.py
        └── train_lightgbm.py

14 directories, 82 files

```

---

## Setup & Installation

1. **Clone the repository**

```bash
   git clone [https://github.com/bouhlet0/forecasting-platform.git](https://github.com/bouhlet0/forecasting-platform.git)
   cd forecasting-platform

```

1. **Create a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate       # Windows

```

1. **Install dependencies**

```bash
pip install -e .

```

1. **Place M5 raw CSV files** Download the data from the M5 Forecasting competition and place the files into `data/raw/`:

- `calendar.csv`
- `sales_train_evaluation.csv`
- `sales_train_validation.csv`
- `sample_submission.csv`
- `sell_prices.csv`

1. **Convert CSV to Parquet** (for high-speed I/O processing)

```bash
python scripts/convert_to_parquet.py

```

---

## Data Preparation

The pipeline processes wide transactional sales data into store-partitioned long formats to handle heavy window functions in memory efficiently.

### Execution Flow

- **`loader.py`** processes the underlying wide array structures, melting daily sales columns into explicit timestamps, and writing clean parquet tables per store into `data/processed/long_by_store/`.
- Every observation maps: `id`, `date`, `sales`, `sell_price`, and structural category/store identifiers alongside historical event indicators.

Execute via:

```bash
python scripts/build_long.py

```

---

## Feature Engineering

All independent transformations are engineered lazily in Polars via `src/features/pipeline.py`. The framework processes **60 dense features** optimized to represent retail dynamics.

| Feature Suite | Columns / Logic | Context Captured |
| --- | --- | --- |
| **Lag Base** | `lag_1`, `lag_2`, `lag_3`, `lag_7`, `lag_14`, `lag_28` | Immediate historical demand anchors |
| **Rolling Windows** | Mean, std, and sum over 7/28/90d horizons; 28-day shifted lag-means | Momentum, macro trends, and lifecycle baselines |
| **Price Signals** | `price_change_w`, `price_rel_mean_90`, `price_vs_market_avg`, `price_std_30` | Markdown velocity, promotions, and category elasticity |
| **Demand Sparsity** | `days_since_last_sale`, `days_since_first_sale`, `zero_frac_30d` | Intermittent purchase rate adjustments |
| **Hierarchical Context** | Dept/category trailing sales and dynamic item-to-group volume ratios | Inter-series cross-sectional scaling signals |
| **Calendar & SNAP** | Holiday lead/lag windows, payload extensions, weekend interactions | Structural macro traffic spikes |

### Compute Features

```bash
python scripts/build_pipeline.py

```

---

## Model Training & Weights Geometry

The training module (`src/training/train_lightgbm.py`) instantiates a global multi-seed boosting layout mapped directly to the actual M5 error properties.

### 1. Robust Median Normalization

To prevent extreme gradient domination from massive-volume SKUs without losing the priority structure of the WRMSSE metric, row-weights undergo alpha-tempered median normalization:

```python
ratio = weight / max(scale, 1e-5)
multiplier = ratio ** 0.5  # Alpha-tempered root compression
train_weights = np.clip(multiplier / np.median(multiplier), 0.2, 5.0)

```

This forces the median sample weight exactly to `1.0`, centering the gradient landscape while protecting tree structures from boundary saturation.

### 2. Symmetrical Boosting Parameters

The architecture implements balanced, regularization-heavy hyperparameters adapted from top-tier competition solutions to maximize out-of-sample generalization.

```python
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
    "n_estimators": 2000,
    "early_stopping_rounds": 150,
    "verbose": -1,
    "n_jobs": -1,
    "force_col_wise": True,
    "boost_from_average": False,
    "max_bin": 100, 
    "seed": 42,
}

```

### 3. Cross-Validation Configuration

A 3-fold rolling-origin backtest handles time-based testing using a fixed 28-day horizon preceded by a strict 28-day gap period to prevent feature leakage.

- **Fold 1:** Train 2011-01-29 to 2016-01-31 | Gap 28d | Val 2016-02-29 to 2016-03-27
- **Fold 2:** Train 2011-01-29 to 2016-02-28 | Gap 28d | Val 2016-03-28 to 2016-04-24
- **Fold 3:** Train 2011-01-29 to 2016-03-27 | Gap 28d | Val 2016-04-25 to 2016-05-22

Launch the cross-validation and optimization stack:

```bash
python scripts/run_training.py

```

---

## Prediction (Recursive)

Because multiple features (such as rolling windows and lags) rely directly on the immediately preceding days, multi-horizon forecasts use **recursive inference** orchestrated via `src/training/predict.py`.

During prediction, day t is estimated using the current state array. The predicted point is instantly appended to the rolling queue, and features for day t+1 are updated lazily. Final predictions are adjusted via Powell-calibrated post-processing parameters mapped at inference time:

```python
from src.training.predict import predict_recursive
import lightgbm as lgb

model = lgb.Booster(model_file="models/lgbm_fold_3.txt")
preds = predict_recursive(model, lazy_frame, forecast_start=date(2016, 4, 25))

```

---

## Evaluation Results

### Cross-Validation Summary (Out-of-Sample Performance)

The following metrics reflect the stable execution output following the data leakage removal and unweighted early-stopping synchronization:

```
============================================================
CV RESULTS
============================================================
RMSE   : 1.9313 +- 0.0311
MAE    : 0.9640
WRMSSE : 0.7777
============================================================

```

### Fold 3 Granular Tracking Metrics

- **Validation Frame Shape:** (853,720, 63)
- **Prediction Target Output Range:** [0.004, 169.471]

| Group Tier | Series / Level Reference | RMSE | MAE | MAPE | Performance Insight |
| --- | --- | --- | --- | --- | --- |
| **High Volume** | `FOODS_3_090_CA_1_evaluation` | 14.765 | 11.462 | 20.6% | Strong scale-stability and clean tracking profile. |
| **Intermittent** | `HOUSEHOLD_2_101_CA_1_evaluation` | 0.585 | 0.440 | 76.3% | High percentage penalty driven by low-integer sparse scale. |
| **Mid-Range** | `FOODS_3_557_CA_3_evaluation` | 3.088 | 2.470 | 70.1% | High variance bounded between rolling trend steps. |
| **Aggregate** | `Foods` (Department Summation) | 318.22 | - | 6.9% | Strong error cancellation under hierarchical aggregation. |

### Top 10 Feature Attributes by Split Gain (Fold 3)

```text
Rank  Feature                Importance (Gain)  Core Functional Utility

1     roll_mean_7            122,092,975        Short-term tracking baseline window
2     roll_mean_28           119,924,063        Medium-term trajectory context
3     days_since_last_sale    23,476,723        Critical zero-inflation velocity flag
4     roll_std_28             22,486,368        28-day demand volatility variance proxy
5     roll_sum_90             13,431,719        Long-horizon historical volume tracking
6     zero_frac_30d           10,644,303        Probability estimation boundary signal
7     roll_max_28             10,019,799        Peak demand capacity envelope
8     roll_mean_90             8,186,242        Macro seasonal trajectory trend
9     roll_std_7               7,038,500        Weekly cyclic noise filtering
10    lag_1                    4,639,869        Direct raw demand baseline anchor

```

---

## Reproducing Results

Execute the following sequential processing pipeline to verify metric outputs:

1. **Verify Source State:** Ensure all raw telemetry tables exist in `data/raw/`.
2. **Compress Storage Arrays:** Convert raw formats to parquet: `python scripts/convert_to_parquet.py`
3. **Restructure Layout:** Pivot wide tables into a long schema: `python scripts/build_long.py`
4. **Generate Features:** Run the feature creation script to construct the training variables: `python scripts/build_pipeline.py`
5. **Execute Fitting Pipeline:** Run the training loop to generate the models and log results: `python scripts/run_training.py`

### Testing and Sanity Validation Utilities

The framework provides an extensive suite of assertion and testing scripts in `scripts/` to isolate and validate data mutations or pipeline logic changes:

```bash
# Data Structure Validation
python scripts/data_sanity_check.py
python scripts/inspect_data.py
python scripts/validate_long.py

# Component Layer Tests
python scripts/test_calendar.py
python scripts/test_demand.py
python scripts/test_lag.py
python scripts/test_price.py
python scripts/test_rolling.py
python scripts/test_hierachical.py

# Structural Framework Verification
python scripts/test_dataset.py
python scripts/test_splits.py
python scripts/test_pipeline.py
python scripts/test_built_features.py
python scripts/test_predict.py
python scripts/test_training.py
python scripts/random_tests.py

```

---

## Experiment Tracking with MLflow

The training pipeline automatically logs hyperparameters, validation iterations, and evaluation scores to a local SQLite database (`mlflow.db`).

To launch the interactive MLflow dashboard and review your training history, metric curves, and performance metrics side-by-side, run:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## Future Improvements

While the baseline platform is mathematically stable, the following targeted enhancements are designed to directly compress the validation WRMSSE score:

### 1. Architectural Scaling & Compute Optimization

- **Full-Dataset Scale:** Lift the training data slice from the current 30% sample (`sample_frac=0.3`) to 100%. Processing the complete target history exposes the Tweedie loss function to rare intermittent sequences, reducing validation edge anomalies.
  
- **RAM Constraints Note:** Full scaling requires optimizing chunk-wise operations or scaling out memory profiles, as the 100% long-form array exceeds a 24GB local footprint.

### 2. Automated Hyperparameter Tuning (Optuna Optimization Loop)

Implement an automated Optuna optimization framework over the structural LightGBM boundaries to bypass manual tuning. For a retail dataset containing zero-inflation and high variance, the tuning space must target three specific mathematical pillars:

- **Tweedie Variance Power ($[1.05, 1.25]$):** This scales the underlying compound Poisson-Gamma distribution loss. Tuning closer to 1.05 forces the model to tightly penalize errors on zero-inflation boundaries (improving intermittent series), while tuning toward 1.25 better handles high-volume variance tracking.
- **Topology Symmetry Balance (`num_leaves` and `min_child_samples` $[127, 511]$):** To prevent structural overfitting on weak signals, look to link these spaces together. Forcing a high min-child-data threshold alongside large leaf arrays ensures deep splits retain historical mass.
- **Regularization Compression (`feature_fraction` and `bagging_fraction` $[0.4, 0.7]$):** Restricting trees to a maximum of 40% to 70% of rows and columns prevents dominant rolling features from overshadowing downstream calendar and price signals, maximizing ensemble diversity.

### 3. Hierarchical Model Ensembling (Top Competition Strategies)

Top-tier submissions for the M5 competition demonstrated that a single global model can be significantly improved by tailoring estimators to different cross-sections of the hierarchy:

- **Level-Specific Specialization:** Train independent LightGBM boosters for specific subsets of the hierarchy (e.g., separate models for individual stores or distinct product categories like Foods vs. Hobbies).
- **Ensemble Blending:** Dynamically combine the outputs of these specialized models at inference time. This allows specialized trees to isolate hyper-local demand behaviors without losing the macro-trend stability captured by the global model.

---

## Dependencies

- `polars>=1.0.0` (High-performance out-of-core data frames)
- `lightgbm>=4.5.0` (Histogram-optimized gradient boosting framework)
- `mlflow>=2.10.0` (Parametric run and metadata visualization tracking)
- `numpy`
- `scipy` (Provides Powell optimizer matrix functions)

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for open-source details.
