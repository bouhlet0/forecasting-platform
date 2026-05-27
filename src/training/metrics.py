from datetime import date, timedelta
import numpy as np
import polars as pl


def compute_fold_weights_and_scales(
    df_full: pl.DataFrame,
    train_end_date: date,
) -> dict:
    """
    Computes true M5 weight and scale factors.
    Filters out leading zeros for accurate scale denominators.
    """
    df_train = df_full.filter(pl.col("date") <= train_end_date)
    df_sorted = df_train.sort(["id", "date"])

    scale_df = (
        df_sorted.with_columns(
            cum_sales=pl.col("sales").cum_sum().over("id")
        )
        .filter(pl.col("cum_sales") > 0)  # Drops leading zero days before product introduction
        .group_by("id")
        .agg(
            scale=pl.col("sales").diff().pow(2).mean()
        )
    )

    weight_window_start = train_end_date - timedelta(days=27)

    weight_df = (
        df_train.filter(pl.col("date") >= weight_window_start)
        .with_columns(
            (pl.col("sales") * pl.col("sell_price").fill_null(0.0)).alias("revenue")
        )
        .group_by("id")
        .agg(pl.col("revenue").sum().alias("item_revenue"))
    )

    m5_info = scale_df.join(weight_df, on="id", how="left")
    
    m5_info = m5_info.with_columns([
        pl.col("scale").fill_null(1.0),
        pl.col("item_revenue").fill_null(0.0)
    ])
    
    m5_info = m5_info.with_columns(
        pl.when(pl.col("scale") < 1e-4).then(1.0).otherwise(pl.col("scale")).alias("scale")
    )

    total_rev = m5_info["item_revenue"].sum()
    if total_rev == 0:
        total_rev = 1.0

    m5_info = m5_info.with_columns(
        (pl.col("item_revenue") / total_rev).alias("weight")
    )

    return {
        row["id"]: {"scale": row["scale"], "weight": row["weight"]}
        for row in m5_info.iter_rows(named=True)
    }


def calculate_sampled_wrmsse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    val_ids: np.ndarray,
    m5_reference: dict,
) -> float:
    """
    Computes true WRMSSE across whatever subset of item IDs is passed in.
    """
    val_groups = {}
    for idx, sid in enumerate(val_ids):
        val_groups.setdefault(sid, []).append(idx)

    weighted_rmsse_sum = 0.0
    active_weights = 0.0

    for sid, val_idx in val_groups.items():
        ref = m5_reference.get(sid)
        if ref is None:
            continue

        v_true = y_true[val_idx]
        v_pred = y_pred[val_idx]

        # Standard M5 item-level RMSSE
        item_rmsse = np.sqrt(np.mean((v_true - v_pred) ** 2)) / np.sqrt(ref["scale"])
        
        weighted_rmsse_sum += item_rmsse * ref["weight"]
        active_weights += ref["weight"]

    if active_weights > 0:
        return float(weighted_rmsse_sum / active_weights)
    return 0.0