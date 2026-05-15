import polars as pl
from src.features.pipeline import build_features_for_store
from src.features.calendar import build_calendar_features

calendar_features = build_calendar_features()
df = build_features_for_store("CA_1", calendar_features)

print(f"Shape: {df.shape}")
print(f"Columns ({len(df.columns)}): {df.columns}")
print("\nNull counts:")
with pl.Config(tbl_rows=28):
    print(df.null_count().transpose(include_header=True)
        .filter(pl.col("column_0") > 0)
        .rename({"column": "feature", "column_0": "null_count"}))
print("\nSample:")
print(df.select(["id", "date", "sales"]).tail(5))

# assertions
expected_cols = {
    # identifiers
    "id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "d", "date",
    # target
    "sales",
    # calendar
    "wm_yr_wk", "wday", "month", "year",
    "event_name_1", "event_type_1", "event_name_2", "event_type_2",
    "is_national_holiday", "is_christmas", "is_weekend", "is_sporting_event",
    "national_lag_1", "national_lag_2", "national_lead_1", "national_lead_2",
    "snap",
    # price
    "sell_price", "has_price", "price_change_w", "price_rel_mean_90",
    "price_std_30", "is_on_promotion",
    # lags
    "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_28",
    # rolling
    "roll_mean_7", "roll_std_7", "roll_sum_7",
    "roll_mean_28", "roll_std_28", "roll_sum_28",
    "roll_mean_90", "roll_std_90", "roll_sum_90",
    "roll_max_28", "roll_median_28",
    "trend_ratio_7_28", "trend_ratio_28_90",
    # demand
    "zero_frac_30d", "days_since_last_sale", "days_since_first_sale",
    # hierarchical
    "dept_sales_lag1", "dept_sales_roll_28",
    "cat_sales_lag1", "cat_sales_roll_28",
    "item_share_dept", "item_share_cat",
}

actual_cols = set(df.columns)
missing = expected_cols - actual_cols
extra = actual_cols - expected_cols

# shape and columns
assert not missing, f"Missing columns: {missing}"
assert len(df.columns) == 60, f"Expected 60 columns, got {len(df.columns)}"

# no nulls in features that must be complete
assert df["sales"].null_count() == 0, "sales has nulls"
assert df["days_since_last_sale"].null_count() == 0, "days_since_last_sale has nulls"
assert df["days_since_first_sale"].null_count() == 0, "days_since_first_sale has nulls"
assert df["snap"].null_count() == 0, "snap has nulls"
assert df["is_national_holiday"].null_count() == 0, "is_national_holiday has nulls"
assert df["is_christmas"].null_count() == 0, "is_christmas has nulls"
assert df["is_weekend"].null_count() == 0, "is_weekend has nulls"
assert df["has_price"].null_count() == 0, "has_price has nulls"
assert df["item_share_dept"].null_count() == 0, "item_share_dept has nulls"
assert df["item_share_cat"].null_count() == 0, "item_share_cat has nulls"

# sentinel fills
assert df["days_since_last_sale"].max() == 9999, "days_since_last_sale max should be 9999"
assert df["days_since_first_sale"].min() == 0, "days_since_first_sale min should be 0"

# value ranges
assert df["sales"].min() >= 0, "negative sales found"
assert df["snap"].is_in([0, 1]).all(), "snap has values outside {0, 1}"
assert df["is_national_holiday"].is_in([0, 1]).all(), "is_national_holiday outside {0, 1}"
assert df["is_christmas"].is_in([0, 1]).all(), "is_christmas outside {0, 1}"
assert df["is_weekend"].is_in([0, 1]).all(), "is_weekend outside {0, 1}"
assert df["has_price"].is_in([0, 1]).all(), "has_price outside {0, 1}"
assert df["is_on_promotion"].drop_nulls().is_in([0, 1]).all(), "is_on_promotion outside {0, 1}"
assert df["zero_frac_30d"].drop_nulls().min() >= 0.0, "zero_frac_30d below 0"
assert df["zero_frac_30d"].drop_nulls().max() <= 1.0, "zero_frac_30d above 1"
assert df["item_share_dept"].min() >= 0.0, "item_share_dept below 0"
assert df["item_share_dept"].max() <= 1.0, "item_share_dept above 1"
assert df["item_share_cat"].min() >= 0.0, "item_share_cat below 0"
assert df["item_share_cat"].max() <= 1.0, "item_share_cat above 1"

# lag null ordering
assert df["lag_1"].null_count() < df["lag_7"].null_count(), "lag null counts out of order"
assert df["lag_7"].null_count() < df["lag_28"].null_count(), "lag null counts out of order"

# leakage proxy
feature_corrs = {
    col: abs(df.select(pl.corr("sales", col)).item())
    for col in ["lag_1", "lag_7", "lag_28", "roll_mean_7", "snap"]
    if df[col].null_count() < len(df) * 0.5
}
assert feature_corrs["lag_1"] > feature_corrs["snap"], \
    f"snap correlation ({feature_corrs['snap']:.3f}) exceeds lag_1 ({feature_corrs['lag_1']:.3f}): possible leakage"
assert feature_corrs["lag_1"] > feature_corrs["lag_28"], \
    f"lag_28 correlation ({feature_corrs['lag_28']:.3f}) exceeds lag_1 ({feature_corrs['lag_1']:.3f}): possible leakage"

if extra:
    print(f"Warning: unexpected columns present: {extra}")

print(f"\nFeature correlations with sales: {feature_corrs}")

# lag correctness
sample_ids = df["id"].unique().sample(5, seed=42)

for sid in sample_ids:
    sub = df.filter(pl.col("id") == sid).sort("date")

    lag_check = (
        sub.select([
            pl.col("sales").shift(1).alias("expected_lag_1"),
            pl.col("lag_1"),
        ])
        .drop_nulls()
    )

    assert (
        (lag_check["expected_lag_1"] == lag_check["lag_1"]).all()
    ), f"LAG_1 mismatch for {sid}"

# rolling mean check
expected = sub["sales"].shift(1).rolling_mean(7)
actual = sub["roll_mean_7"]
assert (expected - actual).abs().max() < 1e-6

# uniqueness / alignment check
dup_count = df.group_by(["id", "date"]).len().filter(pl.col("len") > 1).height
assert dup_count == 0, f"Duplicate (id, date) rows found: {dup_count}"

# time ordering check
for sid in sample_ids:
    sub = df.filter(pl.col("id") == sid)

    dates = sub["date"].to_list()

    assert dates == sorted(dates), f"Date ordering broken for {sid}"
    
# determinism check
df2 = build_features_for_store("CA_1", calendar_features)

assert df.shape == df2.shape, "Re-run shape mismatch"

assert (
    df.select(["sales", "lag_1", "roll_mean_7"])
    .equals(df2.select(["sales", "lag_1", "roll_mean_7"]))
), "Non-deterministic feature output detected"

print("\nAll assertions passed.")