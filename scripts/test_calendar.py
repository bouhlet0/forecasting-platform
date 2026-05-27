import polars as pl
from src.features.calendar import build_calendar_features, add_snap_feature

# TEST CALENDAR FEATURES ALONE
cal = build_calendar_features().collect()
print(f"Calendar shape: {cal.shape}")

print("\nChristmas days:")
print(cal.filter(pl.col("is_christmas") == 1).select(["date", "is_christmas"]))

print("\nSporting event flag value counts:")
print(cal["is_sporting_event"].value_counts())

print("\nSuper Bowl 2011-02-06: (should be = 1)")
print(
    cal.filter(pl.col("date") == pl.date(2011, 2, 6))
    .select(["date", "event_name_1", "event_type_1", "event_name_2", "event_type_2", "is_sporting_event"])
)

print("\nDays around NBA Finals Start (2011-05-31):")
print(
    cal.filter(pl.col("date").is_between(pl.date(2011, 5, 30), pl.date(2011, 6, 2)))
    .select(["date", "event_name_1", "event_type_1", "is_sporting_event"])
)

print("\nNational holiday proximity sample:")
print(cal.filter(pl.col("is_national_holiday") == 1)
    .select(["date", "event_name_1", "is_national_holiday",
             "national_lag_1", "national_lag_2",
             "national_lead_1", "national_lead_2"])
    .head(10))

# NEW COMPREHENSIVE CALENDAR FEATURES TESTS 
print("\n--- NEW: Macro Event Window Verification (Around Independence Day 2011) ---")
print(cal.filter(pl.col("date").is_between(pl.date(2011, 7, 1), pl.date(2011, 7, 6)))
    .select(["date", "event_name_1", "is_event_day", 
             "event_lead_3", "event_lead_2", "event_lead_1", 
             "event_lag_1", "event_lag_2"]))


# TEST SNAP ON JOINED STORE DATA
lf = pl.scan_parquet("data/processed/long_by_store/CA_1.parquet")
result = add_snap_feature(lf).collect()
print(f"\nSales shape after snap: {result.shape}")

print("\nSNAP sample (CA should match snap_CA values):")
print(result.select(["date", "state_id", "snap"]).filter(
    pl.col("snap") == 1
).head(5))

print("\n--- NEW: SNAP Interactions Verification ---")
print(result.select(["date", "state_id", "snap", "snap_x_weekend", "wday", "snap_wday_interaction"])
    .filter(pl.col("snap") == 1)
    .head(10))

print("\nSnap columns dropped - remaining columns with 'snap' in the name:")
print([c for c in result.columns if "snap" in c])

# CONSISTENCY & INTEGRITY ASSERTI0NS
print("\nDays where event_type_2 is Sporting:")
print(
    cal.filter(pl.col("event_type_2") == "Sporting")
    .select(["date", "event_name_1", "event_type_1", "event_name_2", "event_type_2", "is_sporting_event"])
)

# Validation: is_sporting_event == 1  =>  event_type_1 == "Sporting" OR event_type_2 == "Sporting"
cal_with_check = cal.with_columns(
    pl.when(
        (pl.col("event_type_1") == "Sporting") | (pl.col("event_type_2") == "Sporting")
    )
    .then(1)
    .otherwise(0)
    .alias("expected")
)
assert cal_with_check.filter(pl.col("is_sporting_event") != pl.col("expected")).is_empty()
print("\nSporting flag consistency check passed.")

# Verification that missing events are mapped to string "0" instead of "none" or null
null_event_count = cal.filter(pl.col("event_name_1").is_null() | (pl.col("event_name_1") == "none")).shape[0]
assert null_event_count == 0, f"Found {null_event_count} unmapped event entries!"
print("String category baseline integrity check passed ('0' confirmed as empty state).")