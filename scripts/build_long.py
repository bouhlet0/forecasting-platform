import time
import polars as pl
from pathlib import Path
from src.data.loader import build_long_chunked

OUTPUT_DIR = Path("data/processed/long_by_store")

t0 = time.time()
build_long_chunked(output_dir=OUTPUT_DIR)
elapsed = time.time() - t0

print(f"Build time: {elapsed:.1f}s")

# read back lazily using glob
df = pl.scan_parquet(OUTPUT_DIR / "*.parquet")
print(f"Columns: {df.columns}")
print(f"Shape: {df.select(pl.len()).collect().item()} rows")
print(df.head(3).collect())