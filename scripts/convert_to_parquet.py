from pathlib import Path
import polars as pl

RAW_DIR = Path("data/raw")
PARQUET_DIR = Path("data/parquet")

PARQUET_DIR.mkdir(parents=True, exist_ok=True)

for csv_file in RAW_DIR.glob("*.csv"):
    print(f"Loading {csv_file.name}...")

    df = pl.read_csv(csv_file)

    output_file = PARQUET_DIR / f"{csv_file.stem}.parquet"

    df.write_parquet(output_file)

    print(f"Saved -> {output_file.name}")