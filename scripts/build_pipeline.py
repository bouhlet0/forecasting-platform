import time
from src.features.pipeline import build_all_features

t0 = time.time()
build_all_features()
elapsed = time.time() - t0
print(f"Total time: {elapsed:.1f}s")