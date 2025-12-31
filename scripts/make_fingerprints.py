
#!pip install rdkit

import os
import numpy as np
import pandas as pd

from src.toxicity.data.fingerprint_data_featurization import featurize_dataframe

DATA_DIR = "data/processed"
FEATURE_DIR = "data/features"

os.makedirs(FEATURE_DIR, exist_ok=True)

splits = {
    "train": "train_clean.csv",
    "val": "val_clean.csv",
    "test": "test_clean.csv",
}

for split, filename in splits.items():
    print("=" * 80)
    print(f"Processing {split.upper()} split")
    print("=" * 80)

    df = pd.read_csv(f"{DATA_DIR}/{filename}")

    X, df_valid = featurize_dataframe(df, radius=2, n_bits=2048)

    np.save(f"{FEATURE_DIR}/X_{split}_morgan.npy", X)
    df_valid.to_csv(f"{DATA_DIR}/{split}_clean_featurized.csv", index=False)

    print("Saved:")
    print(f"  {FEATURE_DIR}/X_{split}_morgan.npy")
    print(f"  {DATA_DIR}/{split}_clean_featurized.csv")
