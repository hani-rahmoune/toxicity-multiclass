from __future__ import annotations
#pip install rdkit
#import os
#from pathlib import Path
#import matplotlib.pyplot as plt
#import seaborn as sns
#import numpy as np
#import pandas as pd
#from typing import Optional, Tuple, Dict
#from pathlib import Path
#import json


import sys
import json
from pathlib import Path

from src.toxicity.data.download import download_and_prepare_tox21
from src.toxicity.data.cleaning import clean_dataframe


def main() -> None:
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)

    print("Downloading Tox21 dataset from MoleculeNet...\n")
    train_df, val_df, test_df = download_and_prepare_tox21(
        output_dir="data/raw",
        download_fresh=False
    )

    train_clean, train_stats = clean_dataframe(train_df)
    val_clean, val_stats = clean_dataframe(val_df)
    test_clean, test_stats = clean_dataframe(test_df)

    train_clean.to_csv("data/processed/train_clean.csv", index=False)
    val_clean.to_csv("data/processed/val_clean.csv", index=False)
    test_clean.to_csv("data/processed/test_clean.csv", index=False)

    with open("results/cleaning_stats.json", "w") as f:
        json.dump(
            {"train": train_stats, "val": val_stats, "test": test_stats},
            f,
            indent=2
        )

    print("Saved cleaned splits to data/processed/")
    print("Saved cleaning stats to results/cleaning_stats.json")


if __name__ == "__main__":
    main()
