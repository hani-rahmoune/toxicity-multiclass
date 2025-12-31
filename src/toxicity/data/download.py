from __future__ import annotations



"""
Manual Tox21 Dataset Download (MoleculeNet)

This script downloads and prepares the Tox21 dataset directly from the
official MoleculeNet data source

The pipeline performs the following steps:
- Download the dataset archive
- Decompress the CSV file if needed
- Load the dataset into a pandas DataFrame
- Validate dataset structure and label values
- Compute basic dataset statistics
- Split the dataset into train/validation/test sets
- Save the resulting files and metadata to disk

Dataset:
- Source URL: https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz
- Format: CSV containing SMILES strings and 12 toxicity assay labels
"""


import gzip
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import requests



# Dataset configuration

TOX21_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
TOX21_CSV_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv"

TOX21_ASSAYS = [
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
]

EXPECTED_COLUMNS = ["mol_id", "smiles"] + TOX21_ASSAYS



# Download utilities


def download_file(url: str, output_path: str, chunk_size: int = 8192) -> bool:
    
    # Download a file from a remote URL using streaming requests.

   
    
    try:
        output_path = str(output_path)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)

        if total_size > 0 and downloaded != total_size:
            return False

        return True

    except Exception:
        return False


def decompress_gzip(gz_path: str, output_path: str) -> bool:
    
    #Decompress a gzip-compressed file to a target location.
    
    try:
        with gzip.open(gz_path, "rb") as f_in:
            with open(output_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        return True
    except Exception:
        return False



# Data loading and validation


def load_tox21_csv(csv_path: str) -> pd.DataFrame:
    
    # Load the Tox21 CSV file into a pandas DataFrame.
    
    return pd.read_csv(
        csv_path,
        na_values=["", "NA", "NaN", "nan", "null"],
        keep_default_na=True,
    )


def validate_tox21_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate the structure and basic integrity of the Tox21 dataset.

    Checks include:
    - Presence of required columns
    - Existence of the SMILES column
    - Validity of assay label values (0, 1, or NaN)
    """
    report: Dict[str, Any] = {
        "valid": True,
        "issues": [],
        "warnings": [],
        "statistics": {},
    }

    missing_cols = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    if missing_cols:
        report["valid"] = False
        report["issues"].append(f"Missing columns: {missing_cols}")

    if "smiles" not in df.columns:
        report["valid"] = False
        report["issues"].append("Missing 'smiles' column")
    else:
        n_missing_smiles = int(df["smiles"].isna().sum())
        if n_missing_smiles > 0:
            report["warnings"].append(f"{n_missing_smiles} rows have missing SMILES")

    for assay in TOX21_ASSAYS:
        if assay not in df.columns:
            continue
        values = df[assay].dropna()
        invalid_mask = ~values.isin([0, 1, 0.0, 1.0])
        invalid_count = int(invalid_mask.sum())
        if invalid_count > 0:
            report["warnings"].append(
                f"{assay}: {invalid_count} values outside {{0, 1, NaN}}"
            )

    report["statistics"] = {
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "assays_present": int(sum(a in df.columns for a in TOX21_ASSAYS)),
    }

    return report



# Dataset splitting


def create_random_split(
    df: pd.DataFrame,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into training, validation, and test subsets
    using a random shuffle.
    """
    if abs(train_size + val_size + test_size - 1.0) > 1e-6:
        raise ValueError("Split proportions must sum to 1.0")

    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    n = len(df_shuffled)
    train_end = int(n * train_size)
    val_end = train_end + int(n * val_size)

    train_df = df_shuffled.iloc[:train_end].copy()
    val_df = df_shuffled.iloc[train_end:val_end].copy()
    test_df = df_shuffled.iloc[val_end:].copy()

    return train_df, val_df, test_df



# Dataset statistics


def compute_dataset_statistics(df: pd.DataFrame, split_name: str = "") -> Dict[str, Any]:
    """
    Compute basic statistics for a dataset split, including
    SMILES length distribution and per-assay label coverage.
    """
    stats: Dict[str, Any] = {
        "split_name": split_name,
        "n_molecules": int(len(df)),
        "n_columns": int(len(df.columns)),
        "smiles_lengths": None,
        "assays": {},
    }

    if "smiles" in df.columns:
        lengths = df["smiles"].astype(str).str.len()
        stats["smiles_lengths"] = {
            "mean": float(lengths.mean()),
            "median": float(lengths.median()),
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "std": float(lengths.std()),
        }

    for assay in TOX21_ASSAYS:
        if assay not in df.columns:
            continue

        col = df[assay]
        n_total = len(col)
        n_missing = int(col.isna().sum())
        n_valid = n_total - n_missing

        if n_valid > 0:
            n_positive = int((col == 1).sum())
            positive_rate = float(n_positive / n_valid * 100)
        else:
            n_positive = 0
            positive_rate = 0.0

        stats["assays"][assay] = {
            "n_total": int(n_total),
            "n_valid": int(n_valid),
            "n_missing": int(n_missing),
            "positive_rate": positive_rate,
        }

    return stats



# End-to-end pipeline


def download_and_prepare_tox21(
    output_dir: str = "data/raw",
    download_fresh: bool = False,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Execute the full Tox21 download and preparation workflow.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    gz_path = out / "tox21.csv.gz"
    csv_path = out / "tox21.csv"

    if download_fresh or (not gz_path.exists() and not csv_path.exists()):
        if not download_file(TOX21_URL, gz_path):
            if not download_file(TOX21_CSV_URL, csv_path):
                raise RuntimeError("Tox21 download failed")

    if gz_path.exists() and not csv_path.exists():
        if not decompress_gzip(gz_path, csv_path):
            raise RuntimeError("Failed to decompress tox21.csv.gz")

    df = load_tox21_csv(csv_path)

    validation = validate_tox21_data(df)
    if not validation["valid"]:
        raise ValueError(f"Dataset validation failed: {validation['issues']}")

    train_df, val_df, test_df = create_random_split(
        df, 0.8, 0.1, 0.1, random_state
    )

    train_df.to_csv(out / "train_raw.csv", index=False)
    val_df.to_csv(out / "val_raw.csv", index=False)
    test_df.to_csv(out / "test_raw.csv", index=False)

    metadata = {
        "source_url": TOX21_URL,
        "download_date": pd.Timestamp.now().isoformat(),
        "splits": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
            "random_state": random_state,
        },
        "assays": TOX21_ASSAYS,
        "validation": validation,
    }

    with open(out / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return train_df, val_df, test_df


def main() -> None:
    download_and_prepare_tox21(
        output_dir="data/raw",
        download_fresh=False,
        random_state=42,
    )


if __name__ == "__main__":
    main()