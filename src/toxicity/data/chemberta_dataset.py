
# PyTorch Dataset for:
# This module is responsible ONLY for data handling:
# - converting a DataFrame row (SMILES + labels) into tensors
# - handling missing toxicity labels via masking
# - preparing batches compatible with transformer models


import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List
from transformers import AutoTokenizer



class Tox21ChemBERTaDataset(Dataset):
   # Each Dataset item corresponds to ONE molecule:
   # - one SMILES string
   # - one vector of toxicity labels (length = 12)
   # - a mask indicating which labels are present (not NaN)

    #The Dataset:
    #- tokenizes SMILES into input_ids + attention_mask
    #- replaces missing labels with 0
    #- provides a label_mask so the loss can ignore missing labels

    #The Dataset NEVER returns batches.
    #It only defines how to build ONE sample.
    
    
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        smiles_col: str,
        task_cols: List[str],
        max_length: int = 128
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.smiles_col = smiles_col
        self.task_cols = task_cols
        self.max_length = max_length

        self._validate_data()

    
    # Data validation
    # Ensures SMILES column and task columns exist.
    def _validate_data(self):
        if self.smiles_col not in self.df.columns:
            raise ValueError(f"Missing SMILES column: {self.smiles_col}")

        missing_tasks = [c for c in self.task_cols if c not in self.df.columns]
        if missing_tasks:
            raise ValueError(f"Missing task columns: {missing_tasks}")

    
    # Dataset length
    # Required by PyTorch Dataset.
    def __len__(self) -> int:
        return len(self.df)

    
    # Single sample access
    # Returns ONE sample.
    # Batching happens later inside DataLoader.
    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # SMILES string
        smiles = str(row[self.smiles_col])

        # Labels for all tasks (may contain NaN)
        labels = row[self.task_cols].values.astype(np.float32)

        # Mask: 1 = valid label, 0 = missing label
        label_mask = (~np.isnan(labels)).astype(np.float32)

        # Replace NaN labels with 0 (ignored later via mask)
        labels = np.nan_to_num(labels, nan=0.0)

        # Tokenize SMILES
        encoding = self.tokenizer(
            smiles,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Remove tokenizer batch dimension
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(labels, dtype=torch.float32),
            "label_mask": torch.tensor(label_mask, dtype=torch.float32),
        }

    
    # Label statistics (optional analysis helper)
    # Computes class balance and missing-label rates per task.
    def get_label_statistics(self) -> dict:
        stats = {}

        for task in self.task_cols:
            values = self.df[task].values
            valid = ~np.isnan(values)
            valid_values = values[valid]

            if len(valid_values) == 0:
                stats[task] = {
                    "n_total": len(values),
                    "n_valid": 0,
                    "n_missing": len(values),
                    "positive_rate": 0.0,
                }
            else:
                stats[task] = {
                    "n_total": len(values),
                    "n_valid": len(valid_values),
                    "n_missing": len(values) - len(valid_values),
                    "positive_rate": float((valid_values == 1).mean()),
                }

        return stats



# DATALOADER FACTORY

# Convenience function to create train / val / test loaders.
def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    smiles_col: str,
    task_cols: List[str],
    max_length: int = 128,
    batch_size: int = 32,
    num_workers: int = 0,
):
    from torch.utils.data import DataLoader

    train_dataset = Tox21ChemBERTaDataset(
        train_df, tokenizer, smiles_col, task_cols, max_length
    )
    val_dataset = Tox21ChemBERTaDataset(
        val_df, tokenizer, smiles_col, task_cols, max_length
    )
    test_dataset = Tox21ChemBERTaDataset(
        test_df, tokenizer, smiles_col, task_cols, max_length
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader