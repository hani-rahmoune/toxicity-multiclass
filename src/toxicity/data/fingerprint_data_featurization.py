
#pip install rdkit

import numpy as np
import pandas as pd
from typing import Optional, Union, List, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.Fingerprints import FingerprintMols
from tqdm import tqdm
import warnings
# Suppress RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Convert a SMILES string into a Morgan (ECFP) fingerprint.
# The fingerprint encodes local atomic environments as a fixed-length
# binary NumPy array (0/1), suitable for classical ML models such as XGBoost.

def generate_morgan_fingerprint(
    smiles:str ,
    radius:int = 2,
    n_bits:int = 2048,
    use_features: bool = False,
    use_chirality: bool = False
) ->Optional[np.ndarray]:
  try :
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
      return None
    if use_features:
      fp = AllChem.GetMorganFingerprintAsBitVect(
          mol,
          radius=radius,
          nBits=n_bits,
          useFeatures=True,
          useChirality=use_chirality
          )
    else :
      fp = AllChem.GetMorganFingerprintAsBitVect(
          mol,
          radius=radius,
          nBits=n_bits,
          useChirality=use_chirality
      )
    arr = np.zeros((n_bits,),dtype = np.int64)
    Chem.DataStructs.ConvertToNumpyArray(fp,arr)

    return arr

  except Exception as e :
    return None

# Compute physicochemical molecular descriptors from a SMILES string.
# Returns a fixed-order NumPy array of real-valued properties (e.g. MolWt, LogP, TPSA),
# which describe global molecular behavior rather than local structure.
# These descriptors are concatenated with Morgan fingerprints to enrich
# classical ML models (XGBoost here) with size, polarity, and complexity information.


def generate_descriptors(smiles:str, descriptors : Optional[List[str]]=None) -> Optional[np.ndarray]:
  try:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
      return None
    # Default set of descriptors
    if descriptors is None:
      descriptors = [
                'MolWt',              # Molecular weight
                'MolLogP',            # LogP (lipophilicity)
                'NumHDonors',         # H-bond donors
                'NumHAcceptors',      # H-bond acceptors
                'NumRotatableBonds',  # Rotatable bonds
                'NumAromaticRings',   # Aromatic rings
                'TPSA',              # Topological polar surface area
                'NumSaturatedRings',  # Saturated rings
                'NumAliphaticRings',  # Aliphatic rings
                'RingCount',          # Total rings
                'FractionCsp3',       # Fraction sp3 carbons
                'NumHeteroatoms',     # Non-C/H atoms
                'HeavyAtomCount',     # Heavy atoms
                'NumValenceElectrons', # Valence electrons
                'MaxPartialCharge',   # Max partial charge
                'MinPartialCharge',   # Min partial charge
                'MaxAbsPartialCharge', # Max absolute charge
                'BalabanJ',           # Balaban index
                'BertzCT',            # Complexity
                'Chi0v',              # Connectivity index
            ]
    values = []
    for descriptor in descriptors:
      try:
        descriptor_func = getattr(Descriptors, descriptor)
        value = descriptor_func(mol)
        values.append(value)
      except Exception as e:
        values.append(np.nan)
    return np.array(values)
  except Exception as e:
    return None

# Build model inputs from a DataFrame:
# - read the SMILES column
# - generate a fingerprint (and optional descriptors) for each molecule
# - drop rows where the SMILES cannot be featurized
# Returns:
# - X: a 2D NumPy array where each row is one molecule’s fingerprint/features
# - df_valid: the filtered DataFrame matching X (same row order)

def batch_generate_fingerprints(
    smiles_list: List[str],
    show_progress: bool = True,
    radius: int = 2,
    n_bits: int = 2048,
    use_features: bool = False,
    use_chirality: bool = False,
    add_descriptors: bool = False,
    descriptor_names: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[int]]:

    fingerprints = []
    failed_indices = []

    iterator = tqdm(
        enumerate(smiles_list),
        total=len(smiles_list),
        desc="Generating Morgan fingerprints",
        disable=not show_progress
    )

    for idx, smiles in iterator:
        fp = generate_morgan_fingerprint(
            smiles,
            radius=radius,
            n_bits=n_bits,
            use_features=use_features,
            use_chirality=use_chirality
        )

        if fp is None:
            failed_indices.append(idx)
            continue

        if add_descriptors:
            desc = generate_descriptors(smiles, descriptors=descriptor_names)
            if desc is None:
                failed_indices.append(idx)
                continue
            fp = np.concatenate([fp.astype(np.float32), desc.astype(np.float32)], axis=0)

        fingerprints.append(fp)

    if fingerprints:
        fingerprints = np.array(fingerprints)
    else:
        fingerprints = np.array([]).reshape(0, 0)

    n_total = len(smiles_list)
    n_success = len(fingerprints)
    n_failed = len(failed_indices)

    print(f"\n Fingerprint generation complete:")
    print(f"   Total molecules:  {n_total:,}")
    print(f"   Success:          {n_success:,} ({(n_success/n_total*100) if n_total else 0:.2f}%)")
    print(f"   Failed:           {n_failed:,} ({(n_failed/n_total*100) if n_total else 0:.2f}%)")
    print(f"   Output shape:     {fingerprints.shape}")

    return fingerprints, failed_indices

# This function:
# extracts SMILES from the given column
# generates Morgan fingerprints (and optional descriptors)
# removes molecules that cannot be featurized
# returns a feature matrix and a cleaned DataFrame that stay aligned
#
# The goal is to safely go from raw chemical data to X (features) + clean metadata
# without breaking the link between molecules and their labels.
def featurize_dataframe(
    df: pd.DataFrame,
    smiles_col: str = 'smiles',
    radius: int = 2,
    n_bits: int = 2048,
    use_features: bool = False,
    use_chirality: bool = False,
    add_descriptors: bool = False,
    descriptor_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    print(f" Featurizing DataFrame...")
    print(f"   Rows: {len(df):,}")
    print(f"   SMILES column: '{smiles_col}'")

    smiles_list = df[smiles_col].tolist()

    fingerprints, failed_indices = batch_generate_fingerprints(
        smiles_list,
        show_progress=True,
        radius=radius,
        n_bits=n_bits,
        use_features=use_features,
        use_chirality=use_chirality,
        add_descriptors=add_descriptors,
        descriptor_names=descriptor_names
    )

    if failed_indices:
        df_valid = df.drop(index=df.index[failed_indices]).reset_index(drop=True)
        print(f"\n Removed {len(failed_indices)} molecules with failed featurization")
    else:
        df_valid = df.copy()

    return fingerprints, df_valid

if __name__ == "__main__":
    test_smiles = ["CCO", "c1ccccc1", "INVALID", "CC(=O)O"]

    print("\n--- Morgan only ---")
    X, failed = batch_generate_fingerprints(test_smiles, add_descriptors=False)
    print("X:", X.shape, "failed:", failed)

    print("\n--- Morgan + descriptors ---")
    X2, failed2 = batch_generate_fingerprints(test_smiles, add_descriptors=True)
    print("X2:", X2.shape, "failed:", failed2)