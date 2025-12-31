

from rdkit import Chem

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from rdkit.Chem import SaltRemover, Descriptors
from tqdm import tqdm
import warnings

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Validate a SMILES string:
# - checks type and non-emptiness
# - tries to parse it with RDKit
# - returns (True, None) if valid
# - returns (False, "reason") if invalid

def validation_smile(smiles : str) -> Tuple[bool , Optional[str]] :
  if not isinstance(smiles, str) :
    return False , "not a string"
  if(len(smiles.strip())==0) :
    return False , "empty string"
  try:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None :
      return False , "invalid smile"
    if mol.GetNumAtoms() == 0 :
      return False , "invalid smile"
    return True , None
  except Exception as e :
    return False , str(e)

# Turn a SMILES string into a single standard form.
# The same molecule can be written in many ways, so this
# makes sure we always use one consistent version.
# Returns None if the SMILES is invalid or cannot be read.

def canonicalize_smile(smiles : str) -> Optional[str]:
  try:
     mol = Chem.MolFromSmiles(smiles)
     if mol is None :
       return None
     canonical =Chem.MolToSmiles(mol,canonical=True)
     return canonical
  except Exception as e :
     return None

# Remove salts and counterions from a SMILES string.
# Some molecules are written with extra fragments (e.g. ".Cl", ".Na")
# that are used for storage or formulation but are not part of the
# active drug. This function removes those fragments and, if multiple
# pieces remain, keeps the largest one as the main molecule.
# Returns a cleaned SMILES string or None if the input is invalid.

def remove_salts(smiles :str , keep_largest : bool =True)->Optional[str]:
  try:
    mol =Chem.MolFromSmiles(smiles)
    if mol is None: return None
    remover = SaltRemover.SaltRemover()
    mol_desalted = remover.StripMol(mol,dontRemoveEverything=True)
    if keep_largest :
      fragments =Chem.GetMolFrags(mol_desalted, asMols=True,sanitizeFrags=True)
      if len(fragments) == 0 :
        return None
      if len(fragments)>1:
        mol_desalted = max(fragments,key=lambda x:x.GetNumHeavyAtoms())
    return Chem.MolToSmiles(mol_desalted,canonical = True)
  except Exception as e :
    return None

# Filter molecules based on size (number of heavy atoms).
# Very small molecules are usually solvents, ions, or fragments,
# while very large molecules are rare in drug datasets and can
# cause issues for modeling. This function checks whether a
# SMILES string falls within a reasonable size range.
# Returns (True, None) if acceptable, or (False, reason) otherwise

def filter_by_size(smiles : str ,
    min_heavy_atoms: int = 3,
    max_heavy_atoms: int = 150
    )->Tuple[bool,Optional[str]] :
  try:
    mol =Chem.MolFromSmiles(smiles)
    if mol is None: return False , "invalid"

    num_of_heavy_mols = mol.GetNumHeavyAtoms()
    if num_of_heavy_mols < min_heavy_atoms:
      return False ,"too small"
    if num_of_heavy_mols > max_heavy_atoms:
      return False ,"too large"
    return True , None
  except Exception as e :
    return False , str(e)


# Clean a SMILES string using an ordered pipeline.
# The function validates the input, removes salts if requested,
# canonicalizes the molecule, and filters it by size.
# It returns the cleaned SMILES (or None if cleaning fails)
# along with a dictionary explaining what happened.
# The `verbose` flag controls whether messages are printed
# during the cleaning process.

def clean_molecule(
    smiles: str,
    canonicalize: bool = True,
    remove_salt: bool = True,
    min_atoms: int = 3,
    max_atoms: int = 150,
    verbose: bool = False
) -> Tuple[Optional[str], Dict[str, Any]]:

    info = {
        'original': smiles,
        'valid': False,
        'reason': None,
        'steps': [],
        'changed': False
    }

    # validation
    isvalid, error = validation_smile(smiles)
    if not isvalid:
        info['reason'] = f"validation failed: {error}"
        if verbose:
            print(info['reason'])
        return None, info

    cleaned = smiles

    # salts removing
    if remove_salt:
        cleaned_temp = remove_salts(cleaned)
        if cleaned_temp is None:
            info['reason'] = "salts removal failed"
            if verbose:
                print(info['reason'])
            return None, info

        if cleaned != cleaned_temp:
            info['steps'].append('desalted')
            info['changed'] = True

        cleaned = cleaned_temp  

    # canonicalize
    if canonicalize:
        cleaned_temp = canonicalize_smile(cleaned)
        if cleaned_temp is None:
            info['reason'] = "canonicalization failed"
            if verbose:
                print(info['reason'])
            return None, info

        if cleaned != cleaned_temp:
            info['steps'].append('canonicalized')
            info['changed'] = True

        cleaned = cleaned_temp  

    # size check
    isvalid, error = filter_by_size(cleaned, min_atoms, max_atoms)
    if not isvalid:
        info['reason'] = f"size check failed: {error}"
        if verbose:
            print(info['reason'])
        return None, info

    info['steps'].append('size_checked')

    # success
    info['valid'] = True
    info['cleaned'] = cleaned

    if verbose:
        if info['changed']:
            print("molecule was modified")
        else:
            print("no changes were needed")

    return cleaned, info

# Clean an entire DataFrame of SMILES strings.
# Applies the clean_molecule pipeline row by row, tracks failures,
# removes invalid entries and duplicates, and returns:
# 1) a cleaned DataFrame
# 2) summary statistics describing what happened

def clean_dataframe(
    df : pd.DataFrame ,
    smiles_col : str ='smiles' ,
    show_progress: bool = True ,
    return_audit : bool = False ,
    **cleaning_kwargs
)->Tuple[pd.DataFrame,Dict[str,Any]]:

  #validating the input
  if smiles_col not in df.columns :
    raise ValueError(f"column {smiles_col} not found in the dataframe")

  cleaned_smiles = []
  failed_reasons = []
  num_steps = []
  #progress bar
  iterator = tqdm(df[smiles_col], desc="Cleaning molecules", disable=not show_progress)
  #cleaning each mol
  for smiles in iterator:
    cleaned , info = clean_molecule(smiles,verbose= False,**cleaning_kwargs)
    if cleaned is not None :
      cleaned_smiles.append(cleaned)
      failed_reasons.append(None)
      num_steps.append(len(info['steps']))
    else :
      cleaned_smiles.append(None)
      failed_reasons.append(info['reason'])
      num_steps.append(0)

  df_result = df.copy()
  df_result['_cleaned_smiles'] = cleaned_smiles
  df_result['_cleaning_reason'] = failed_reasons
  df_result['_cleaning_steps'] = num_steps

  #remove failed rows
  n_before_removal=len(df)
  df_clean = df_result[df_result['_cleaned_smiles'].notna()].copy()
  n_failed = n_before_removal - len(df_clean)

  #replace og with cleaned smiles
  df_clean[smiles_col]=df_clean['_cleaned_smiles']
  #remove temp cols
  df_clean = df_clean.drop(columns=['_cleaned_smiles','_cleaning_reason','_cleaning_steps'])

  #remove dups
  n_before_dedup = len(df_clean)
  df_clean = df_clean.drop_duplicates(subset=[smiles_col],keep ='first')
  n_duplicates = n_before_dedup - len(df_clean)
  #stats
  failure_counts = {}
  if n_failed > 0:
      failure_series = pd.Series([r for r in failed_reasons if r is not None])
      failure_counts = failure_series.value_counts().to_dict()

  stats = {
      'original_count': len(df),
      'cleaned_count': len(df_clean),
      'failed_count': n_failed,
      'failed_percentage': n_failed / len(df) * 100 if len(df) > 0 else 0,
      'duplicates_removed': n_duplicates,
      'duplicates_percentage': n_duplicates / len(df) * 100 if len(df) > 0 else 0,
      'failure_reasons': failure_counts,
      'final_percentage': len(df_clean) / len(df) * 100 if len(df) > 0 else 0
  }
  print(f"\n Cleaning Summary:")
  print(f"   Original molecules:    {stats['original_count']:,}")
  print(f"   Failed cleaning:    {stats['failed_count']:,} ({stats['failed_percentage']:.2f}%)")
  print(f"   Duplicates removed: {stats['duplicates_removed']:,} ({stats['duplicates_percentage']:.2f}%)")
  print(f"   Final clean data:   {stats['cleaned_count']:,} ({stats['final_percentage']:.2f}%)")


  if failure_counts:
    print(f"\n   Failure breakdown:")
    for reason, count in sorted(failure_counts.items(), key=lambda x: x[1], reverse=True):
      print(f" {reason}: {count:,}")
  if return_audit:
    return df_clean, stats, df_result
  else:
    return df_clean, stats

# Quickly validate a list of SMILES strings without modifying them.
# This function is meant for fast data quality checks before running
# the full (and more expensive) cleaning pipeline.
# It reports how many SMILES are valid or invalid and where the
# invalid entries are located.

def batch_validate(smiles_list : List[str])->Dict[str,Any] :
  results = [validation_smile(s)for s in smiles_list]
  valid_flags=[r[0] for r in results]
  return {
        'total': len(smiles_list),
        'valid': sum(valid_flags),
        'invalid': len(smiles_list) - sum(valid_flags),
        'valid_percentage': sum(valid_flags) / len(smiles_list) * 100 if smiles_list else 0,
        'invalid_indices': [i for i, v in enumerate(valid_flags) if not v]
    }

# Compare two SMILES strings by molecule identity, not by text.
# Different SMILES strings can describe the same molecule, so this
# function canonicalizes both inputs and checks if they match.
# Returns False if either SMILES is invalid.
def compare_smiles(smiles1 : str , smiles2 : str)->bool :
  can1 = canonicalize_smile(smiles1)
  can2 = canonicalize_smile(smiles2)
  if can1 is None or can2 is None :
    return False
  return can1 == can2

