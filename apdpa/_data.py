###############################################
# _data.py
###############################################

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Default configuration.
# "condition" here corresponds to both cell lines and protein plates.
DEFAULT_CONFIG = {
    # SAMPLE TYPES
    "type_col": "type",                       # Column storing the sample type.
    "single_type": "singleDrug",              # Value for single perturbation samples.
    "no_type": "noDrug",                      # Value for no perturbation samples.
    "combo_type": "drugCombination",          # Value for combination samples.

    # PERTURBATION COLUMNS
    "perturbation_a_col": "anchor_drug",       # Column for the first perturbation.
    "perturbation_b_col": "library_drug",       # Column for the second perturbation.

    # CONDITION (e.g. cell line, protein plate)
    "condition_col": "protein_plate",         # Column representing the condition.
    # Optional group override; if you wish to force specific conditions to a split, set this equal to condition_col.
    "group_col": None,

    # PERTURBATION STRENGTH COLUMNS
    # For single perturbation samples, only strength_a is used.
    "strength_a_col": "perturbation_strength_a",   # Strength for perturbation A (used for single samples and combo A).
    "strength_b_col": "perturbation_strength_b",   # Strength for perturbation B (used only for combos).
    # For no perturbation samples, no strength is needed.
}


###########################################################
# 1) Global splitting of combo samples
###########################################################
def create_combo_split(
    adata,
    test_frac=0.2,
    val_frac=0.1,
    test_groups=None,
    val_groups=None,
    random_state=42,
    config=DEFAULT_CONFIG,
):
    """
    Globally splits samples with type config["combo_type"] (e.g. drugCombination).
    For each such row, a unique combo ID is derived by concatenating the values in
    config["perturbation_a_col"] and config["perturbation_b_col"] (with a plus sign).
    The unique combo IDs are then partitioned globally into train, val, and test so that
    if a combo (e.g. "A+B") appears in any condition, it is entirely assigned to one split.
    """
    rng = np.random.RandomState(random_state)
    if test_groups is None:
        test_groups = []
    if val_groups is None:
        val_groups = []

    type_col = config["type_col"]
    combo_type = config["combo_type"]
    pert_a = config["perturbation_a_col"]
    pert_b = config["perturbation_b_col"]
    group_col = config["group_col"]

    df = adata.obs.copy()
    df["obs_i"] = np.arange(df.shape[0])

    # Filter for combo samples and derive a unique combo ID.
    combos_df = df[df[type_col] == combo_type].copy()
    combos_df["combo_id"] = (
        combos_df[pert_a].astype(str) + "+" + combos_df[pert_b].astype(str)
    )

    # Get unique combo IDs and partition them globally.
    all_combos = combos_df["combo_id"].unique()
    rng.shuffle(all_combos)
    n = len(all_combos)
    n_test = int(round(test_frac * n))
    n_val = int(round(val_frac * n))

    test_set = set(all_combos[:n_test])
    val_set = set(all_combos[n_test:n_test+n_val])
    train_set = set(all_combos[n_test+n_val:])

    def assign_combo(row):
        c = row["combo_id"]
        if c in train_set:
            return "train"
        elif c in val_set:
            return "val"
        elif c in test_set:
            return "test"
        return "none"

    combos_df["split"] = combos_df.apply(assign_combo, axis=1)
    df.loc[combos_df.index, "split"] = combos_df["split"]

    # Optional: override based on group if provided.
    if group_col is not None:
        if len(test_groups) > 0:
            mask_test = df[group_col].isin(test_groups)
            df.loc[mask_test, "split"] = "test"
        if len(val_groups) > 0:
            mask_val = df[group_col].isin(val_groups)
            df.loc[mask_val, "split"] = "val"

    combos_only = df[df[type_col] == combo_type]
    train_idxs = combos_only.index[combos_only["split"] == "train"].tolist()
    val_idxs = combos_only.index[combos_only["split"] == "val"].tolist()
    test_idxs = combos_only.index[combos_only["split"] == "test"].tolist()

    train_combo_idxs = df.loc[train_idxs, "obs_i"].tolist()
    val_combo_idxs = df.loc[val_idxs, "obs_i"].tolist()
    test_combo_idxs = df.loc[test_idxs, "obs_i"].tolist()

    return train_combo_idxs, val_combo_idxs, test_combo_idxs


def get_all_single_perturbation_idx(adata, config=DEFAULT_CONFIG):
    """
    Returns row indices for all samples with type equal to config["single_type"]
    (e.g. singleDrug). For single perturbation samples, only the strength in the A role is relevant.
    """
    type_col = config["type_col"]
    single_type = config["single_type"]
    mask = adata.obs[type_col] == single_type
    return np.where(mask)[0].tolist()


###########################################################
# 2) Dataset #1: SinglePerturbation Dataset
###########################################################
class SinglePerturbationDataset(Dataset):
    """
    For each sample with type config["single_type"] (i.e. singleDrug) in the primary indices,
    returns a 3-tuple:
         (X_noPerturbation, X_singlePerturbation, strength_singlePerturbation)
    where the noPerturbation sample is randomly chosen from the same condition.
    
    Note: The sampling dictionaries are built from pool_indices, which should contain all indices
    (or at least those of types no_type and single_type).
    """
    def __init__(self, primary_indices, pool_indices, adata, config=DEFAULT_CONFIG, seed=42):
        """
        primary_indices: indices for single perturbation samples to iterate over.
        pool_indices: indices used to build dictionaries (should include all noPerturbation samples).
        """
        self.adata = adata
        self.config = config
        self.condition_col = config["condition_col"]
        self.rng = np.random.RandomState(seed)
        type_col = config["type_col"]
        single_type = config["single_type"]
        no_type = config["no_type"]

        self.primary_indices = primary_indices
        
        # Build dictionary for noPerturbation samples from the pool.
        self.condition2noPerturbation = {}
        for i in pool_indices:
            row = adata.obs.iloc[i]
            if row[type_col] == no_type:
                cond = row[self.condition_col]
                self.condition2noPerturbation.setdefault(cond, []).append(i)
        for cond in self.condition2noPerturbation:
            self.condition2noPerturbation[cond] = np.array(self.condition2noPerturbation[cond], dtype=int)

    def __len__(self):
        return len(self.primary_indices)

    def __getitem__(self, idx):
        strength_a_col = self.config["strength_a_col"]
        perturbation_a_col = self.config["perturbation_a_col"]
        single_idx = self.primary_indices[idx]
        row_single = self.adata.obs.iloc[single_idx]
        cond = row_single[self.config["condition_col"]]

        if cond not in self.condition2noPerturbation or len(self.condition2noPerturbation[cond]) == 0:
            raise ValueError(f"No noPerturbation sample found in condition: {cond}")
        noPerturbation_idx = self.rng.choice(self.condition2noPerturbation[cond])

        X_single = self.adata.X[single_idx]
        strength_single = self.adata.obs.iloc[single_idx][strength_a_col]
        perturbation_single = self.adata.obs.iloc[single_idx][perturbation_a_col]
        X_noPerturbation = self.adata.X[noPerturbation_idx]

        X_single = torch.tensor(X_single, dtype=torch.float32)
        strength_single = torch.tensor(float(str(strength_single)), dtype=torch.float32)
        X_noPerturbation = torch.tensor(X_noPerturbation, dtype=torch.float32)

        return (X_noPerturbation, X_single, perturbation_single, strength_single, cond)


###########################################################
# 3) Dataset #2: ComboPerturbation Dataset
###########################################################
class ComboPerturbationDataset(Dataset):
    """
    For each sample with type config["combo_type"] (e.g. drugCombination) in the primary indices,
    returns a 9-tuple:
      (X_combo, strength_combo_a, strength_combo_b,
       X_perturbation_a, strength_a,
       X_perturbation_b, strength_b,
       X_no1, X_no2)
    where:
      - X_combo is the feature vector of the combo sample.
      - For the combo sample, strength values are taken from config["strength_a_col"] and config["strength_b_col"].
      - X_perturbation_a is a randomly chosen single perturbation sample (from the pool) whose value in
        config["perturbation_a_col"] matches the combo sample's value; its strength is taken from config["strength_a_col"].
      - X_perturbation_b is similarly chosen using config["perturbation_b_col"] and its strength from config["strength_b_col"].
      - X_no1 and X_no2 are two randomly selected noPerturbation samples from the same condition.
        (No strength is returned for noPerturbation samples.)
    """
    def __init__(self, primary_indices, pool_indices, adata, config=DEFAULT_CONFIG, seed=42):
        """
        primary_indices: indices for combo samples to iterate over.
        pool_indices: indices used to build sampling dictionaries (should include all single and no samples).
        """
        self.adata = adata
        self.config = config
        self.condition_col = config["condition_col"]
        self.perturbation_a_col = config["perturbation_a_col"]
        self.perturbation_b_col = config["perturbation_b_col"]
        self.strength_a_col = self.config["strength_a_col"]
        self.strength_b_col = self.config["strength_b_col"]
        self.rng = np.random.RandomState(seed)
        type_col = config["type_col"]
        combo_type = config["combo_type"]
        single_type = config["single_type"]
        no_type = config["no_type"]

        self.primary_indices = primary_indices  # Only these are iterated over.

        # Build dictionaries from the pool.
        self.condition_perturbation2singles = {}
        self.condition2noPerturbation = {}

        for i in pool_indices:
            row = adata.obs.iloc[i]
            cond = row[self.condition_col]
            if row[type_col] == single_type:
                key_single = (cond, row[self.perturbation_a_col])
                self.condition_perturbation2singles.setdefault(key_single, []).append(i)
            elif row[type_col] == no_type:
                self.condition2noPerturbation.setdefault(cond, []).append(i)

        for k in self.condition_perturbation2singles:
            self.condition_perturbation2singles[k] = np.array(self.condition_perturbation2singles[k], dtype=int)
        for k in self.condition2noPerturbation:
            self.condition2noPerturbation[k] = np.array(self.condition2noPerturbation[k], dtype=int)

    def __len__(self):
        return len(self.primary_indices)

    def __getitem__(self, idx):

        primary_idx = self.primary_indices[idx]
        row_combo = self.adata.obs.iloc[primary_idx]
        cond = row_combo[self.config["condition_col"]]
        pert_a = row_combo[self.perturbation_a_col]
        pert_b = row_combo[self.perturbation_b_col]

        X_combo = self.adata.X[primary_idx]
        strength_combo_a = self.adata.obs.iloc[primary_idx][self.strength_a_col]
        strength_combo_b = self.adata.obs.iloc[primary_idx][self.strength_b_col]

        # Sample a single perturbation for A.
        key_a = (cond, pert_a)
        if key_a not in self.condition_perturbation2singles or len(self.condition_perturbation2singles[key_a]) == 0:
            raise ValueError(f"No single sample for perturbation_a {pert_a} in condition {cond}")
        idx_a = self.rng.choice(self.condition_perturbation2singles[key_a])
        X_a = self.adata.X[idx_a]
        strength_a = self.adata.obs.iloc[idx_a][self.strength_a_col]
        perturbation_a = self.adata.obs.iloc[idx_a][self.perturbation_a_col]

        # Sample a single perturbation for B.
        key_b = (cond, pert_b)
        if key_b not in self.condition_perturbation2singles or len(self.condition_perturbation2singles[key_b]) == 0:
            raise ValueError(f"No single sample for perturbation_b {pert_b} in condition {cond}")
        idx_b = self.rng.choice(self.condition_perturbation2singles[key_b])
        X_b = self.adata.X[idx_b]
        strength_b = self.adata.obs.iloc[idx_b][self.strength_a_col]
        perturbation_b = self.adata.obs.iloc[idx_b][self.perturbation_a_col]

        # Sample two noPerturbation samples from the same condition.
        if cond not in self.condition2noPerturbation or len(self.condition2noPerturbation[cond]) < 2:
            raise ValueError(f"Not enough noPerturbation samples in condition {cond} (need 2).")
        no_indices = self.rng.choice(self.condition2noPerturbation[cond], size=2, replace=False)
        X_no1 = self.adata.X[no_indices[0]]
        X_no2 = self.adata.X[no_indices[1]]
        # No strength for noPerturbation samples.

        X_combo = torch.tensor(X_combo, dtype=torch.float32)
        strength_combo_a = torch.tensor(float(str(strength_combo_a)), dtype=torch.float32)
        strength_combo_b = torch.tensor(float(str(strength_combo_b)), dtype=torch.float32)
        X_a = torch.tensor(X_a, dtype=torch.float32)
        strength_a = torch.tensor(float(str(strength_a)), dtype=torch.float32)
        X_b = torch.tensor(X_b, dtype=torch.float32)
        strength_b = torch.tensor(float(str(strength_b)), dtype=torch.float32)
        X_no1 = torch.tensor(X_no1, dtype=torch.float32)
        X_no2 = torch.tensor(X_no2, dtype=torch.float32)

        return (X_combo, strength_combo_a, strength_combo_b,
                X_a, strength_a, perturbation_a,
                X_b, strength_b, perturbation_b,
                X_no1, X_no2, cond)


###########################################################
# Example usage: creating DataLoaders
###########################################################
def get_data_loaders(adata, config=DEFAULT_CONFIG, seed=42, 
                     batch_size_single=64, batch_size_combo=32,
                     test_frac=0.2, val_frac=0.1):
    """
    Splits the data and creates DataLoaders.
    
    1) Globally splits combo samples using create_combo_split.
    2) Collects all single samples using get_all_single_perturbation_idx.
    3) Uses the entire dataset (all indices) as the pool for building sampling dictionaries.
    
    Returns:
      train_loader_1: DataLoader for the singlePerturbation dataset.
      train_loader_2: DataLoader for the comboPerturbation dataset.
      val_loader_2:   DataLoader for validation combo samples.
      test_loader_2:  DataLoader for test combo samples.
    """
    # Use the entire dataset as the pool.
    pool_indices = list(range(len(adata.obs)))
    
    # Split combo samples globally.
    train_combo_idxs, val_combo_idxs, test_combo_idxs = create_combo_split(
        adata, config=config, test_frac=test_frac, val_frac=val_frac, random_state=seed)
    
    # Get all single perturbation indices (for training only).
    train_single_idxs = get_all_single_perturbation_idx(adata, config=config)
    
    # Build dataset objects using both primary and pool indices.
    train_ds_1 = SinglePerturbationDataset(train_single_idxs, pool_indices, adata, config=config, seed=seed)
    train_ds_2 = ComboPerturbationDataset(train_combo_idxs, pool_indices, adata, config=config, seed=seed)
    
    # Validation and test datasets (for combos) use pool_indices for auxiliary sampling.
    val_ds_2 = ComboPerturbationDataset(val_combo_idxs, pool_indices, adata, config=config, seed=seed)
    test_ds_2 = ComboPerturbationDataset(test_combo_idxs, pool_indices, adata, config=config, seed=seed)
    
    # Create DataLoaders.
    train_loader_1 = DataLoader(train_ds_1, batch_size=batch_size_single, shuffle=True)
    train_loader_2 = DataLoader(train_ds_2, batch_size=batch_size_combo, shuffle=True)
    val_loader_2 = DataLoader(val_ds_2, batch_size=batch_size_combo, shuffle=False)
    test_loader_2 = DataLoader(test_ds_2, batch_size=batch_size_combo, shuffle=False)
    
    return train_loader_1, train_loader_2, val_loader_2, test_loader_2
