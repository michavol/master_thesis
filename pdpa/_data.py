###############################################
# _data.py
###############################################

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Default configuration. All names are configurable.
# Note: In this setting, "condition" corresponds to both cell line and protein plate.
DEFAULT_CONFIG = {
    "type_col": "type",                       # Column in adata.obs storing the sample type.
    "single_type": "singleDrug",      # Value for single perturbation samples.
    "no_type": "noDrug",              # Value for no perturbation samples.
    "combo_type": "drugCombination",        # Value for combination samples.
    "perturbation_a_col": "anchor_drug",     # Column for the first perturbation in a combo.
    "perturbation_b_col": "library_drug",     # Column for the second perturbation in a combo.
    "condition_col": "protein_plate",             # Column representing the condition (cell line/protein plate).
    # If you wish to force certain conditions to a specific split (e.g. protein plates),
    # set group_col equal to condition_col. Otherwise, leave as None.
    "group_col": None,                        
    "strength_col": "perturbation_strength",  # Column with the perturbation strength.
}


###############################################
# _data.py
###############################################

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Default configuration.
# "condition" here corresponds to both cell line and protein plate.
DEFAULT_CONFIG = {
    # SAMPLE TYPES
    "type_col": "type",                       # Column in adata.obs storing the sample type.
    "single_type": "singleDrug",      # Value for single perturbation samples.
    "no_type": "noDrug",              # Value for no perturbation samples.
    "combo_type": "drugCombination",        # Value for combination samples.

    # CONDITIONS
    "condition_col": "protein_plate",             # Column representing the condition (cell line/protein plate).
    "group_col": None,   # Optional group override; if you wish to force specific conditions to a split, set this equal to condition_col.

    # PERTURBATIONS                        
    "perturbation_a_col": "anchor_drug",     # Column for the first perturbation in a combo.
    "perturbation_b_col": "library_drug",     # Column for the second perturbation in a combo.
    "strength_a_col": "perturbation_strength_a",   # Strength for perturbation A (used for single samples and combo A).
    "strength_b_col": "perturbation_strength_b",   # Strength for perturbation B (used only for combos).
    # For noPerturbation samples, no strength is needed.
}


###########################################################
# 1) Global splitting of comboPerturbation samples and isolation of singlePerturbation samples.
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
    Globally splits samples with type config["combo_type"] (i.e. comboPerturbation).
    For each such row, a unique combo ID is derived by concatenating the values in
    config["perturbation_a_col"] and config["perturbation_b_col"] (with a plus sign).
    The unique combo IDs are then partitioned globally into train, val, and test so that if a
    combo (e.g. "A+B") appears in any condition, it is entirely assigned to one split.
    
    Parameters
    ----------
    adata : AnnData-like object
        Expects adata.obs to contain at least the columns defined in config.
    test_frac : float
        Fraction of unique combos to assign to test.
    val_frac : float
        Fraction of unique combos to assign to val.
    test_groups : list of str, optional
        If provided and if config["group_col"] is not None, all rows with that group value are forced to test.
    val_groups : list of str, optional
        Similarly for validation.
    random_state : int
        Seed for reproducible shuffling.
    config : dict
        Configuration dictionary.
    
    Returns
    -------
    train_combo_idxs, val_combo_idxs, test_combo_idxs : lists of int
        Row indices (0-based, as in adata.obs) for samples with type equal to config["combo_type"]
        assigned to train, val, and test.
    """
    rng = np.random.RandomState(random_state)
    if test_groups is None:
        test_groups = []
    if val_groups is None:
        val_groups = []

    type_col = config["type_col"]
    combo_type = config["combo_type"]
    perturbation_a_col = config["perturbation_a_col"]
    perturbation_b_col = config["perturbation_b_col"]
    group_col = config["group_col"]

    df = adata.obs.copy()
    df["obs_i"] = np.arange(df.shape[0])

    # Filter for comboPerturbation rows and derive a unique combo ID.
    combos_df = df[df[type_col] == combo_type].copy()
    combos_df["combo_id"] = (
        combos_df[perturbation_a_col].astype(str) +
        "+" +
        combos_df[perturbation_b_col].astype(str)
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

    # Optional: Override based on group if provided.
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
    (e.g. singlePerturbation). For single perturbation samples, only the strength in the
    A role is relevant.
    """
    type_col = config["type_col"]
    single_type = config["single_type"]
    mask = adata.obs[type_col] == single_type
    return np.where(mask)[0].tolist()

###########################################################
# 2) Dataset #1: noPerturbation + singlePerturbation Pairs
###########################################################
class SinglePerturbationDataset(Dataset):
    """
    For each sample with type config["single_type"] (i.e. singlePerturbation) in the given indices,
    returns a 3-tuple:
         (X_noPerturbation, X_singlePerturbation, strength_singlePerturbation)
    where the noPerturbation sample is randomly chosen from the same condition.
    (Note: For noPerturbation samples, no strength is returned.)
    """
    def __init__(self, adata, indices, config=DEFAULT_CONFIG, seed=42):
        super().__init__()
        self.adata = adata
        self.config = config
        self.condition_col = config["condition_col"]
        self.rng = np.random.RandomState(seed)

        type_col = config["type_col"]
        single_type = config["single_type"]
        no_type = config["no_type"]

        self.single_obs_list = []
        self.condition2noPerturbation = {}

        for i in indices:
            row = adata.obs.iloc[i]
            typ = row[type_col]
            cond = row[self.condition_col]

            if typ == no_type:
                self.condition2noPerturbation.setdefault(cond, []).append(i)
            elif typ == single_type:
                self.single_obs_list.append(i)

        for cond in self.condition2noPerturbation:
            self.condition2noPerturbation[cond] = np.array(self.condition2noPerturbation[cond], dtype=int)

    def __len__(self):
        return len(self.single_obs_list)

    def __getitem__(self, idx):
        # For singlePerturbation samples, use strength from strength_a_col.
        strength_a_col = self.config["strength_a_col"]
       
        single_idx = self.single_obs_list[idx]
        row_single = self.adata.obs.iloc[single_idx]
        cond = row_single[self.config["condition_col"]]

        if cond not in self.condition2noPerturbation or len(self.condition2noPerturbation[cond]) == 0:
            raise ValueError(f"No noPerturbation sample found in condition: {cond}")

        noPerturbation_idx = self.rng.choice(self.condition2noPerturbation[cond])

        X_single = self.adata.X[single_idx]
        strength_single = self.adata.obs.iloc[single_idx][strength_a_col]

        X_noPerturbation = self.adata.X[noPerturbation_idx]
        # For noPerturbation samples, we ignore strength.

        X_single = torch.tensor(X_single, dtype=torch.float32)
        strength_single = torch.tensor(float(str(strength_single)), dtype=torch.float32)
        X_noPerturbation = torch.tensor(X_noPerturbation, dtype=torch.float32)

        return (X_noPerturbation, X_single, strength_single)


###########################################################
# 3) Dataset #2: Combo with Perturbation_A/B and 2× noPerturbation
###########################################################
class ComboPerturbationDataset(Dataset):
    """
    For each sample with type config["combo_type"] (i.e. comboPerturbation) in the given indices,
    returns a 9-tuple:
      (X_combo, strength_combo_a, strength_combo_b,
       X_perturbation_a, strength_a,
       X_perturbation_b, strength_b,
       X_no1, X_no2)
    where:
      - X_combo is the feature vector of the combo sample.
      - For the combo sample, strength values are taken from config["strength_a_col"] and config["strength_b_col"].
      - X_perturbation_a is a randomly chosen singlePerturbation sample whose value in config["perturbation_a_col"]
        matches the combo sample's value; its strength is taken from config["strength_a_col"].
      - X_perturbation_b is similarly chosen using config["perturbation_b_col"] and its strength from config["strength_b_col"].
      - X_no1 and X_no2 are two randomly selected noPerturbation samples from the same condition (with no strength returned).
    """
    def __init__(self, adata, indices, config=DEFAULT_CONFIG, seed=42):
        super().__init__()
        self.adata = adata
        self.config = config
        self.condition_col = config["condition_col"]
        self.perturbation_a_col = config["perturbation_a_col"]
        self.perturbation_b_col = config["perturbation_b_col"]
        self.rng = np.random.RandomState(seed)

        type_col = config["type_col"]
        combo_type = config["combo_type"]

        # Keep only indices corresponding to comboPerturbation samples.
        self.combo_indices = [i for i in indices if adata.obs.iloc[i][type_col] == combo_type]

        # Build dictionaries for sampling singlePerturbation and noPerturbation.
        self.condition_perturbation2singles_a = {}
        self.condition_perturbation2singles_b = {}
        self.condition2noPerturbation = {}

        single_type = config["single_type"]
        no_type = config["no_type"]

        for i in indices:
            row = adata.obs.iloc[i]
            typ = row[type_col]
            cond = row[self.condition_col]

            if typ == single_type:
                key_a = (cond, row[self.perturbation_a_col])
                self.condition_perturbation2singles_a.setdefault(key_a, []).append(i)
                key_b = (cond, row[self.perturbation_b_col])
                self.condition_perturbation2singles_b.setdefault(key_b, []).append(i)
            elif typ == no_type:
                self.condition2noPerturbation.setdefault(cond, []).append(i)

        for k in self.condition_perturbation2singles_a:
            self.condition_perturbation2singles_a[k] = np.array(self.condition_perturbation2singles_a[k], dtype=int)
        for k in self.condition_perturbation2singles_b:
            self.condition_perturbation2singles_b[k] = np.array(self.condition_perturbation2singles_b[k], dtype=int)
        for k in self.condition2noPerturbation:
            self.condition2noPerturbation[k] = np.array(self.condition2noPerturbation[k], dtype=int)

    def __len__(self):
        return len(self.combo_indices)

    def __getitem__(self, idx):
        # For combo samples, use both strength_a_col and strength_b_col.
        strength_a_col = self.config["strength_a_col"]
        strength_b_col = self.config["strength_b_col"]

        combo_idx = self.combo_indices[idx]
        row_combo = self.adata.obs.iloc[combo_idx]
        cond = row_combo[self.condition_col]
        perturbation_a_val = row_combo[self.perturbation_a_col]
        perturbation_b_val = row_combo[self.perturbation_b_col]

        X_combo = self.adata.X[combo_idx]
        strength_combo_a = self.adata.obs.iloc[combo_idx][strength_a_col]
        strength_combo_b = self.adata.obs.iloc[combo_idx][strength_b_col]

        # Sample a singlePerturbation for perturbation_a.
        key_a = (cond, perturbation_a_val)
        if key_a not in self.condition_perturbation2singles_a or len(self.condition_perturbation2singles_a[key_a]) == 0:
            raise ValueError(f"No singlePerturbation sample for perturbation_a {perturbation_a_val} in condition {cond}")
        idx_a = self.rng.choice(self.condition_perturbation2singles_a[key_a])
        X_a = self.adata.X[idx_a]
        strength_a = self.adata.obs.iloc[idx_a][strength_a_col]

        # Sample a singlePerturbation for perturbation_b.
        key_b = (cond, perturbation_b_val)
        if key_b not in self.condition_perturbation2singles_b or len(self.condition_perturbation2singles_b[key_b]) == 0:
            raise ValueError(f"No singlePerturbation sample for perturbation_b {perturbation_b_val} in condition {cond}")
        idx_b = self.rng.choice(self.condition_perturbation2singles_b[key_b])
        X_b = self.adata.X[idx_b]
        strength_b = self.adata.obs.iloc[idx_b][strength_b_col]

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
                X_a, strength_a,
                X_b, strength_b,
                X_no1, X_no2)


###############################################
# Example usage:
###############################################
def example_usage(adata, config=DEFAULT_CONFIG):
    """
    Demonstrates how to split the data and create DataLoaders.
    
    1) Globally splits samples of type config["combo_type"] so that a combo (derived from 
       config["perturbation_a_col"] and config["perturbation_b_col"]) appears only in train, val, or test.
    2) Collects all samples with type config["single_type"] for training.
    3) Creates two datasets for training:
         - NoPerturbationSinglePerturbationDataset (for the first loss)
         - ComboDataset (for the second loss)
    4) Creates ComboDatasets for validation and testing.
    5) Returns PyTorch DataLoaders for each.
    """
    from torch.utils.data import DataLoader

    # Split comboPerturbation samples globally.
    train_combo_idxs, val_combo_idxs, test_combo_idxs = create_combo_split(adata, config=config)

    # Get all singlePerturbation indices (for training only).
    train_single_idxs = get_all_single_perturbation_idx(adata, config=config)

    # Build training datasets.
    train_ds_1 = SinglePerturbationDataset(adata, train_single_idxs, config=config)
    train_ds_2 = ComboPerturbationDataset(adata, train_combo_idxs, config=config)

    # Build validation and test datasets (only for combos).
    val_ds_2 = ComboPerturbationDataset(adata, val_combo_idxs, config=config)
    test_ds_2 = ComboPerturbationDataset(adata, test_combo_idxs, config=config)

    # Wrap in DataLoaders.
    train_loader_1 = DataLoader(train_ds_1, batch_size=64, shuffle=True)
    train_loader_2 = DataLoader(train_ds_2, batch_size=32, shuffle=True)
    val_loader_2 = DataLoader(val_ds_2, batch_size=32, shuffle=False)
    test_loader_2 = DataLoader(test_ds_2, batch_size=32, shuffle=False)

    return train_loader_1, train_loader_2, val_loader_2, test_loader_2