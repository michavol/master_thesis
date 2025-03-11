"""
data_preprocessing.py

This script replicates the logic in the original 'data_preprocessing.r' file.
It reads in raw data, does the following:
    - Selects protein columns and removes columns with only one value.
    - Imputes missing values using a custom rule (replace with 0.8 * min of column).
    - Transforms values by taking the log.
    - One-hot encodes drug identifiers (similar to dummy_cols in R).
    - Identifies combination drugs and sets their values according to 'Anchor_dose' and 'Library_dose'.
    - Combines everything into a final DataFrame with metadata and an IC50 response column.
    - (Optional) Saves to pickle/CSV for further use.
"""

import pandas as pd
import numpy as np

def na_imputation(series: pd.Series) -> pd.Series:
    """
    Replace NA values with 0.8 * min of the series (excluding NA).
    Equivalent to the na_imputation() function in R.
    """
    min_val = series.min(skipna=True)
    series_filled = series.fillna(min_val * 0.8)
    return series_filled

def main():
    # ----------------------------------------------------------------------
    # 1. Read data
    # ----------------------------------------------------------------------
    # Adjust the path/filename as needed:
    dat = pd.read_csv('data/ProteinMatrix_sampleID_MapEC50_20240229.csv')
    print(dat.head(5))
    quit()
    
    # ----------------------------------------------------------------------
    # 2. Identify column indices for protein data
    #    In the R script, p_idx was 2:5586, which implies columns
    #    2 through 5586 in 1-based indexing. In Python (0-based),
    #    that typically means columns 1 through 5585, but adjust
    #    based on your actual data structure.
    # ----------------------------------------------------------------------
    # Here we assume the first protein column is index 1 and the last is index 5585.
    # If your file's structure differs, update as needed.
    p_idx_start = 1
    p_idx_end = 5585
    
    data_protein = dat.iloc[:, p_idx_start:(p_idx_end + 1)].copy()
    
    # ----------------------------------------------------------------------
    # 3. Remove columns with only one unique value
    # ----------------------------------------------------------------------
    unique_counts = data_protein.nunique(dropna=False)
    cols_to_remove = unique_counts[unique_counts == 1].index
    data_protein.drop(columns=cols_to_remove, inplace=True)
    
    # Keep only columns with "HUMAN" in their name (as in R code)
    data_protein = data_protein.loc[:, data_protein.columns.str.contains("HUMAN")]
    
    # ----------------------------------------------------------------------
    # 4. Impute missing values
    # ----------------------------------------------------------------------
    data_protein = data_protein.apply(na_imputation)
    
    # ----------------------------------------------------------------------
    # 5. Log transform
    # ----------------------------------------------------------------------
    data_protein = np.log(data_protein)
    
    # ----------------------------------------------------------------------
    # 6. Prepare "drug" data, akin to dummy_cols(data.frame(drug = drugs))
    # ----------------------------------------------------------------------
    drugs = dat['pert_id'].astype(str)
    # Convert to one-hot (pandas.get_dummies is similar to fastDummies in R)
    data_drugs = pd.get_dummies(drugs, prefix='drug', dtype=float)
    
    # We note in R code: 'drug_no' and 'drug_' columns are removed or used specially.
    # We'll remove them (if present) to replicate the "no drug" reference logic.
    for col in ['drug_no', 'drug_']:
        if col in data_drugs.columns:
            data_drugs.drop(columns=col, inplace=True)
    
    # ----------------------------------------------------------------------
    # 7. Identify combination rows (where drug_ was blank in R)
    #    The R code uses combination_idx <- which(data_drugs[, "drug_"] == 1)
    #    In the Python approach, we mimic that by checking drug_ col, 
    #    but we already dropped 'drug_', so we rely on separate logic:
    # 
    #    The R code eventually uses 'drugIdAB' for combos. We'll detect combos
    #    by checking rows where 'drugIdAB' is not null if that’s the marker
    #    or by some other logic. For clarity, we copy the same approach used in R:
    # ----------------------------------------------------------------------
    combination_idx = dat.index[dat['pert_id'] == ''].tolist()  # If that matches the R meaning
    # Alternatively, if R is using 'drugIdAB' as the indicator:
    # combination_idx = dat.index[dat['drugIdAB'].notna() & (dat['drugIdAB'] != '')]
    
    # ----------------------------------------------------------------------
    # 8. Adjust the one-hot encoding for combination doses
    #    The original R code sets data_drugs[i, drugAB[1]] = Anchor_dose
    #    and data_drugs[i, drugAB[2]] = Library_dose for combos.
    # ----------------------------------------------------------------------
    for i in combination_idx:
        # 'drugIdAB' holds something like 'A B' for the two drugs
        drugAB = str(dat.loc[i, 'drugIdAB']).split()
        # Prepend 'drug_' to match the column naming
        drugAB_cols = [f"drug_{d}" for d in drugAB]
        
        anchor_dose = dat.loc[i, 'Anchor_dose']
        library_dose = dat.loc[i, 'Library_dose']
        
        # If the columns exist, set them accordingly
        if drugAB_cols[0] in data_drugs.columns:
            data_drugs.at[i, drugAB_cols[0]] = anchor_dose
        if drugAB_cols[1] in data_drugs.columns:
            data_drugs.at[i, drugAB_cols[1]] = library_dose
    
    # Multiply the entire drug matrix by 10, as in R
    data_drugs *= 10
    
    # ----------------------------------------------------------------------
    # 9. Additional metadata columns
    # ----------------------------------------------------------------------
    data_additional = dat[['pert_time', 'protein_plate', 'machine', 'BioRep', 'Sample_ID',
                           'Anchor_dose', 'Library_dose']].copy()
    data_additional['Anchor_dose'] = data_additional['Anchor_dose'].fillna(0)
    data_additional['Library_dose'] = data_additional['Library_dose'].fillna(0)
    
    # ----------------------------------------------------------------------
    # 10. Response data: singleDrug uses 'EC50', combos use 'Combo.IC50'
    # ----------------------------------------------------------------------
    data_response = dat['EC50'].copy()
    combo_ic50 = dat['Combo.IC50'].copy()
    data_response.loc[combination_idx] = combo_ic50.loc[combination_idx]
    
    # ----------------------------------------------------------------------
    # 11. Type and pertLabel columns
    # ----------------------------------------------------------------------
    # Mark as 'drugCombination' for combos, 'singleDrug' otherwise, 'noDrug' if 'no'
    type_col = pd.Series(['singleDrug'] * len(dat), index=dat.index)
    type_col.loc[combination_idx] = 'drugCombination'
    type_col.loc[drugs == 'no'] = 'noDrug'
    
    pertLabel = dat['pert_id'].astype(str).copy()
    # For combos, set label to drugIdAB
    pertLabel.loc[combination_idx] = dat.loc[combination_idx, 'drugIdAB'].astype(str)
    
    # If 'no no' or 'no' means no drug, we can handle that similarly:
    # data_response[pertLabel == 'no'] = np.inf  # e.g. replicate R's assignment of Inf
    
    # ----------------------------------------------------------------------
    # 12. Combine everything
    # ----------------------------------------------------------------------
    data_final = pd.concat([data_protein.reset_index(drop=True),
                            data_drugs.reset_index(drop=True),
                            data_additional.reset_index(drop=True)], axis=1)
    data_final['type'] = type_col.values
    data_final['pertLabel'] = pertLabel.values
    data_final['IC50'] = data_response.values
    data_final['NY'] = dat['NY'].values  # If that column exists
    
    # ----------------------------------------------------------------------
    # 13. Log2 transform of singleDrug IC50 only
    # ----------------------------------------------------------------------
    single_drug_mask = (data_final['type'] == 'singleDrug') & data_final['IC50'].notna()
    data_final.loc[single_drug_mask, 'IC50'] = np.log2(data_final.loc[single_drug_mask, 'IC50'])
    
    # ----------------------------------------------------------------------
    # 14. Save final data
    # ----------------------------------------------------------------------
    # In R, we used 'save(data, file="data/prepData.RData")'.
    # For Python, you might do:
    # data_final.to_pickle('data/prepData.pkl')
    # Or to CSV:
    # data_final.to_csv('data/prepData.csv', index=False)
    
    print("Data preprocessing complete. Example final shape:", data_final.shape)

if __name__ == "__main__":
    main()