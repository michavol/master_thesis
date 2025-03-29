"""
reformat_raw_data.py

It reads in raw data and presents it in a usable format.
"""

import pandas as pd
import numpy as np

def main():
    # ----------------------------------------------------------------------
    # 1. Read data
    # ----------------------------------------------------------------------
    # Adjust the path/filename as needed:
    dat = pd.read_csv('data/ProteinMatrix_sampleID_MapEC50_20240229.csv', 
                      index_col="Sample_ID",
                      low_memory=False)
    
    # ----------------------------------------------------------------------
    # 2. Get protein Data
    # ----------------------------------------------------------------------
    p_idx_end = 5585
    data_protein = dat.iloc[:, :p_idx_end].copy()
    
    # ----------------------------------------------------------------------
    # 3. Remove columns with only one unique value
    # ----------------------------------------------------------------------
    unique_counts = data_protein.nunique(dropna=False)
    cols_to_remove = unique_counts[unique_counts == 1].index
    data_protein.drop(columns=cols_to_remove, inplace=True)
    
    # Keep only columns with "HUMAN" in their name (as in R code)
    data_protein = data_protein.loc[:, data_protein.columns.str.contains("HUMAN")]
    

    # ----------------------------------------------------------------------
    # 4. Identify combination rows
    # ----------------------------------------------------------------------
    combination_idx = dat.index[dat['pert_id'].isna()].tolist()  # all entries with nans correspond to drug combinations

    
    # ----------------------------------------------------------------------
    # 5. Additional metadata columns
    # ----------------------------------------------------------------------
    data_additional = dat[['pert_time', 'protein_plate', 'machine', 'BioRep',
                           'Anchor_dose', 'Library_dose']].copy()
    data_additional['Anchor_dose'] = data_additional['Anchor_dose'].fillna(0)
    data_additional['Library_dose'] = data_additional['Library_dose'].fillna(0)
    
    # ----------------------------------------------------------------------
    # 6. Response data: singleDrug uses 'EC50', combos use 'Combo.IC50'
    # ----------------------------------------------------------------------
    data_response = dat['EC50'].copy()
    combo_ic50 = dat['Combo.IC50'].copy()
    data_response.loc[combination_idx] = combo_ic50.loc[combination_idx]
    
    # ----------------------------------------------------------------------
    # 7. Type and pertLabel columns
    # ----------------------------------------------------------------------
    # Mark as 'drugCombination' for combos, 'singleDrug' otherwise, 'noDrug' if 'no'
    type_col = pd.Series(['singleDrug'] * len(dat), index=dat.index)
    type_col.loc[combination_idx] = 'drugCombination'
    
    # Adding the combination controls is new from the original R code
    drugs = dat['pert_id'].astype(str)
    combination_control_mask = dat["drugIdAB"] == 'no no'
    type_col.loc[(drugs == 'no') | combination_control_mask] = 'noDrug' 
    
    pertLabel = dat['pert_id'].astype(str).copy()
    # For combos, set label to drugIdAB
    pertLabel.loc[combination_idx] = dat.loc[combination_idx, 'drugIdAB'].astype(str)
    
    # ----------------------------------------------------------------------
    # 8. Combine everything
    # ----------------------------------------------------------------------
    data_final = pd.concat([data_protein,
                            data_additional], 
                            axis=1)
    
    data_final['type'] = type_col.values
    data_final['pertLabel'] = pertLabel.values
    data_final['IC50'] = data_response.values
    data_final['NY'] = dat['NY'].values  
    
    # ----------------------------------------------------------------------
    # 9. Log2 transform of singleDrug IC50 only #TODO: Not entierely sure why
    # ----------------------------------------------------------------------
    single_drug_mask = (data_final['type'] == 'singleDrug') & data_final['IC50'].notna()
    data_final.loc[single_drug_mask, 'IC50'] = np.log2(data_final.loc[single_drug_mask, 'IC50'])
    
    # ---------------------------------------------------------------------
    # 10. Create column with dosages as tuples.
    # ---------------------------------------------------------------------
    data_final.loc[data_final['type'] == 'singleDrug', 'Anchor_dose'] = 10 # Single drug dose is 10uM
    data_final.loc[data_final['type'] == 'noDrug', 'Anchor_dose'] = 0
    data_final.loc[data_final['type'] == 'singleDrug', 'Library_dose'] = 0 # Single drug dose is 10uM
    data_final.loc[data_final['type'] == 'noDrug', 'Library_dose'] = 0

    # Rename anchor dose to "anchorDose" and library dose to "libraryDose"
    data_final.rename(columns={"Anchor_dose": "anchor_dose", "Library_dose": "library_dose", "BioRep": "bio_rep"}, inplace=True)

    # ---------------------------------------------------------------------
    # 11. Change Pert label column to tuples.
    def parse_pert_label(label):
        parts = label.split()
        # If there is only one part, we’ll fill the second with None
        if len(parts) == 1:
            anchor = parts[0]
            library = "no"
        # If there are at least two parts:
        elif len(parts) >= 2:
            anchor = parts[0]
            library = parts[1]
        else:
            # If somehow empty, just return Nones
            anchor, library = None, None
    
        return anchor, library

    # Apply the parsing function and unpack into two new columns
    data_final["anchor_drug"], data_final["library_drug"] = zip(*data_final["pertLabel"].copy().apply(parse_pert_label))

    # Drop the original pertLabel column
    data_final.drop(columns=["pertLabel"], inplace=True)

    # ----------------------------------------------------------------------
    # 12. Save final data
    # ----------------------------------------------------------------------
    data_final.to_csv('data/raw_reformatted.csv')

    
    print("Data reformatting complete. Example final shape:", data_final.shape)

if __name__ == "__main__":
    main()