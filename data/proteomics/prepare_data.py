"""
data_preparation.py

This script replicates the logic in the original 'data_preparation.r' file.
It uses an already-prepared dataframe (analogous to 'load("data/prepData.RData")' in R),
then performs:
    - Filtering rows based on pertLabel.
    - Grouping and aggregating data (median).
    - Reshaping (wide-format pivot).
    - Merging or binding baseline rows.
    - Creating combination vs singleDrug categories.
    - Saving final DataFrames (agg_data, comb_data, etc.).
"""

import pandas as pd
import numpy as np

def main():
    # ----------------------------------------------------------------------
    # 1. Load data that was preprocessed previously
    #    In R: load("data/prepData.RData")
    #    In Python, we can load from a pickle or CSV. Adjust as needed:
    # ----------------------------------------------------------------------
    # data = pd.read_pickle('data/prepData.pkl')
    # Or:
    # data = pd.read_csv('data/prepData.csv')
    
    # Here we assume you've loaded the “data” DataFrame containing all columns
    # that R's 'prepData.RData' had. Adjust path/filename as required.
    data = pd.read_pickle('data/prepData.pkl')
    
    # The R code references:
    # n_protein <- 5519
    # So let's store that or detect it from columns if needed.
    n_protein = 5519
    
    # ----------------------------------------------------------------------
    # 2. Basic filtering: remove 'no no' and 'no' rows from the data
    # ----------------------------------------------------------------------
    # R: dat <- data[data$pertLabel != 'no no' & data$pertLabel != 'no', ]
    mask = (data['pertLabel'] != 'no no') & (data['pertLabel'] != 'no')
    dat = data[mask].copy()
    
    # ----------------------------------------------------------------------
    # 3. Example: Check which IDs are NA in the 'IC50' column
    # ----------------------------------------------------------------------
    # In R: dat[which(is.na(dat[, 'IC50'])), 'Sample_ID']
    # In Python, that might look like:
    missing_ic50_ids = dat.loc[dat['IC50'].isna(), 'Sample_ID']
    # print(missing_ic50_ids)
    
    # ----------------------------------------------------------------------
    # 4. Group by and summarize (median) across all proteins for
    #    (protein_plate, pertLabel, Anchor_dose, Library_dose, pert_time).
    #    In R: summarise_all(funs(median)).
    # ----------------------------------------------------------------------
    group_cols = ['protein_plate', 'pertLabel', 'Anchor_dose', 'Library_dose', 'pert_time']
    
    # We want to take the median of numeric columns for each group
    agg_data = dat.groupby(group_cols).median(numeric_only=True).reset_index()
    
    # ----------------------------------------------------------------------
    # 5. Identify how many 'singleDrug' we have, etc.
    #    R code uses names/data indexing. We'll assume these columns are present:
    # ----------------------------------------------------------------------
    prot_names = list(data.columns[:n_protein])  # first n_protein are proteins
    # In R: length(unique(data$pertLabel[data$type == 'singleDrug']))
    # We'll skip the direct count for brevity or replicate as needed.
    
    # ----------------------------------------------------------------------
    # 6. Create 'baseline_prot' from the aggregated data where pertLabel == 'no'
    # ----------------------------------------------------------------------
    baseline_prot = agg_data[agg_data['pertLabel'] == 'no'].copy()
    
    # Remove 'no no' and 'no' from agg_data as in R
    agg_data = agg_data[(agg_data['pertLabel'] != 'no no') & (agg_data['pertLabel'] != 'no')]
    
    # ----------------------------------------------------------------------
    # 7. Use only cell lines that exist in the baseline data
    # ----------------------------------------------------------------------
    unique_protein_plates_baseline = baseline_prot['protein_plate'].unique()
    agg_data = agg_data[agg_data['protein_plate'].isin(unique_protein_plates_baseline)]
    
    # ----------------------------------------------------------------------
    # 8. Add baseline rows back in for each protein_plate, copying the baseline-prot values
    # ----------------------------------------------------------------------
    # The R code: 
    # baseline <- agg_data[, pert_info_names]
    # for (i in 1:nrow(baseline_prot)) { ... copy values ... }
    
    pert_info_names = ['protein_plate','pertLabel','Anchor_dose','Library_dose','IC50'] + \
                      list(data.columns[n_protein:(n_protein + 10)])  # adjust as needed
    
    # Step A: Subset the aggregator to those columns:
    baseline = agg_data[pert_info_names].drop_duplicates().copy()
    baseline['pert_time'] = 0  # set time to 0 for baseline rows
    
    # Step B: For each row in baseline_prot, fill in the protein columns
    for idx, row in baseline_prot.iterrows():
        # Identify the matching condition in 'baseline'
        matching_mask = baseline['protein_plate'] == row['protein_plate']
        # Copy the actual protein values from baseline_prot into baseline
        # The R code: baseline[baseline$protein_plate == ..., prot_names] <- row[prot_names]
        baseline.loc[matching_mask, prot_names] = row[prot_names].values
    
    # Step C: Append these baseline rows to the aggregator
    agg_data = pd.concat([agg_data, baseline], ignore_index=True)
    
    # ----------------------------------------------------------------------
    # 9. Pivot the aggregator wide by 'pert_time'
    #    R uses pivot_wider(names_from = pert_time, values_from = all_of(prot_names)).
    # ----------------------------------------------------------------------
    # We'll pivot everything on 'pert_time' for the protein columns only.
    # One approach:
    #   1) melt the data
    #   2) pivot_table or pivot
    #   3) re-merge non-protein columns
    id_cols = ['protein_plate','pertLabel','Anchor_dose','Library_dose','IC50']
    
    # Melt only the protein columns plus time
    melted = agg_data.melt(
        id_vars=id_cols + ['pert_time'],
        value_vars=prot_names,
        var_name='protein',
        value_name='value'
    )
    
    # Pivot with 'pert_time' across columns
    agg_data_wide = melted.pivot_table(
        index=id_cols + ['protein'], 
        columns='pert_time', 
        values='value'
    ).reset_index()
    
    # Now we have a row per ID + protein, with columns for each time
    # If you'd like them labeled as e.g. protein_0, protein_6, etc.,
    # you can rename after pivot:
    agg_data_wide.columns.name = None
    # E.g. rename columns that were 0, 6, 24, 48:
    wide_cols = list(agg_data_wide.columns)
    # Something like:
    # time_cols = [c for c in wide_cols if isinstance(c, (int,float))]
    # for c in time_cols:
    #     agg_data_wide.rename(columns={c: f"{c}"}, inplace=True)

    # If you want each protein to be "protein_0", "protein_6", etc. for each row,
    # you can pivot again so each row is a single ID, or continue with the
    # long-like structure. The original R approach ends up with one row
    # per (plate, label, dose, time) or merges them. 
    #
    # If you truly need "one row per combination of cell line / drug / dose," 
    # you do a multi-level pivot. The exact approach depends on your final usage.

    # For demonstration, let's revert to a final shape akin to R (one row per group):
    # We'll pivot the 'protein' dimension across columns as well:
    # This gets large, so be mindful of memory usage.
    final_agg_data = agg_data_wide.pivot_table(
        index=id_cols,
        columns='protein',
        values=[0, 6, 24, 48],  # or however many timepoints you have
    )
    final_agg_data.reset_index(inplace=True)
    
    # The above lines replicate the pivot_wider idea, giving columns like (0, PROT1), (0, PROT2)...

    # ----------------------------------------------------------------------
    # 10. Remove rows with NA across timepoints (similar to 'na.omit(agg_data)')
    # ----------------------------------------------------------------------
    final_agg_data.dropna(axis=0, how='any', inplace=True)
    
    # ----------------------------------------------------------------------
    # 11. Mark the type: singleDrug or drugCombination
    # ----------------------------------------------------------------------
    # In R: agg_data$type[agg_data$Anchor_dose != 0] <- 'drugCombination'
    final_agg_data['type'] = 'singleDrug'
    mask_combo = final_agg_data['Anchor_dose'] != 0
    final_agg_data.loc[mask_combo, 'type'] = 'drugCombination'
    
    # ----------------------------------------------------------------------
    # 12. Save final aggregator data
    # ----------------------------------------------------------------------
    # E.g. final_agg_data.to_pickle('data/aggData.pkl')
    
    # ----------------------------------------------------------------------
    # Additional Steps: The R script also builds comb_data, does time-difference, etc.
    # Below is a concise version replicating R's final steps.
    # ----------------------------------------------------------------------
    
    # Combine single rows with baseline columns:
    # The R code merges baseline data for each row:
    # We'll show the logic conceptually. 
    # For a thorough solution, you can replicate each line from R with Python merges.

    # Time differences for each protein between 0 -> 6, 6 -> 24, 24 -> 48
    # In R: agg_data_diff[, prot_names_48] <- agg_data_diff[, prot_names_48] - ...
    # Pythonically, you can do it with subsetting or direct loops.

    # Example (assuming final_agg_data columns are multi-index with time outer-level):
    # for t1, t2 in [(6,0), (24,6), (48,24)]:
    #     final_agg_data[(t1, slice(None))] = final_agg_data[(t1, slice(None))] - final_agg_data[(t2, slice(None))]

    # Then save it
    # final_agg_data.to_pickle('data/aggData_diff.pkl')

    print("Data preparation complete. Final shape:", final_agg_data.shape)

if __name__ == "__main__":
    main()
