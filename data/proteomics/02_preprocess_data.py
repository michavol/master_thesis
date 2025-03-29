"""
preprocess_data.py

#TODO: Currently only creates small debugging dataset with two cell lines.

Uses reformated data and implements a series of preprocessing steps and saves a AnnData object.
- Imputation
- Log transform
- z-score normalization
- Batch correction via harmonypy
"""

import pandas as pd
import anndata
import numpy as np
import harmonypy as hm
import ast
import scanpy as sc
pd.options.mode.copy_on_write = True

def main():
    # ----------------------------------------------------------------------
    # 1. Read data
    # ----------------------------------------------------------------------
    # Adjust the path/filename as needed:
    dat = pd.read_csv('data/raw_reformatted.csv', index_col=0)

    # Make small dataset for testing
    dat = dat.loc[(dat['pert_time'] == 24) | (dat['pert_time'] == 0)]
    dat = dat.loc[(dat['protein_plate'] == 'MCF7') | (dat['protein_plate'] == 'T47D')]

    # Separate meta and protein data
    dat_proteins = dat.loc[:, dat.columns.str.contains('HUMAN')]
    dat_meta = dat.loc[:, ~dat.columns.str.contains('HUMAN')]

    # Filter pert_time = 24h
    dat_meta.drop(columns=["pert_time"], inplace=True)
    
    # ----------------------------------------------------------------------
    # 2. Create AnnData object
    # ----------------------------------------------------------------------
    adata = anndata.AnnData(X=dat_proteins.values, obs=dat_meta)
    
    # ----------------------------------------------------------------------
    # 3. Filter proteins and samples
    # ----------------------------------------------------------------------
    # Before 730 × 5519

    # Example filtering: remove cells with low gene/protein counts and features detected in very few cells
    sc.pp.filter_cells(adata, min_genes=100)   # adjust threshold as appropriate for your data
    sc.pp.filter_genes(adata, min_cells=5)       # adjust threshold as appropriate

    # After 715 × 4611

    

    # ----------------------------------------------------------------------
    # 4. Log transformation
    # ----------------------------------------------------------------------
    #sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # ----------------------------------------------------------------------
    # 5. Data imputation
    # ----------------------------------------------------------------------
    # Iterate over each protein (column)
    # Ensure data is in dense format if needed.
    if hasattr(adata.X, "toarray"):
        data = adata.X.toarray().copy()
    else:
        data = adata.X.copy()

    # Create a copy for the imputed data.
    data_imputed = data.copy()

    for i in range(data.shape[1]):
        col = data[:, i]
        missing = np.isnan(col)
        observed = col[~missing]
        
        # Only perform imputation if there are observed values.
        if len(observed) > 0:
            mean_obs = np.mean(observed)
            # get 1st percentile
            perc1_obs = np.percentile(observed, 1)
            std_obs = np.std(observed)
            
            # Define per-protein imputation parameters:
            impute_mean = perc1_obs - 0.5 * std_obs
            impute_std = 0.1 * std_obs
            
            # Generate imputed values for missing entries.
            imputed_values = np.random.normal(impute_mean, impute_std, size=missing.sum())
            data_imputed[missing, i] = imputed_values
        else:
            # Optionally, handle columns where all values are missing.
            # For now, we'll leave them as NaN.
            data_imputed[:, i] = np.nan

    # Update data
    adata.X = data_imputed

    # ----------------------------------------------------------------------
    # 6. Apply harmonpy batch correction with respect to "machine"
    # ----------------------------------------------------------------------
    # Here, include all metadata for inspection, but note that we only use "machine" for correction.
    meta_data = adata.obs[["machine", "anchor_drug", "library_drug", "protein_plate"]]

    # Run Harmony with "machine" as the batch key
    ho = hm.run_harmony(adata.X, meta_data, vars_use=["machine"], max_iter_harmony=20)

    # The corrected embeddings are stored in ho.Z_corr (each column corresponds to a cell)
    adata.X = ho.Z_corr.T
    
    # ----------------------------------------------------------------------
    # 7. Scale data
    # ----------------------------------------------------------------------
    sc.pp.scale(adata)

    # ----------------------------------------------------------------------
    # 8. Save data
    # ----------------------------------------------------------------------
    adata.write("data/preprocessed_small.h5ad")
    
    print("Data preprocessing complete. Example final shape:", adata.shape)

if __name__ == "__main__":
    main()