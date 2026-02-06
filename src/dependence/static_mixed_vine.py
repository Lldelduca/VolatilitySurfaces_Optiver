import pyvinecopulib as pv

# Fits a Static Mixed R-Vine Copula to the data
def fit_static_mixed_vine(u_data):
    T, N = u_data.shape
    
    # Define the Fit Controls
    controls = pv.FitControlsVinecop(
        family_set=[
            pv.BicopFamily.indep,    # Sparsity (No dependence)
            pv.BicopFamily.gaussian, # Standard Correlation
            pv.BicopFamily.student,  # Symmetric Tail Dependence
            pv.BicopFamily.frank,    # Symmetric Body Dependence (No tails)
            pv.BicopFamily.clayton,  # Lower Tail (Crashes)
            pv.BicopFamily.gumbel,   # Upper Tail (Rallies)
        ],
        selection_criterion="bic",
        trunc_lvl=N-1,               # Fit full tree
        tree_criterion="tau",        # Use correlation to order the tree 
        allow_rotations=True         # Enables negative correlations
    )

    model = pv.Vinecop(d=N)

    # Structure Selection & Parameter Fitting (The function finds the best tree structure, families, and parameters)
    model.select(u_data, controls=controls)
    
    return model
