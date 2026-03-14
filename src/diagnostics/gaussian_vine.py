import pyvinecopulib as pv
import pandas as pd
import numpy as np
import os
import config.settings as g
import random
import json

# =============================================================================
# --- Gaussian Model Fitting ---
# =============================================================================

def fit_static_gaussian_vine(u_data, optimal_trunc_lvl):
    T, N = u_data.shape
    # We restrict to Gaussian only to isolate the Tail Contagion Premium
    controls = pv.FitControlsVinecop(
        family_set=[pv.BicopFamily.gaussian],
        selection_criterion="aic",
        tree_criterion="tau",        
        allow_rotations=True,        
        num_threads=os.cpu_count()-1,
        threshold=0.0, # No independence test; force Gaussian structure
        trunc_lvl=optimal_trunc_lvl
    )
    model = pv.Vinecop(d=N)
    model.select(u_data, controls=controls)
    return model

# =============================================================================
# --- Main Execution Block ---
# =============================================================================

if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

    res_dir = os.path.join(project_root, "results", "dynamics")
    # Save to a 'gaussian' subfolder so it doesn't overwrite your Mixed M0
    out_dir = os.path.join(project_root, "results", "copulas", "gaussian")
    os.makedirs(out_dir, exist_ok=True)
    
    # --- LOAD FULL TRAIN DATA ---
    u_spot_file = os.path.join(res_dir, "NGARCH", "uniforms_ngarch_train.csv")
    u_har_file = os.path.join(res_dir, "HAR_GARCH", "uniforms_har_garch_evt_train.csv")
    u_nsde_file = os.path.join(res_dir, "NSDE", "uniforms_nsde_train.csv")

    u_spot = pd.read_csv(u_spot_file, index_col='Date', parse_dates=True)
    u_har = pd.read_csv(u_har_file, index_col='Date', parse_dates=True)
    u_nsde = pd.read_csv(u_nsde_file, index_col='Date', parse_dates=True)

    # Date Synchronization
    global_valid_dates = u_spot.index.intersection(u_har.index).intersection(u_nsde.index)
    u_spot = u_spot.loc[global_valid_dates]
    u_har = u_har.loc[global_valid_dates]
    u_nsde = u_nsde.loc[global_valid_dates]

    factor_sets = {"HAR-GARCH-EVT": u_har, "NSDE": u_nsde}

    for factor_name, u_factors in factor_sets.items():
        print(f"\nFitting Static Gaussian Baseline: Spot + {factor_name}")

        combined_u_train = pd.concat([u_spot, u_factors], axis=1)
        np_data_train = combined_u_train.to_numpy()
        
        # Use same K as your Mixed Vine to ensure scientific comparability
        if factor_name == "HAR-GARCH-EVT":
            opt_level = g.K_HAR_GARCH
        else:
            opt_level = g.K_NSDE

        print(f"Truncating at Tree {opt_level} to match Mixed Vine topology...")
        gaussian_model = fit_static_gaussian_vine(np_data_train, optimal_trunc_lvl=opt_level)

        # Output Summary
        print(f"  Log-Likelihood: {gaussian_model.loglik(np_data_train):.2f}")

        # Save JSON for Melvin's backtester
        save_name = f"gaussian_vine_spot_{factor_name.lower().replace('-', '_')}_model.json"
        json_path = os.path.join(out_dir, save_name)
        with open(json_path, "w") as f:
            f.write(gaussian_model.to_json())
            
    print("\nGaussian Baselines Generated successfully.")