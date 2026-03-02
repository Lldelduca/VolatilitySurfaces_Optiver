import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skfda.preprocessing.dim_reduction import FPCA
from skfda.representation.grid import FDataGrid

import config.settings as g
from src.compression.data_preparation import get_clean_4d_tensor

def find_total_variance_rank(r2_base, local_residual_variances, target_total_r2=96.0, max_rank=4):
    """
    Selects the minimum rank required so that (Global Beta + Local PCs) 
    explains `target_total_r2`% of the ORIGINAL asset variance.
    """
    # The percentage of the total asset variance left over for the local model to explain
    remaining_total_variance = 100.0 - r2_base 
    current_total_r2 = r2_base
    
    for r, marginal_resid_r2 in enumerate(local_residual_variances):
        # Convert the PC's explanation of the *residual* into its explanation of the *total asset*
        marginal_total_contribution = marginal_resid_r2 * (remaining_total_variance / 100.0)
        current_total_r2 += marginal_total_contribution
        
        if current_total_r2 >= target_total_r2:
            return min(r + 1, max_rank)
            
    # If it never hits the target, return the cap to prevent noise-fitting
    return max_rank

if __name__ == "__main__":
    print("Loading Data for SKFDA Rank Exploration...")
    X_original = get_clean_4d_tensor()
    train_slice = slice(None, g.JAN_2025)
    
    N_OBS, N_ASSETS, N_MAT, N_MON = X_original.shape
    S_PTS = N_MAT * N_MON
    grid_points = np.arange(S_PTS)

    # 1. FANOVA Mean Subtraction (Train Only)
    grand_mean = X_original[train_slice].mean(axis=(0, 1))
    asset_bias = X_original[train_slice].mean(axis=0) - grand_mean
    market_effect = X_original.mean(axis=1) - grand_mean

    # ==========================================
    # STEP A: EXPLORE GLOBAL RANK
    # ==========================================
    print("\n--- Exploring Global Tensor Rank (SKFDA) ---")
    
    fd_global = FDataGrid(
        data_matrix=market_effect.reshape(N_OBS, S_PTS),
        grid_points=grid_points
    )
    
    max_test_rank = 10
    fpca_global_test = FPCA(n_components=max_test_rank)
    fpca_global_test.fit(fd_global[train_slice])
    
    global_r2_scores = fpca_global_test.explained_variance_ratio_.cumsum() * 100
    
    for r in range(max_test_rank):
        print(f"Global Rank {r+1}: Explains {global_r2_scores[r]:.2f}% of Market Variance")

    # Plot Global Scree
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, max_test_rank + 1), global_r2_scores, marker='o', linestyle='-', color='b')
    plt.title("Global Factor: PCA Cumulative Scree Plot")
    plt.xlabel("Number of Principal Components (Rank)")
    plt.ylabel("Cumulative Explained Variance (%)")
    plt.grid(True)
    plt.show()

    CHOSEN_GLOBAL_RANK = int(input("Enter chosen Global Rank based on the plot (e.g., 3): "))

    # ==========================================
    # STEP B: EXPLORE LOCAL RANKS (Toni's Total Variance Method)
    # ==========================================
    TARGET_R2 = 95.0
    MAX_LOCAL_RANK = 5
    
    print(f"\n--- Exploring Local Ranks (Target: {TARGET_R2}% Total Variance, Cap: {MAX_LOCAL_RANK}) ---")
    
    # Re-fit Global with chosen rank to get scores for the regression
    fpca_global = FPCA(n_components=CHOSEN_GLOBAL_RANK)
    fpca_global.fit(fd_global[train_slice])
    global_scores = fpca_global.transform(fd_global)
    
    X_reg = global_scores[:, :CHOSEN_GLOBAL_RANK]
    X_reg_train = X_reg[train_slice]

    optimal_local_ranks = []

    for idx_asset, symbol in enumerate(g.SYMBOLS):
        # Calculate Y_j (Centered Asset)
        Y_j = (X_original[:, idx_asset, :, :].reshape(N_OBS, S_PTS) 
               - (grand_mean + asset_bias[idx_asset]).reshape(1, S_PTS))
        Y_j_train = Y_j[train_slice]

        # Fit OLS Beta strictly on training data
        B_j, _, _, _ = np.linalg.lstsq(X_reg_train, Y_j_train, rcond=None)
        
        # 1. Get the Base R2 from the OLS fit (How much did the Beta explain?)
        Y_j_train_dm = Y_j_train - Y_j_train.mean(axis=0)
        ss_total = np.sum(Y_j_train_dm ** 2)
        resid_train = Y_j_train - (X_reg_train @ B_j)
        ss_resid = np.sum(resid_train ** 2)
        
        r2_base = (1.0 - (ss_resid / ss_total)) * 100
        
        # 2. Local FPCA on the training residuals
        fd_local = FDataGrid(
            data_matrix=resid_train,
            grid_points=grid_points
        )
        
        fpca_local_test = FPCA(n_components=max_test_rank)
        fpca_local_test.fit(fd_local[train_slice])
        
        # 3. Get the individual variance explained by each PC
        local_marginal_variances = fpca_local_test.explained_variance_ratio_ * 100
        
        # 4. Apply Toni's Total Variance logic
        chosen_rank = find_total_variance_rank(
            r2_base=r2_base, 
            local_residual_variances=local_marginal_variances, 
            target_total_r2=TARGET_R2,
            max_rank=MAX_LOCAL_RANK
        )
        optimal_local_ranks.append(int(chosen_rank))
        
        print(f"{symbol:>5}: Beta Explains {r2_base:>5.1f}% | Total hit {TARGET_R2}% at Rank {chosen_rank}")
    
    # Economic interpretability. Force a minimum of 3 (Level, Skew, Curvature)
    optimal_local_ranks = [max(3, rank) for rank in optimal_local_ranks]

    print("\n==================================================")
    print("PHASE 1 COMPLETE. UPDATE YOUR config/settings.py WITH:")
    print(f"N_PC_GLOBAL = {CHOSEN_GLOBAL_RANK}")
    print(f"N_PC_LOCAL = {optimal_local_ranks}")
    print("==================================================")