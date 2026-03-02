import numpy as np
import pandas as pd
import skfda
from skfda.preprocessing.dim_reduction import FPCA
from skfda.representation.grid import FDataGrid
import pickle as pkl

# --- ADD THIS TO SUPPRESS PLOT POP-UPS ---
import matplotlib
matplotlib.use('Agg') 
# -----------------------------------------

import config.settings as g
from src.compression.data_preparation import get_clean_4d_tensor
from src.compression.visualisation_function import plot_surfaces_for_latex

if __name__ == "__main__":
    # --- Configuration ---
    n_pc_global  = g.N_PC_GLOBAL
    n_pc_local   = g.N_PC_LOCAL
    plot_surfaces = True
    train_slice  = slice(None, g.JAN_2025)

    X_original = get_clean_4d_tensor()
    N_OBS, N_ASSETS, N_MAT, N_MON = X_original.shape
    
    S_PTS       = N_MAT * N_MON
    grid_points = np.arange(S_PTS)
    dates       = pd.to_datetime(g.DATES)

    # ── helpers ──────────────────────────────────────────────────────────
    WIDTH = 76
    SEP2 = "═" * WIDTH
    SEP1 = "─" * WIDTH

    def print_header(title):
        print(f"\n{SEP2}\n {title.upper().center(WIDTH-2)}\n{SEP2}")

    def print_section(title):
        print(f"\n{SEP1}\n {title}\n{SEP1}")

    def print_asset_local_table(symbol, r2_grand, r2_asset, r2_global, r2_local, pc_ratios, r2_ks):
        # Dynamically calculate the top border length based on symbol length
        top_line = f"─ {symbol} "
        pad_len = 76 - len(top_line) - 2
        
        print(f"\n┌{top_line}{'─' * pad_len}┐")
        print(f"│ {'Metric':<32} │ {'R²':>8} │ {'Δ R²':>10} │ {'% of Residual':>13} │")
        print(f"├{'─'*34}┼{'─'*10}┼{'─'*12}┼{'─'*15}┤")
        
        print(f"│ {'Grand Mean only':<32} │ {r2_grand:>8.4f} │ {'—':>10} │ {'—':>13} │")
        print(f"│ {'+ Asset Bias':<32} │ {r2_asset:>8.4f} │ {r2_asset - r2_grand:>10.4f} │ {'—':>13} │")
        print(f"│ {'+ Global (surface β, no int.)':<32} │ {r2_global:>8.4f} │ {r2_global - r2_asset:>10.4f} │ {'—':>13} │")

        prev_r2 = r2_global
        for k, (ratio, r2k) in enumerate(zip(pc_ratios, r2_ks)):
            delta = r2k - prev_r2
            # ratio*100 formatting takes 12 chars + 1 for the '%', total 13
            print(f"│ {'+ Local PC ' + str(k+1):<32} │ {r2k:>8.4f} │ {delta:>10.4f} │ {ratio*100:>12.1f}% │")
            prev_r2 = r2k

        print(f"├{'─'*34}┼{'─'*10}┼{'─'*12}┼{'─'*15}┤")
        print(f"│ {'Total Explained (All Local PCs)':<32} │ {r2_local:>8.4f} │ {'—':>10} │ {'—':>13} │")
        print(f"│ {'Unexplained Residual':<32} │ {1.0 - r2_local:>8.4f} │ {'—':>10} │ {'—':>13} │")
        print(f"└{'─'*74}┘")

    # ────────────────────────────────────────────────────────────────────

    # ===== Step 1: Functional ANOVA Decomposition =====
    print_section("Step 1: Functional ANOVA Decomposition (Train Only)")

    grand_mean    = X_original[train_slice].mean(axis=(0, 1))
    asset_bias    = X_original[train_slice].mean(axis=0) - grand_mean
    # Market effect must subtract the static train grand_mean
    market_effect = X_original.mean(axis=1) - grand_mean

    # ===== Step 2: Global FPCA =====
    print_section("Step 2: Global FPCA on Market Effect")

    fd_global = FDataGrid(
        data_matrix=market_effect.reshape(N_OBS, S_PTS),
        grid_points=grid_points,
        dataset_name="Global_Market_Effect"
    )

    fpca_global   = FPCA(n_components=n_pc_global)
    fpca_global.fit(fd_global[train_slice]) # Fit strictly on train
    global_scores = fpca_global.transform(fd_global) # Transform full for OOS

    global_components = fpca_global.components_.data_matrix[:n_pc_global]

    if plot_surfaces:
        plot_surfaces_for_latex(
            global_components.reshape(n_pc_global, g.N_MATURITY, g.N_MONEYNESS),
            "results/factors/global",
            color_multiplier=1
        )
        
    print(f" Global PCs kept : {n_pc_global}  (from g.N_PC_GLOBAL)")
    print(f" Expl. Var / PC  : {(fpca_global.explained_variance_ratio_[:n_pc_global]*100).round(2).tolist()}%")

    # ── Build OLS regressor matrix (no intercept) ──────────────────────
    X_reg       = global_scores[:, :n_pc_global]
    X_reg_train = X_reg[train_slice]

    Residuals      = []
    ols_fitted     = []
    r2_global_list = []
    B_j_dict       = {}

    for idx_asset, symbol in enumerate(g.SYMBOLS):
        # Y_j : Centered by static train FANOVA means
        Y_j       = (X_original[:, idx_asset, :, :].reshape(N_OBS, S_PTS)
                     - (grand_mean + asset_bias[idx_asset]).reshape(1, S_PTS))
        Y_j_train = Y_j[train_slice]

        # Fit Beta strictly on train
        B_j, _, _, _ = np.linalg.lstsq(X_reg_train, Y_j_train, rcond=None)

        # Apply Beta to full dataset
        fitted_j = X_reg @ B_j
        resid_j  = Y_j  - fitted_j

        # R² logic
        Y_tr_dm     = Y_j_train - Y_j_train.mean(axis=0)
        ss_total_tr = np.sum(Y_tr_dm ** 2)
        r2_train    = 1.0 - np.sum((Y_j_train - X_reg_train @ B_j) ** 2) / ss_total_tr

        ss_total = np.sum((Y_j - Y_j.mean(axis=0)) ** 2)
        r2_full  = 1.0 - np.sum(resid_j ** 2) / ss_total

        Residuals.append(resid_j.reshape(N_OBS, g.N_MATURITY, g.N_MONEYNESS))
        ols_fitted.append(fitted_j.reshape(N_OBS, g.N_MATURITY, g.N_MONEYNESS))
        r2_global_list.append(r2_full)
        B_j_dict[symbol] = B_j

    Residuals  = np.array(Residuals).transpose(1, 0, 2, 3)
    ols_fitted = np.array(ols_fitted).transpose(1, 0, 2, 3)

    # ── Orthogonality sanity check (FIXED: TRAIN ONLY) ──────────────────
    print(f"\n Orthogonality check — max |Cov(resid, z_k)| on TRAIN set:")
    for k in range(n_pc_global):
        max_cov = max(
            np.abs(np.cov(Residuals[train_slice, j, :, :].reshape(-1, S_PTS).T,
                          global_scores[train_slice, k])[:-1, -1]).max()
            for j in range(len(g.SYMBOLS))
        )
        print(f"    PC{k+1}: {max_cov:.2e}  (should be ≈ 0 on train window)")

    # ===== Step 3: Local FPCA per Asset =====
    print_section("Step 3: Local FPCA per Asset")

    fpca_per_asset    = {}
    local_factor_dfs  = []
    local_scores_dict = {}

    for j, symbol in enumerate(g.SYMBOLS):
        X_j = X_original[:, j, :, :]

        X_j_dm   = X_j - X_j.mean(axis=0)
        ss_total = np.sum(X_j_dm ** 2)

        r2_grand = 1.0 - np.sum((X_j - grand_mean) ** 2) / ss_total
        r2_asset = 1.0 - np.sum((X_j - (grand_mean + asset_bias[j])) ** 2) / ss_total

        ols_full_fit = grand_mean + asset_bias[j] + ols_fitted[:, j, :, :]
        r2_ols       = 1.0 - np.sum((X_j - ols_full_fit) ** 2) / ss_total

        local_residuals_j = Residuals[:, j, :, :]

        fd_local = FDataGrid(
            data_matrix=local_residuals_j.reshape(N_OBS, S_PTS),
            grid_points=grid_points,
            dataset_name=f"Local_Residual_{symbol}"
        )

        # Handle N_PC_LOCAL mapping safely
        M_j = g.N_PC_LOCAL[j] if isinstance(g.N_PC_LOCAL, (list, np.ndarray)) else n_pc_local

        fpca_local   = FPCA(n_components=M_j)
        fpca_local.fit(fd_local[train_slice]) # Fit strictly on train
        local_scores = fpca_local.transform(fd_local) # Transform full for OOS
        
        fpca_per_asset[symbol]    = fpca_local
        local_scores_dict[symbol] = local_scores[:, :M_j]

        local_recon_flat = (local_scores[:, :M_j] @ fpca_local.components_.data_matrix[:M_j].reshape(M_j, -1))
        local_recon_surf = local_recon_flat.reshape(N_OBS, g.N_MATURITY, g.N_MONEYNESS)

        r2_local = 1.0 - np.sum((X_j - (ols_full_fit + local_recon_surf)) ** 2) / ss_total

        pc_ratios, r2_ks = [], []
        for k in range(M_j):
            pc_ratios.append(fpca_local.explained_variance_ratio_[k])
            partial_flat = (local_scores[:, :k+1] @ fpca_local.components_.data_matrix[:k+1].reshape(k+1, -1))
            partial_surf = partial_flat.reshape(N_OBS, g.N_MATURITY, g.N_MONEYNESS)
            r2_ks.append(1.0 - np.sum((X_j - (ols_full_fit + partial_surf)) ** 2) / ss_total)

        print_asset_local_table(symbol, r2_grand, r2_asset, r2_ols, r2_local, pc_ratios, r2_ks)

        if plot_surfaces:
            plot_surfaces_for_latex(
                fpca_local.components_.data_matrix[:M_j].reshape(M_j, g.N_MATURITY, g.N_MONEYNESS),
                f"results/factors/local/{symbol}",
                color_multiplier=0.7
            )

        cols     = [f"L_PC_{symbol}_{k+1}" for k in range(M_j)]
        df_local = pd.DataFrame(local_scores[:, :M_j], index=dates, columns=cols)
        local_factor_dfs.append(df_local)

    # ===== Step 4: Save Outputs =====
    df_global = pd.DataFrame(
        global_scores[:, :n_pc_global],
        index=dates,
        columns=[f"G_PC_{k+1}" for k in range(n_pc_global)]
    )

    df_all_factors = pd.concat([df_global] + local_factor_dfs, axis=1)
    
    out_path = "results/factors/factors.csv"
    df_all_factors.to_csv(out_path)
