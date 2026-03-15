import numpy as np
import random
import pandas as pd
import skfda
from skfda.preprocessing.dim_reduction import FPCA
from skfda.representation.grid import FDataGrid
import pickle as pkl
from pathlib import Path

# --- ADD THIS TO SUPPRESS PLOT POP-UPS ---
import matplotlib
matplotlib.use('Agg') 
# -----------------------------------------

import config.settings as g
from src.compression.helpers.data_preparation import get_clean_4d_tensor
from src.compression.helpers.visualisation_function import plot_surfaces_for_latex
from src.compression.helpers.reconstruction import Reconstruction

if __name__ == "__main__":
    # --- Configuration ---
    n_pc_global   = g.N_PC_GLOBAL
    n_pc_local    = g.N_PC_LOCAL
    plot_surfaces = True
    train_slice   = slice(None, g.JAN_2025)
    np.random.seed(42)
    random.seed(42)

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

    def print_asset_local_table(symbol, 
                                r2_g_tr, r2_a_tr, r2_o_tr, r2_l_tr, 
                                r2_g_fu, r2_a_fu, r2_o_fu, r2_l_fu, 
                                pc_ratios, r2_ks_tr, r2_ks_fu):
        
        top_line = f"─ {symbol} "
        pad_len = WIDTH - len(top_line) - 2
        
        print(f"\n┌{top_line}{'─' * pad_len}┐")
        print(f"│ {'Metric':<31} │ {'R² (Train)':>10} │ {'R² (Full)':>10} │ {'% of Resid':>12} │")
        print(f"├{'─'*33}┼{'─'*12}┼{'─'*12}┼{'─'*14}┤")
        
        print(f"│ {'Grand Mean only':<31} │ {r2_g_tr:>10.4f} │ {r2_g_fu:>10.4f} │ {'—':>12} │")
        print(f"│ {'+ Asset Bias':<31} │ {r2_a_tr:>10.4f} │ {r2_a_fu:>10.4f} │ {'—':>12} │")
        print(f"│ {'+ Global (surface β, no int.)':<31} │ {r2_o_tr:>10.4f} │ {r2_o_fu:>10.4f} │ {'—':>12} │")

        for k, (ratio, r2_tr, r2_fu) in enumerate(zip(pc_ratios, r2_ks_tr, r2_ks_fu)):
            print(f"│ {'+ Local PC ' + str(k+1):<31} │ {r2_tr:>10.4f} │ {r2_fu:>10.4f} │ {ratio*100:>11.1f}% │")

        print(f"├{'─'*33}┼{'─'*12}┼{'─'*12}┼{'─'*14}┤")
        print(f"│ {'Total Explained (All PCs)':<31} │ {r2_l_tr:>10.4f} │ {r2_l_fu:>10.4f} │ {'—':>12} │")
        print(f"│ {'Unexplained Residual':<31} │ {1.0 - r2_l_tr:>10.4f} │ {1.0 - r2_l_fu:>10.4f} │ {'—':>12} │")
        print(f"└{'─'*74}┘")
    
    def print_idiosyncrasy_diagnostic(local_scores_dict, global_scores, n_pc_global):
        symbols  = list(local_scores_dict.keys())
        N_ASSETS = len(symbols)
        n_local  = local_scores_dict[symbols[0]].shape[1]

        # ── Part 1: local scores vs global scores ────────────────────────
        print(f"\n  ┌─ Part 1 : Temporal Orthogonality  corr(ξ_jm=1, z_k) {'─'*14}┐")
        print(f"  │  Guaranteed ≈ 0 by OLS — sanity check.                             │")
        print(f"  │  Action only if max |corr| > 0.10                                  │")
        print(f"  │                                                                    │")
        print(f"  │  {'':>6}" + "".join(f"{'z_' + str(k+1):>8}" for k in range(n_pc_global)) + "  │")
        print(f"  │  {'─'*6}" + "".join("─"*8 for _ in range(n_pc_global)) + "  │")

        max_part1 = 0.0
        for sym in symbols:
            scores_j = local_scores_dict[sym]
            row = f"  │  {sym:<6}"
            for k in range(n_pc_global):
                c = np.corrcoef(scores_j[:, 0], global_scores[:, k])[0, 1]
                max_part1 = max(max_part1, abs(c))
                row += f"{c:>8.3f}"
            row += "  │"
            print(row)

        verdict_p1 = ("✓ clean" if max_part1 < 0.10 else "✗ leakage — increase N_PC_GLOBAL")
        print(f"  │                                                                    │")
        print(f"  │  Max |corr| = {max_part1:.3f}   {verdict_p1:<40}   │")
        print(f"  └{'─' * 66}┘")

        # ── Part 2: cross-asset local PC correlations ────────────────────
        print(f"\n  ┌─ Part 2 : Cross-Asset Local Score Correlation {'─'*16}┐")
        print(f"  │  Dependence signal — feed directly to copula.                      │")
        print(f"  │  High values are expected and desirable, NOT a problem.            │")

        for m in range(n_local):
            pc_m_matrix = np.column_stack([local_scores_dict[sym][:, m] for sym in symbols])
            C = np.corrcoef(pc_m_matrix.T)
            np.fill_diagonal(C, np.nan)
            abs_C = np.abs(C)

            max_corr  = np.nanmax(abs_C)
            mean_corr = np.nanmean(abs_C)
            p95_corr  = np.nanpercentile(abs_C, 95)

            print(f"  │                                                                    │")
            print(f"  │  Local PC {m+1}   max={max_corr:.3f}  mean={mean_corr:.3f}  p95={p95_corr:.3f}{'':>20}│")
            print(f"  │  {'':>6}" + "".join(f"{s:>8}" for s in symbols) + "  │")
            print(f"  │  {'─'*6}" + "".join("─"*8 for _ in symbols) + "  │")

            for i, sym in enumerate(symbols):
                row = f"  │  {sym:<6}"
                for jj in range(N_ASSETS):
                    val = C[i, jj]
                    row += f"{'—':>8}" if np.isnan(val) else f"{val:>8.3f}"
                row += "  │"
                print(row)

        print(f"  └{'─' * 66}┘")

        # ===== Step 1: Functional ANOVA Decomposition =====
    print_section("Step 1: Functional ANOVA Decomposition (Train Only)")

    grand_mean    = X_original[train_slice].mean(axis=(0, 1))
    asset_bias    = X_original[train_slice].mean(axis=0) - grand_mean
    market_effect = X_original.mean(axis=1) - grand_mean

    # ===== Step 2: Global FPCA =====
    print_section("Step 2: Global FPCA on Market Effect")

    fd_global = FDataGrid(
        data_matrix=market_effect.reshape(N_OBS, S_PTS),
        grid_points=grid_points,
        dataset_name="Global_Market_Effect"
    )

    fpca_global   = FPCA(n_components=n_pc_global)
    fpca_global.fit(fd_global[train_slice]) 
    global_scores = fpca_global.transform(fd_global) 

    global_components = fpca_global.components_.data_matrix[:n_pc_global]

    if plot_surfaces:
        plot_surfaces_for_latex(
            global_components.reshape(n_pc_global, g.N_MATURITY, g.N_MONEYNESS),
            "results/factors/global",
            color_multiplier=1
        )
        
    print(f" Global PCs kept : {n_pc_global}  (from g.N_PC_GLOBAL)")
    print(f" Expl. Var / PC  : {(fpca_global.explained_variance_ratio_[:n_pc_global]*100).round(2).tolist()}%")

    X_reg       = global_scores[:, :n_pc_global]
    X_reg_train = X_reg[train_slice]

    Residuals      = []
    ols_fitted     = []
    r2_global_list = []
    B_j_dict       = {}

    for idx_asset, symbol in enumerate(g.SYMBOLS):
        Y_j       = (X_original[:, idx_asset, :, :].reshape(N_OBS, S_PTS)
                     - (grand_mean + asset_bias[idx_asset]).reshape(1, S_PTS))
        Y_j_train = Y_j[train_slice]

        B_j, _, _, _ = np.linalg.lstsq(X_reg_train, Y_j_train, rcond=None)

        fitted_j = X_reg @ B_j
        resid_j  = Y_j  - fitted_j

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

    # ===== Step 3: Local FPCA per Asset =====
    print_section("Step 3: Local FPCA per Asset")

    fpca_per_asset    = {}
    local_factor_dfs  = []
    local_scores_dict = {}
    surfaces_dict     = {}

    for j, symbol in enumerate(g.SYMBOLS):
        # Full Sample variables
        X_j = X_original[:, j, :, :]
        X_j_dm = X_j - X_j.mean(axis=0)
        ss_total_full = np.sum(X_j_dm ** 2)

        # Train Sample variables
        X_j_train = X_j[train_slice]
        X_j_dm_train = X_j_train - X_j_train.mean(axis=0)
        ss_total_train = np.sum(X_j_dm_train ** 2)

        r2_grand_full = 1.0 - np.sum((X_j - grand_mean) ** 2) / ss_total_full
        r2_grand_tr   = 1.0 - np.sum((X_j_train - grand_mean) ** 2) / ss_total_train

        r2_asset_full = 1.0 - np.sum((X_j - (grand_mean + asset_bias[j])) ** 2) / ss_total_full
        r2_asset_tr   = 1.0 - np.sum((X_j_train - (grand_mean + asset_bias[j])) ** 2) / ss_total_train

        ols_full_fit  = grand_mean + asset_bias[j] + ols_fitted[:, j, :, :]
        ols_train_fit = ols_full_fit[train_slice]
        
        r2_ols_full = 1.0 - np.sum((X_j - ols_full_fit) ** 2) / ss_total_full
        r2_ols_tr   = 1.0 - np.sum((X_j_train - ols_train_fit) ** 2) / ss_total_train

        local_residuals_j = Residuals[:, j, :, :]

        fd_local = FDataGrid(
            data_matrix=local_residuals_j.reshape(N_OBS, S_PTS),
            grid_points=grid_points,
            dataset_name=f"Local_Residual_{symbol}"
        )

        M_j = g.N_PC_LOCAL[j] if isinstance(g.N_PC_LOCAL, (list, np.ndarray)) else n_pc_local

        fpca_local   = FPCA(n_components=M_j)
        fpca_local.fit(fd_local[train_slice]) 
        local_scores = fpca_local.transform(fd_local) 
        
        fpca_per_asset[symbol]    = fpca_local
        local_scores_dict[symbol] = local_scores[:, :M_j]

        local_recon_flat = (local_scores[:, :M_j] @ fpca_local.components_.data_matrix[:M_j].reshape(M_j, -1))
        local_recon_surf_full = local_recon_flat.reshape(N_OBS, g.N_MATURITY, g.N_MONEYNESS)
        local_recon_surf_tr   = local_recon_surf_full[train_slice]

        r2_local_full = 1.0 - np.sum((X_j - (ols_full_fit + local_recon_surf_full)) ** 2) / ss_total_full
        r2_local_tr   = 1.0 - np.sum((X_j_train - (ols_train_fit + local_recon_surf_tr)) ** 2) / ss_total_train

        pc_ratios, r2_ks_tr, r2_ks_full = [], [], []
        for k in range(M_j):
            pc_ratios.append(fpca_local.explained_variance_ratio_[k])
            
            partial_flat = (local_scores[:, :k+1] @ fpca_local.components_.data_matrix[:k+1].reshape(k+1, -1))
            partial_surf_full = partial_flat.reshape(N_OBS, g.N_MATURITY, g.N_MONEYNESS)
            partial_surf_tr   = partial_surf_full[train_slice]
            
            r2_ks_full.append(1.0 - np.sum((X_j - (ols_full_fit + partial_surf_full)) ** 2) / ss_total_full)
            r2_ks_tr.append(1.0 - np.sum((X_j_train - (ols_train_fit + partial_surf_tr)) ** 2) / ss_total_train)

        print_asset_local_table(symbol, 
                                r2_grand_tr, r2_asset_tr, r2_ols_tr, r2_local_tr, 
                                r2_grand_full, r2_asset_full, r2_ols_full, r2_local_full, 
                                pc_ratios, r2_ks_tr, r2_ks_full)

        if plot_surfaces:
            plot_surfaces_for_latex(
                fpca_local.components_.data_matrix[:M_j].reshape(M_j, g.N_MATURITY, g.N_MONEYNESS),
                f"results/factors/local/{symbol}",
                color_multiplier=0.7
            )

        cols     = [f"L_PC_{symbol}_{k+1}" for k in range(M_j)]
        df_local = pd.DataFrame(local_scores[:, :M_j], index=dates, columns=cols)
        local_factor_dfs.append(df_local)

        # ── CREATE THE OOP RECONSTRUCTION OBJECT ──
        # Try fetching grid arrays if they exist, otherwise pass None
        t_grid = g.T_GRID if hasattr(g, 'T_GRID') else None
        k_grid = g.K_GRID if hasattr(g, 'K_GRID') else None

        surfaces_dict[symbol] = Reconstruction(
            asset            = symbol,
            grand_mean       = grand_mean,
            asset_bias       = asset_bias[j],
            global_scores    = global_scores[:, :n_pc_global],
            B_j              = B_j_dict[symbol],
            local_components = fpca_local.components_.data_matrix[:M_j].reshape(M_j, g.N_MATURITY, g.N_MONEYNESS),
            local_scores     = local_scores[:, :M_j],
            residuals        = Residuals[:, j, :, :].reshape(N_OBS, S_PTS),
            maturity_labels  = [str(m) for m in t_grid] if t_grid is not None else None,
            moneyness_labels = [str(k) for k in k_grid] if k_grid is not None else None,
        )

    # ===== Step 3b: Idiosyncrasy Diagnostic =====
    print_section("Step 3b: Idiosyncrasy Diagnostic")
    print_idiosyncrasy_diagnostic(local_scores_dict, global_scores, n_pc_global)

    # ===== Step 4: Save Outputs =====
    print_header("Step 4: Save Outputs")

    df_global = pd.DataFrame(
        global_scores[:, :n_pc_global],
        index=dates,
        columns=[f"G_PC_{k+1}" for k in range(n_pc_global)]
    )

    df_all_factors = pd.concat([df_global] + local_factor_dfs, axis=1)
    
    # Save the CSV
    out_path_csv = "results/factors/factors.csv"
    Path(out_path_csv).parent.mkdir(parents=True, exist_ok=True)
    df_all_factors.to_csv(out_path_csv)
    
    # Save the Object Dictionary
    out_path_pkl = "results/factors/surfaces_dict.pkl"
    with open(out_path_pkl, "wb") as f:
        pkl.dump(surfaces_dict, f)
