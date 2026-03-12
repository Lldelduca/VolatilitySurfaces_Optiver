import pyvinecopulib as pv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random

def analyze_tree_subsets_interactive(model, u_train, var_names, name, graph_dir):
    d = len(var_names)
    is_spot = {i: ('PC' not in var_names[i]) for i in range(d)}
    
    model_json = model.to_json()
    M = np.array(model.matrix, dtype=np.int64)
    if M.max() == d: M = M - 1  
    if np.sum(M[0] >= 0) > np.sum(M[-1] >= 0): M = np.flipud(M)
        
    results = []
    max_tree = d - 1 
    
    print(f"\nAnalyzing Economic Subsets and In-Sample LL for {name} ({max_tree} trees)...")
    
    # Track the algorithmic cutoff (First time Spot-Spot density <= 5%)
    algorithmic_cutoff = None
    
    # 1. Print Header
    print(f"\n{'Tree':<6} | {'Spot-Spot':<12} | {'Density':<10} | {'Spot-Vol':<10} | {'Vol-Vol':<10}")
    print("-" * 60)
    
    for tree_idx in range(max_tree): 
        lvl = tree_idx + 1
        spot_spot, spot_vol, vol_vol = 0, 0, 0
        edges = d - 1 - tree_idx
        row = d - 1 - tree_idx
        
        for col in range(edges):
            v1_spot = is_spot[M[row, col]]
            v2_spot = is_spot[M[col, col]]
            
            if v1_spot and v2_spot: spot_spot += 1
            elif not v1_spot and not v2_spot: vol_vol += 1
            else: spot_vol += 1
                
        total_edges = spot_spot + spot_vol + vol_vol
        spot_density = (spot_spot / total_edges) if total_edges > 0 else 0.0
        
        # Algorithmic Detection: 5% Topological Density Rule
        if algorithmic_cutoff is None and spot_density <= 0.05:
            algorithmic_cutoff = lvl
                
        pct_involving_spot = ((spot_spot + spot_vol) / total_edges) * 100 if total_edges > 0 else 0
        
        trunc_model = pv.Vinecop.from_json(model_json)
        trunc_model.truncate(lvl) 
        is_ll = trunc_model.loglik(u_train)
        
        results.append({
            'Tree': lvl, 'Spot_Spot': spot_spot, 'Spot_Vol': spot_vol, 'Vol_Vol': vol_vol,
            'Pct_Involving_Spot': pct_involving_spot, 'IS_LL': is_ll
        })
        
        # 2. Print Exact Numbers (only up to tree 30 to avoid terminal spam)
        if lvl <= 30:
            marker = "<-- 5% CUTOFF MET" if lvl == algorithmic_cutoff else ""
            print(f"{lvl:<6} | {spot_spot:<12} | {spot_density*100:>5.1f}%    | {spot_vol:<10} | {vol_vol:<10} {marker}")

    df_res = pd.DataFrame(results)
    
    # Fallback if it never hits 5% (extremely unlikely)
    if algorithmic_cutoff is None:
        algorithmic_cutoff = max_tree
    
    print(f"\n" + "="*50)
    print(f"--- DYNAMIC ECONOMIC INSIGHTS: {name} ---")
    print("="*50)
    print(f"The algorithm detected the 5% density terminal plateau at Tree {algorithmic_cutoff}.")
    
    # --- PLOT GRAPH WITHOUT THE LINE ---
    sns.set_theme(style="white", context="paper", font_scale=1.2)
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    ax1.bar(df_res['Tree'], df_res['Spot_Spot'], label='Spot-Spot (Contagion)', color='#3f51b5', edgecolor='white', linewidth=0.5)
    ax1.bar(df_res['Tree'], df_res['Spot_Vol'], bottom=df_res['Spot_Spot'], label='Spot-Vol (Leverage)', color='#c62828', edgecolor='white', linewidth=0.5)
    ax1.bar(df_res['Tree'], df_res['Vol_Vol'], bottom=df_res['Spot_Spot']+df_res['Spot_Vol'], label='Vol-Vol (Conditional Noise)', color='#bdbdbd', alpha=0.7, edgecolor='white', linewidth=0.5)
    
    ax1.set_xlabel("Vine Tree Level", fontsize=14)
    ax1.set_ylabel("Number of Edges (Subset Composition)", fontsize=14)
    ax1.set_xticks(np.arange(1, d, 2))
    ax1.set_xlim(0, d)
    
    ax2 = ax1.twinx()
    ax2.plot(df_res['Tree'], df_res['IS_LL'], color='darkgreen', marker='o', linewidth=2.5, markersize=5, label='In-Sample Log-Likelihood')
    ax2.set_ylabel("In-Sample Log-Likelihood", color='darkgreen', fontsize=14)
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right', frameon=True, shadow=True)
    
    plt.title(f"Structural Validation & Economic Decay: {name}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 3. Show the plot interactively to the user
    print("\n[!] Please review the popup graph.")
    plt.show() 
    
    # 4. User Input 
    user_input = input(f"\nPress Enter to accept Algorithm Truncation [{algorithmic_cutoff}] or type a manual override: ")
    chosen_k = int(user_input) if user_input.strip() else algorithmic_cutoff
    
    # 5. Draw the final line and save the PNG
    ax1.axvline(x=chosen_k, color='black', linestyle='--', linewidth=2.5, label=rf'Contagion Density $\leq$ 5% (Tree {chosen_k})')
    
    # Refresh legend to include the new black line
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right', frameon=True, shadow=True)
    
    save_path = os.path.join(graph_dir, f"Decay_{name}.png")
    fig.savefig(save_path, dpi=300)
    print(f"Saved final graph with truncation line at {chosen_k} to {save_path}")
    
    return chosen_k

if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    res_dir = os.path.join(project_root, "results", "dynamics")
    graph_dir = os.path.join(project_root, "results", "copulas", "static", "plots")
    os.makedirs(graph_dir, exist_ok=True)

    u_spot_file = os.path.join(res_dir, "NGARCH", "uniforms_ngarch_train.csv")
    u_har_file = os.path.join(res_dir, "HAR_GARCH", "uniforms_har_garch_evt_train.csv")
    u_nsde_file = os.path.join(res_dir, "NSDE", "uniforms_nsde_train.csv")

    u_spot = pd.read_csv(u_spot_file, index_col='Date', parse_dates=True)
    u_har = pd.read_csv(u_har_file, index_col='Date', parse_dates=True)
    u_nsde = pd.read_csv(u_nsde_file, index_col='Date', parse_dates=True)

    global_valid_dates = u_spot.index.intersection(u_har.index).intersection(u_nsde.index)
    u_spot, u_har, u_nsde = u_spot.loc[global_valid_dates], u_har.loc[global_valid_dates], u_nsde.loc[global_valid_dates]

    factor_sets = {"HAR-GARCH-EVT": u_har, "NSDE": u_nsde}
    optimal_ranks = {}

    for factor_name, u_factors in factor_sets.items():
        print(f"\n{'='*50}\n--- Exploring Joint Copula: Spot + {factor_name} ---\n{'='*50}")
        combined_u_train = pd.concat([u_spot, u_factors], axis=1)
        np_data_train = combined_u_train.to_numpy()
        var_names = combined_u_train.columns.tolist()
        d = np_data_train.shape[1]

        controls = pv.FitControlsVinecop(
            family_set=[pv.BicopFamily.indep, pv.BicopFamily.gaussian, pv.BicopFamily.student, 
                        pv.BicopFamily.frank, pv.BicopFamily.clayton, pv.BicopFamily.gumbel],
            selection_criterion="aic", tree_criterion="tau", allow_rotations=True,
            num_threads=os.cpu_count()-1, threshold=0.05, trunc_lvl=d-1
        )
        exploratory_model = pv.Vinecop(d=d)
        exploratory_model.select(np_data_train, controls=controls)
        
        chosen_k = analyze_tree_subsets_interactive(exploratory_model, np_data_train, var_names, factor_name, graph_dir)
        optimal_ranks[factor_name] = chosen_k

    print("\n" + "="*50)
    print("PHASE 1 COMPLETE. UPDATE YOUR config/settings.py WITH:")
    print(f"K_HAR_GARCH = {optimal_ranks['HAR-GARCH-EVT']}")
    print(f"K_NSDE = {optimal_ranks['NSDE']}")
    print("="*50)