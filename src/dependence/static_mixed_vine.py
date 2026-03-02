import pyvinecopulib as pv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os

# Economic Subset Analysis for Truncation
# --- Comprehensive Truncation Analysis (IS LL + Economic Subsets) ---
def analyze_tree_subsets_and_is_ll(model, u_train, var_names, name, save_path):
    d = len(var_names)
    is_spot = {i: ('PC' not in var_names[i]) for i in range(d)}
    json = model.to_json()
    
    # Safely extract and format the R-Vine Matrix
    M = np.array(model.matrix, dtype=np.int64)
    if M.max() == d: M = M - 1  
    if np.sum(M[0] >= 0) > np.sum(M[-1] >= 0): M = np.flipud(M)
        
    results = []
    max_tree = d - 1 
    
    print(f"\nAnalyzing Economic Subsets and In-Sample LL for {name} ({max_tree} trees)...")
    
    for tree_idx in range(max_tree): 
        lvl = tree_idx + 1
        spot_spot, spot_vol, vol_vol = 0, 0, 0
        edges = d - 1 - tree_idx
        row = d - 1 - tree_idx
        
        # --- A. Subset Analysis ---
        for col in range(edges):
            v1_spot = is_spot[M[row, col]]
            v2_spot = is_spot[M[col, col]]
            
            if v1_spot and v2_spot: spot_spot += 1
            elif not v1_spot and not v2_spot: vol_vol += 1
            else: spot_vol += 1
                
        total_edges = spot_spot + spot_vol + vol_vol
        pct_involving_spot = ((spot_spot + spot_vol) / total_edges) * 100 if total_edges > 0 else 0
        
        # --- B. In-Sample Log-Likelihood Analysis ---
        # Load a fresh copy from JSON so .truncate() evaluates cleanly
        trunc_model = pv.Vinecop.from_json(json) 
        trunc_model.truncate(lvl) 
        is_ll = trunc_model.loglik(u_train)
        
        results.append({
            'Tree': lvl, 
            'Spot_Spot': spot_spot,
            'Spot_Vol': spot_vol,
            'Vol_Vol': vol_vol,
            'Pct_Involving_Spot': pct_involving_spot,
            'IS_LL': is_ll
        })
        
        print(f"  Tree {lvl:2d} | IS LL: {is_ll:.2f} | S-S: {spot_spot:2d}, S-V: {spot_vol:2d}, V-V: {vol_vol:2d} | Spot%: {pct_involving_spot:.1f}%")

    df_res = pd.DataFrame(results)
    
    # --- FIND OPTIMAL ECONOMIC TRUNCATION ---
    # We truncate the moment Spot involvement drops below 5%
    mask = df_res['Pct_Involving_Spot'] < 5.0
    if mask.any():
        optimal_k = int(df_res[mask]['Tree'].min()) - 1
        optimal_k = max(1, optimal_k) 
    else:
        optimal_k = int(df_res['Tree'].max())
        
    print(f"\n>> Optimal Economic Truncation Level (Spot edges < 5%): Tree {optimal_k} <<")
    
    # --- PLOT DUAL-AXIS GRAPH ---
    sns.set_theme(style="white", context="paper", font_scale=1.2)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Left Axis: Subsets (Bar Chart)
    ax1.bar(df_res['Tree'], df_res['Spot_Spot'], label='Spot-Spot (Contagion)', color='darkblue', alpha=0.7)
    ax1.bar(df_res['Tree'], df_res['Spot_Vol'], bottom=df_res['Spot_Spot'], label='Spot-Vol (Leverage)', color='darkred', alpha=0.7)
    ax1.bar(df_res['Tree'], df_res['Vol_Vol'], bottom=df_res['Spot_Spot']+df_res['Spot_Vol'], label='Vol-Vol (Conditional Noise)', color='gray', alpha=0.4)
    ax1.set_xlabel("Vine Tree Level", fontsize=14)
    ax1.set_ylabel("Number of Edges (Subset Composition)", fontsize=14)
    ax1.set_xticks(df_res['Tree'][::2])
    
    # Right Axis: In-Sample Log-Likelihood (Line Chart)
    ax2 = ax1.twinx()
    ax2.plot(df_res['Tree'], df_res['IS_LL'], color='darkgreen', marker='o', linewidth=3, markersize=6, label='In-Sample Log-Likelihood')
    ax2.set_ylabel("In-Sample Log-Likelihood", color='darkgreen', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='darkgreen')
    
    # Highlight Optimal K
    #ax1.axvline(x=optimal_k, color='black', linestyle='--', linewidth=2.5, label=f'Optimal Truncation ($K={optimal_k}$)')
    
    # Legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right', frameon=True, shadow=True)
    
    plt.title(f"Structural Validation & Economic Decay: {name}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return optimal_k

# --- Fits Final Static Mixed R-Vine Copula ---
def fit_static_mixed_vine(u_data, optimal_trunc_lvl):
    T, N = u_data.shape
    controls = pv.FitControlsVinecop(
        family_set=[
            pv.BicopFamily.indep, pv.BicopFamily.gaussian, pv.BicopFamily.student,
            pv.BicopFamily.frank, pv.BicopFamily.clayton, pv.BicopFamily.gumbel,
        ],
        selection_criterion="aic",
        tree_criterion="tau",        
        allow_rotations=True,        
        num_threads=os.cpu_count()-1,
        threshold=0.05,
        trunc_lvl=optimal_trunc_lvl
    )
    model = pv.Vinecop(d=N)
    model.select(u_data, controls=controls)
    return model

# --- Visualizations ---
def plot_tree1_network(u_df, name, save_path):
    # 1. Calculate empirical Kendall's Tau
    tau_matrix = u_df.corr(method='kendall')
    
    # 2. Create a NetworkX Graph
    G = nx.Graph()
    for col in tau_matrix.columns:
        G.add_node(col)
        
    # Add all edges with absolute tau as the weight
    for i in range(len(tau_matrix.columns)):
        for j in range(i + 1, len(tau_matrix.columns)):
            col1 = tau_matrix.columns[i]
            col2 = tau_matrix.columns[j]
            real_tau = tau_matrix.iloc[i, j]
            weight = abs(real_tau)
            G.add_edge(col1, col2, weight=weight, tau=real_tau)
            
    # 3. Find the Maximum Spanning Tree (Tree 1 of the Vine)
    mst = nx.maximum_spanning_tree(G, weight='weight')
    
    # 4. Plotting Setup
    plt.figure(figsize=(16, 12))
    
    # Color coding: Volatility Factors vs Spot Assets
    node_colors = []
    for node in mst.nodes():
        if 'PC' in node:  # Volatility Factors
            node_colors.append('lightcoral')
        else:             # Spot Assets
            node_colors.append('lightblue')
            
    # Edge widths and colors based on Tau strength and sign
    edges = mst.edges(data=True)
    edge_widths = [d['weight'] * 6 for u, v, d in edges] 
    edge_colors = ['darkred' if d['tau'] < 0 else 'darkgreen' for u, v, d in edges]
    
    # Layout 
    pos = nx.spring_layout(mst, k=0.5, iterations=50, seed=42)
    
    nx.draw_networkx_nodes(mst, pos, node_color=node_colors, node_size=2000, edgecolors='black', linewidths=1.5)
    nx.draw_networkx_edges(mst, pos, width=edge_widths, edge_color=edge_colors, alpha=0.7)
    nx.draw_networkx_labels(mst, pos, font_size=10, font_weight='bold')
    
    import matplotlib.lines as mlines
    pos_line = mlines.Line2D([], [], color='darkgreen', linewidth=3, label='Positive Dep (Tau > 0)')
    neg_line = mlines.Line2D([], [], color='darkred', linewidth=3, label='Negative Dep (Tau < 0)')
    spot_patch = mlines.Line2D([], [], color='lightblue', marker='o', linestyle='None', markersize=15, markeredgecolor='black', label='Spot Asset')
    fac_patch = mlines.Line2D([], [], color='lightcoral', marker='o', linestyle='None', markersize=15, markeredgecolor='black', label='Volatility Factor')
    
    plt.legend(handles=[pos_line, neg_line, spot_patch, fac_patch], loc='upper left', fontsize=12, framealpha=0.9)
    plt.title(f"Vine Copula Tree 1 (Market Hubs) - {name}", fontsize=20, fontweight='bold')
    plt.axis('off')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_large_heatmap(u_df, name, save_path):
    plt.figure(figsize=(20, 16))
    tau_matrix = u_df.corr(method='kendall')
    sns.heatmap(tau_matrix, annot=False, cmap='coolwarm', center=0, 
                cbar_kws={'label': "Kendall's Tau"})
    plt.title(f"Empirical Kendall's Tau (Train Data): {name}", fontsize=20)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_family_dist(model, name, save_path):
    families = []
    for tree_cops in model.pair_copulas:
        for bicop in tree_cops:
            families.append(str(bicop.family).split('.')[-1].capitalize())
    
    plt.figure(figsize=(10, 6))
    pd.Series(families).value_counts().plot(kind='bar', color='steelblue', edgecolor='black')
    plt.title(f"Vine Copula Family Distribution: {name}", fontsize=16)
    plt.ylabel("Frequency (Edges)")
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_simulated_vs_empirical(model, u_df, name, save_path):
    """Simulates from the copula to verify goodness-of-fit on a diverse set of pairs."""
    # Force a deep copy so the underlying array is completely mutable
    tau_matrix = u_df.corr(method='kendall').copy()
    
    # Safely overwrite the diagonal using Pandas native indexer
    for i in range(len(tau_matrix)):
        tau_matrix.iloc[i, i] = np.nan
    
    # 1. Strongest Positive Pair
    pos_idx = tau_matrix.unstack().idxmax()
    # 2. Strongest Negative Pair
    neg_idx = tau_matrix.unstack().idxmin()
    # 3. Weakest Pair (Closest to 0)
    abs_tau = tau_matrix.abs()
    weak_idx = abs_tau.unstack().idxmin()

    selected_pairs = [pos_idx, neg_idx, weak_idx]
    pair_labels = ["Strongest Positive", "Strongest Negative", "Weakest (Independent)"]

    # Simulate from Copula
    sim_u = model.simulate(n=len(u_df), seeds=[])
    sim_df = pd.DataFrame(sim_u, columns=u_df.columns)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Goodness of Fit: Empirical vs Simulated ({name})", fontsize=18, fontweight='bold')
    
    for i, ((var1, var2), label) in enumerate(zip(selected_pairs, pair_labels)):
        # Empirical
        axes[0, i].scatter(u_df[var1], u_df[var2], s=5, alpha=0.4, color='darkblue')
        axes[0, i].set_title(f"Empirical | {label}\n{var1} vs {var2}", fontsize=12)
        axes[0, i].set_xlim(0, 1); axes[0, i].set_ylim(0, 1)
        axes[0, i].grid(alpha=0.3, linestyle='--')
        
        # Simulated
        axes[1, i].scatter(sim_df[var1], sim_df[var2], s=5, alpha=0.4, color='darkred')
        axes[1, i].set_title(f"Simulated | {label}\n{var1} vs {var2}", fontsize=12)
        axes[1, i].set_xlim(0, 1); axes[1, i].set_ylim(0, 1)
        axes[1, i].grid(alpha=0.3, linestyle='--')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

    res_dir = os.path.join(project_root, "results", "dynamics")
    out_dir = os.path.join(project_root, "results", "copulas", "static")
    graph_dir = os.path.join(project_root, "results", "copulas", "static", "plots")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)
    
    # --- LOAD FULL TRAIN DATA (100%) ---
    u_spot_file = os.path.join(res_dir, "NGARCH", "train_uniforms_ngarch_t.csv")
    u_har_file = os.path.join(res_dir, "HAR_GARCH", "train_uniforms_har_garch_evt.csv")
    u_nsde_file = os.path.join(res_dir, "NSDE", "train_nsde_uniforms.csv")

    u_spot = pd.read_csv(u_spot_file, index_col='Date', parse_dates=True)
    u_har = pd.read_csv(u_har_file, index_col='Date', parse_dates=True)
    u_nsde = pd.read_csv(u_nsde_file, index_col='Date', parse_dates=True)

    u_spot.index = pd.to_datetime(u_spot.index).normalize()
    u_har.index = pd.to_datetime(u_har.index).normalize()
    u_nsde.index = pd.to_datetime(u_nsde.index).normalize()

    global_valid_dates = u_spot.index.intersection(u_har.index).intersection(u_nsde.index)
    u_spot = u_spot.loc[global_valid_dates]
    u_har = u_har.loc[global_valid_dates]
    u_nsde = u_nsde.loc[global_valid_dates]

    print(f"Total Training Period: {global_valid_dates[0].date()} to {global_valid_dates[-1].date()} (N={len(global_valid_dates)})\n")

    factor_sets = {"HAR-GARCH-EVT": u_har, "NSDE": u_nsde}

    for factor_name, u_factors in factor_sets.items():
        print(f"{'='*50}")
        print(f"--- Fitting Joint Copula: Spot + {factor_name} ---")
        print(f"{'='*50}")

        combined_u_train = pd.concat([u_spot, u_factors], axis=1)
        np_data_train = combined_u_train.to_numpy()
        var_names = combined_u_train.columns.tolist()
        
        save_prefix = f"joint_vine_spot_{factor_name.lower().replace('-', '_')}"
        T, d = np_data_train.shape

        # 1. Fit Exploratory Model on 100% of Train Data
        print(f"Fitting exploratory model (Full {d-1} Trees)...")
        exploratory_model = fit_static_mixed_vine(np_data_train, optimal_trunc_lvl=d-1)
        
        # 2. Comprehensive Analysis (IS LL + Economic Decay)
        comp_plot_path = os.path.join(graph_dir, f"{save_prefix}_comprehensive_truncation.png")
        opt_level = analyze_tree_subsets_and_is_ll(exploratory_model, np_data_train, var_names, factor_name, comp_plot_path)

        # 3. Fit Final Model truncated at the optimal economic level
        print(f"\nFitting final parsimonious model truncated at Tree {opt_level}...")
        joint_model = fit_static_mixed_vine(np_data_train, optimal_trunc_lvl=opt_level)

        # Print Final Statistics
        order = joint_model.order
        ordered_names = [combined_u_train.columns[int(i) - 1] for i in order]
        print("")
        print(f"Top 5 Root Nodes (Market Hubs): {ordered_names[:5]}")
        print(f"In-Sample Log-Likelihood: {joint_model.loglik(np_data_train):.2f}")
        print(f"In-Sample AIC:            {joint_model.aic(np_data_train):.2f}")
        print(f"In-Sample BIC:            {joint_model.bic(np_data_train):.2f}")

        # Diagnostics & Save JSON
        plot_large_heatmap(combined_u_train, factor_name, os.path.join(graph_dir, f"{save_prefix}_heatmap.png"))
        plot_family_dist(joint_model, factor_name, os.path.join(graph_dir, f"{save_prefix}_families.png"))
        plot_simulated_vs_empirical(joint_model, combined_u_train, factor_name, os.path.join(graph_dir, f"{save_prefix}_simulated.png"))
        plot_tree1_network(combined_u_train, factor_name, os.path.join(graph_dir, f"{save_prefix}_tree1_network.png"))

        with open(os.path.join(out_dir, f"{save_prefix}_model.json"), "w") as f:
            f.write(joint_model.to_json())