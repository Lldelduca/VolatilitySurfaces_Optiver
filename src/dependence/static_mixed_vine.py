import pyvinecopulib as pv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os
import config.settings as g
import random

# =============================================================================
# --- Model Fitting ---
# =============================================================================

def fit_static_mixed_vine(u_data, optimal_trunc_lvl):
    T, N = u_data.shape
    controls = pv.FitControlsVinecop(
        family_set=[
            pv.BicopFamily.indep, 
            pv.BicopFamily.gaussian, 
            pv.BicopFamily.student,
            pv.BicopFamily.frank, 
            pv.BicopFamily.clayton, 
            pv.BicopFamily.gumbel,
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

# =============================================================================
# --- Visualizations ---
# =============================================================================

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
    d = len(u_df.columns)
    cpp_seeds = np.random.randint(0, 100000, size=d).tolist()
    
    sim_u = model.simulate(n=len(u_df), seeds=cpp_seeds)
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
    out_dir = os.path.join(project_root, "results", "copulas", "static")
    graph_dir = os.path.join(project_root, "results", "copulas", "static", "plots")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)
    
    # --- LOAD FULL TRAIN DATA ---
    u_spot_file = os.path.join(res_dir, "NGARCH", "uniforms_ngarch_train.csv")
    u_har_file = os.path.join(res_dir, "HAR_GARCH", "uniforms_har_garch_evt_train.csv")
    u_nsde_file = os.path.join(res_dir, "NSDE", "uniforms_nsde_train.csv")

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
        print(f"\n{'='*50}")
        print(f"--- Fitting Final Joint Copula: Spot + {factor_name} ---")
        print(f"{'='*50}")

        combined_u_train = pd.concat([u_spot, u_factors], axis=1)
        np_data_train = combined_u_train.to_numpy()
        
        # Pull the optimal K from your config file
        if factor_name == "HAR-GARCH-EVT":
            opt_level = g.K_HAR_GARCH
        else:
            opt_level = g.K_NSDE

        save_prefix = f"joint_vine_spot_{factor_name.lower().replace('-', '_')}"

        print(f"Fitting parsimonious model truncated at Tree {opt_level}...")
        joint_model = fit_static_mixed_vine(np_data_train, optimal_trunc_lvl=opt_level)

        # --- Print Final Statistics ---
        order = joint_model.order
        ordered_names = [combined_u_train.columns[int(i) - 1] for i in order]
        print("")
        print(f"Top 5 Root Nodes (Market Hubs): {ordered_names[:5]}")
        print(f"In-Sample Log-Likelihood: {joint_model.loglik(np_data_train):.2f}")
        print(f"In-Sample AIC:            {joint_model.aic(np_data_train):.2f}")
        print(f"In-Sample BIC:            {joint_model.bic(np_data_train):.2f}")

        # --- Diagnostics & Plotting ---
        print("\nGenerating Diagnostic Plots...")
        plot_large_heatmap(combined_u_train, factor_name, os.path.join(graph_dir, f"{save_prefix}_heatmap.png"))
        plot_family_dist(joint_model, factor_name, os.path.join(graph_dir, f"{save_prefix}_families.png"))
        plot_simulated_vs_empirical(joint_model, combined_u_train, factor_name, os.path.join(graph_dir, f"{save_prefix}_simulated.png"))
        plot_tree1_network(combined_u_train, factor_name, os.path.join(graph_dir, f"{save_prefix}_tree1_network.png"))

        # --- Save Final JSON for GAS Model ---
        json_path = os.path.join(out_dir, f"{save_prefix}_model.json")
        with open(json_path, "w") as f:
            f.write(joint_model.to_json())
