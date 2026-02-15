import pyvinecopulib as pv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import json
import os

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
            #pv.BicopFamily.bb1,      # Clayton-Gumbel Mix
            #pv.BicopFamily.bb7       # Joe-Clayton Mix
        ],
        selection_criterion="aic",
        trunc_lvl=N-1,               # Fit full tree
        tree_criterion="tau",        # Use correlation to order the tree 
        allow_rotations=True,        # Enables negative correlations
        num_threads=os.cpu_count()-1
    )

    model = pv.Vinecop(d=N)

    # Structure Selection & Parameter Fitting (The function finds the best tree structure, families, and parameters)
    model.select(u_data, controls=controls)
    
    return model

# Visualizations
def plot_tree1_network(u_df, name, save_path):
    # 1. Calculate empirical Kendall's Tau
    tau_matrix = u_df.corr(method='kendall')
    
    # 2. Create a NetworkX Graph
    G = nx.Graph()
    for col in tau_matrix.columns:
        G.add_node(col)
        
    # Add all edges with absolute tau as the weight (since Vines link absolute dependence first)
    for i in range(len(tau_matrix.columns)):
        for j in range(i + 1, len(tau_matrix.columns)):
            col1 = tau_matrix.columns[i]
            col2 = tau_matrix.columns[j]
            real_tau = tau_matrix.iloc[i, j]
            weight = abs(real_tau)
            G.add_edge(col1, col2, weight=weight, tau=real_tau)
            
    # 3. Find the Maximum Spanning Tree (This IS Tree 1 of the Vine)
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
    edge_widths = [d['weight'] * 6 for u, v, d in edges]  # Thicker = stronger dependence
    edge_colors = ['darkred' if d['tau'] < 0 else 'darkgreen' for u, v, d in edges]
    
    # Layout (Spring layout pulls strongly connected hubs to the center)
    pos = nx.spring_layout(mst, k=0.5, iterations=50, seed=42)
    
    # Draw Nodes, Edges, and Labels
    nx.draw_networkx_nodes(mst, pos, node_color=node_colors, node_size=2000, edgecolors='black', linewidths=1.5)
    nx.draw_networkx_edges(mst, pos, width=edge_widths, edge_color=edge_colors, alpha=0.7)
    nx.draw_networkx_labels(mst, pos, font_size=10, font_weight='bold')
    
    # Custom Legend
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
    """Simulates from the copula to verify goodness-of-fit on top pairs."""
    tau_matrix = u_df.corr(method='kendall').abs()
    np.fill_diagonal(tau_matrix.values, 0)
    
    top_pairs = []
    for _ in range(3):
        idx = tau_matrix.unstack().idxmax()
        top_pairs.append(idx)
        tau_matrix.loc[idx[0], idx[1]] = 0
        tau_matrix.loc[idx[1], idx[0]] = 0

    # Simulate from Copula
    sim_u = model.simulate(n=len(u_df), seeds=[])
    sim_df = pd.DataFrame(sim_u, columns=u_df.columns)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Goodness of Fit: Empirical vs Simulated ({name})", fontsize=18)
    
    for i, (var1, var2) in enumerate(top_pairs):
        # Empirical
        axes[0, i].scatter(u_df[var1], u_df[var2], s=5, alpha=0.4, color='darkblue')
        axes[0, i].set_title(f"Empirical: {var1} vs {var2}")
        axes[0, i].set_xlim(0, 1); axes[0, i].set_ylim(0, 1)
        axes[0, i].grid(alpha=0.2)
        
        # Simulated
        axes[1, i].scatter(sim_df[var1], sim_df[var2], s=5, alpha=0.4, color='darkred')
        axes[1, i].set_title(f"Simulated: {var1} vs {var2}")
        axes[1, i].set_xlim(0, 1); axes[1, i].set_ylim(0, 1)
        axes[1, i].grid(alpha=0.2)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

    res_dir = os.path.join(project_root, "outputs", "dynamics")
    out_dir = os.path.join(project_root, "outputs", "copulas")
    graph_dir = os.path.join(project_root, "outputs", "copulas", "plots", "static")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)
    
    u_spot_file = os.path.join(res_dir, "train_uniforms_ngarch_t.csv")
    u_har_file = os.path.join(res_dir, "train_uniforms_har_garch_evt.csv")
    u_nsde_file = os.path.join(res_dir, "train_nsde_uniforms.csv")

    u_spot = pd.read_csv(u_spot_file, index_col='Date', parse_dates=True)
    u_har = pd.read_csv(u_har_file, index_col='Date', parse_dates=True)
    u_nsde = pd.read_csv(u_nsde_file, index_col='Date', parse_dates=True)

    # Find common dates and truncate to ensure comparability
    global_valid_dates = u_spot.index.intersection(u_har.index).intersection(u_nsde.index)
    u_spot = u_spot.loc[global_valid_dates]
    u_har = u_har.loc[global_valid_dates]
    u_nsde = u_nsde.loc[global_valid_dates]
    print(f"Evaluation Period: {global_valid_dates[0].date()} to {global_valid_dates[-1].date()}")

    factor_sets = {"HAR-GARCH-EVT": u_har, "NSDE": u_nsde}

    for factor_name, u_factors in factor_sets.items():
        print("")
        print(f"--- Fitting Joint Copula: Spot + {factor_name} ---")

        combined_u = pd.concat([u_spot, u_factors], axis=1)
        joint_model = fit_static_mixed_vine(combined_u.to_numpy())

        # Print Statistics
        order = joint_model.order
        ordered_names = [combined_u.columns[int(i) - 1] for i in order]
        print("")
        print(f"Top 5 Root Nodes (Market Hubs): {ordered_names[:5]}")
        print(f"In-Sample Log-Likelihood: {joint_model.loglik(combined_u.to_numpy()):.2f}")
        print(f"In-Sample AIC:            {joint_model.aic(combined_u.to_numpy()):.2f}")
        print(f"In-Sample BIC:            {joint_model.bic(combined_u.to_numpy()):.2f}")

        # Diagnostics
        save_prefix = f"joint_vine_spot_{factor_name.lower().replace('-', '_')}"
        
        plot_large_heatmap(combined_u, factor_name, os.path.join(graph_dir, f"{save_prefix}_heatmap.png"))
        plot_family_dist(joint_model, factor_name, os.path.join(graph_dir, f"{save_prefix}_families.png"))
        plot_simulated_vs_empirical(joint_model, combined_u, factor_name, os.path.join(graph_dir, f"{save_prefix}_simulated.png"))
        plot_tree1_network(combined_u, factor_name, os.path.join(graph_dir, f"{save_prefix}_tree1_network.png"))

        # Save Model 
        with open(os.path.join(out_dir, f"{save_prefix}_model.json"), "w") as f:
            f.write(joint_model.to_json())
