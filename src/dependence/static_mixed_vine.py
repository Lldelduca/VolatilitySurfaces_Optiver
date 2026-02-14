import pyvinecopulib as pv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
def copula_diagnostics(model, u_df, name="Static Mixed R-Vine", save=None):
    n_vars = u_df.shape[1]
    
    # Identify the 3 strongest pairs empirically (Bypassing the missing get_tree API)
    tau_matrix = u_df.corr(method='kendall')
    upper_tri = tau_matrix.where(np.triu(np.ones(tau_matrix.shape), k=1).astype(bool))
    sorted_pairs = upper_tri.unstack().dropna().abs().sort_values(ascending=False)
    
    top_pairs = []
    for i in range(min(3, len(sorted_pairs))):
        col1, col2 = sorted_pairs.index[i]
        top_pairs.append((col1, col2))

    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    fig.suptitle(f'Copula Diagnostics: {name}', fontsize=16, fontweight='bold')

    # 1. Tree 1 Structure
    ax0 = fig.add_subplot(gs[0, 0:2])
    try:
        model.plot(tree=[0], vars_names=u_df.columns.tolist())
        ax0.set_title("Primary Dependence Structure (Tree 1)")
    except Exception as e:
        ax0.text(0.5, 0.5, f"Vine Plotting not supported in this pyvinecopulib version.\n({e})", 
                 ha='center', va='center', fontsize=12)
        ax0.set_title("Primary Dependence Structure")
        ax0.axis('off')

    # 2. Family Distribution
    ax1 = fig.add_subplot(gs[0, 2])
    families = []
    for tree_cops in model.pair_copulas:
        for bicop in tree_cops:
            families.append(str(bicop.family).split('.')[-1].capitalize())
            
    pd.Series(families).value_counts().plot(kind='barh', ax=ax1, color='teal')
    ax1.set_title("Selected Copula Families")
    ax1.grid(axis='x', alpha=0.3)

    # 3. Kendall's Tau Heatmap
    ax2 = fig.add_subplot(gs[1, 0:2])
    sns.heatmap(tau_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax2)
    ax2.set_title("Empirical Kendall's Tau (Joint Dependencies)")

    # 4. Model Log-Likelihood & AIC Info
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.axis('off')
    
    # Calculate stats
    u_numpy = u_df.to_numpy()
    stats_text = (
        f"Variables: {model.dim}\n"
        f"Observations: {u_df.shape[0]}\n"
        f"Log-Likelihood: {model.loglik(u_numpy):.2f}\n"
        f"AIC: {model.aic(u_numpy):.2f}\n"
        f"BIC: {model.bic(u_numpy):.2f}\n"
        f"Thread Count: {os.cpu_count()-1 if os.cpu_count() else 1}"
    )
    ax3.text(0.1, 0.5, stats_text, fontsize=12, family='monospace', 
             bbox=dict(boxstyle='round', fc='aliceblue', alpha=0.5))
    ax3.set_title("Model Fit Statistics")

    # 5. Top 3 Pairwise Scatters
    for i, (c1, c2) in enumerate(top_pairs):
        ax = fig.add_subplot(gs[2, i])
        ax.scatter(u_df[c1], u_df[c2], s=2, alpha=0.4, color='darkblue')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(c1)
        ax.set_ylabel(c2)
        ax.set_title(f"Top Pair {i+1}: {c1} vs {c2}")
        ax.grid(alpha=0.2)

    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    return fig


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

    res_dir = os.path.join(project_root, "outputs", "dynamics")
    out_dir = os.path.join(project_root, "outputs", "copulas")
    os.makedirs(out_dir, exist_ok=True)
    
    u_spot_file = os.path.join(res_dir, "uniforms_ngarch_t.csv")
    u_spot = pd.read_csv(u_spot_file, index_col=0)
    u_spot.index = pd.to_datetime(u_spot.index).date

    # Global Valid Dates based on NSDE
    u_har_path = os.path.join(res_dir, "uniforms_har_garch_evt.csv")
    u_nsde_path = os.path.join(res_dir, "nsde_uniforms.csv")

    if os.path.exists(u_har_path) and os.path.exists(u_nsde_path):
        u_har_ref = pd.read_csv(u_har_path, index_col=0)
        u_har_ref.index = pd.to_datetime(u_har_ref.index).date
        
        u_nsde_ref = pd.read_csv(u_nsde_path, index_col=False)
        
        valid_har_dates = u_spot.index.intersection(u_har_ref.index)
        global_valid_dates = valid_har_dates[-len(u_nsde_ref):]
        
        print(f"Enforcing strict comparability. Models will be evaluated on {len(global_valid_dates)} overlapping dates.")
        print(f"Evaluation Period: {global_valid_dates[0]} to {global_valid_dates[-1]}")
    else:
        print("Missing required files to establish global dates.")
        exit()

    # Fit Models 
    factor_sets = {
        "HAR-GARCH-EVT": "uniforms_har_garch_evt.csv",
        "NSDE": "nsde_uniforms.csv"
    }

    for factor_name, file_name in factor_sets.items():
        u_factor_path = os.path.join(res_dir, file_name)
        
        if os.path.exists(u_factor_path):
            print(f"\n--- Fitting Joint Copula: Spot (NGARCH-t) + Factors ({factor_name}) ---")
            
            if factor_name == "NSDE":
                u_factors = pd.read_csv(u_factor_path, index_col=False) 
                u_factors.index = global_valid_dates
            else:
                u_factors = pd.read_csv(u_factor_path, index_col=0)
                u_factors.index = pd.to_datetime(u_factors.index).date

            combined_u = pd.concat([u_spot, u_factors], axis=1, join='inner').dropna()
            combined_u = combined_u.loc[global_valid_dates] 

            if combined_u.empty:
                print(f"ERROR: The combined dataset is empty! Could not align dates.")
                continue
            
            joint_model = fit_static_mixed_vine(combined_u.to_numpy())
            print(joint_model.structure)
            print(f"Variable Order: {joint_model.order}")
            
            save_name = f"joint_vine_spot_{factor_name.lower().replace('-', '_')}"
            diag_path = os.path.join(out_dir, f"{save_name}_diag.png")
            
            copula_diagnostics(
                joint_model, 
                combined_u, 
                name=f"Spot + {factor_name} Joint Vine", 
                save=diag_path
            )

            joint_model.save(os.path.join(out_dir, f"{save_name}.json"))
    
        else:
            print(f"Warning: {file_name} not found. Skipping {factor_name} loop.")
