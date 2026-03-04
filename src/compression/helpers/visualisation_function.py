import matplotlib.pyplot as plt
import numpy as np
import config.settings as g
import seaborn as sns
import pandas as pd
import matplotlib.colors as colors
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm
import matplotlib.ticker as ticker
from pathlib import Path

def plot_asset_pc_heatmap(partial_r2, use_log=False):

    # Create DataFrame for better labeling
    df = pd.DataFrame(
        partial_r2, 
        index=g.SYMBOLS, 
        columns=[f"PC{i+1}" for i in range(partial_r2.shape[1])]
    )

    plt.figure(figsize=(16, 8))
    
    # Selection of normalization to handle the dominance of PC1
    norm = colors.LogNorm(vmin=1e-5, vmax=1) if use_log else None
    fmt = ".2f"
    
    sns.heatmap(df, 
                annot=True, 
                fmt=fmt, 
                cmap="viridis", 
                norm=norm,
                cbar_kws={'label': 'Explained Variance Ratio'})

    #plt.title(f"Metric A: Explained Variance per Asset per PC {'(Log Scale)' if use_log else ''}")
    plt.xlabel("Principal Components")
    #plt.ylabel("Assets")
    plt.tight_layout()
    plt.show()


def plot_score_evolution(fpca_scores, n_comps=3):
    """
    Plots the time series of PC scores for each asset.
    fpca_scores: np.array of shape (N_Samples * N_Assets, K_Components)
    """
    # Reshape scores back to (N_Samples, N_Assets, K_Components)
    reshaped_scores = fpca_scores.reshape(g.N_OBS, g.N_ASSETS, -1)
    
    fig, axes = plt.subplots(n_comps, 1, figsize=(12, 4 * n_comps), sharex=True)
    if n_comps == 1: axes = [axes]
    
    times = g.DATES # Or your quote_datetime index
    
    for i in range(n_comps):
        for j, symbol in enumerate(g.SYMBOLS):
            axes[i].plot(times, reshaped_scores[:, j, i], label=symbol, alpha=0.7)
        
        axes[i].set_title(f"Evolution of PC {i+1} Scores")
        axes[i].set_ylabel("Score Weight")
        if i == 0:
            axes[i].legend(loc='upper right', ncol=3, fontsize='small')
            
    plt.xlabel("Time Steps")
    plt.tight_layout()
    plt.show()


def plot_factor_diagnostics(surfaces, scores_3d, assets, dates, 
                            k_grid, t_grid, 
                            title_prefix="Factor", 
                            labels=None):

    if labels is None:
        labels = [f"{title_prefix} 1", f"{title_prefix} 2", f"{title_prefix} 3"]
        
    # --- 1. Data Preparation (Flattening Scores) ---
    N_DATES = len(dates)
    N_ASSETS = len(assets)
    
    # Transpose to (Assets, Dates, 3) so that when we reshape, 
    # we get all dates for Asset 1, then all dates for Asset 2, etc.
    scores_ordered = scores_3d.transpose(1, 0, 2).reshape(-1, 3)
    
    asset_col = np.repeat(assets, N_DATES)
    date_col = np.tile(dates, N_ASSETS)
    
    df = pd.DataFrame({
        'Asset': asset_col,
        'Date': date_col,
        'F1': scores_ordered[:, 0],
        'F2': scores_ordered[:, 1],
        'F3': scores_ordered[:, 2]
    })
    
    # --- 2. Plot Setup ---
    fig = plt.figure(figsize=(18, 12))
    colors = plt.cm.tab20.colors
    M, T = np.meshgrid(k_grid, t_grid)
    
    # Calculate global z-limits for consistent colorbar across surfaces
    z_min = surfaces.min()
    z_max = surfaces.max()
    z_limit = max(abs(z_min), abs(z_max))
    
    # --- Row 1: Surfaces ---
    for i in range(3):
        ax = fig.add_subplot(3, 3, i + 1, projection="3d")
        surf = ax.plot_surface(M, T, surfaces[i], cmap="RdBu_r", 
                               vmin=-z_limit, vmax=z_limit, alpha=0.85)
        ax.set_title(f"{labels[i]} Surface")
        ax.set_xlabel("Moneyness")
        ax.set_ylabel("Maturity")
        ax.set_zlim(-z_limit, z_limit)
        
        # Add colorbar only to the last surface to save space
        if i == 2:
            fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1)

    # --- Row 2: Profiles (Avg over T and Avg over K) ---
    
    # Moneyness Profile (Averaged over Maturity)
    ax4 = fig.add_subplot(3, 3, 4)
    for i in range(3):
        ax4.plot(k_grid, surfaces[i].mean(axis=0), lw=2, label=labels[i])
    ax4.set_title("Moneyness Profiles (Avg over Maturity)")
    ax4.set_xlabel("Moneyness")
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Maturity Profile (Averaged over Moneyness)
    ax5 = fig.add_subplot(3, 3, 5)
    for i in range(3):
        ax5.plot(t_grid, surfaces[i].mean(axis=1), lw=2, label=labels[i])
    ax5.set_title("Maturity Profiles (Avg over Moneyness)")
    ax5.set_xlabel("Maturity")
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    # Legend Holder (Empty slot used for legend)
    axLegend = fig.add_subplot(3, 3, 6)
    axLegend.axis('off')

    # --- Row 3: Scatter Plots (Pairwise Scores) ---
    
    # Helper for scatters
    def add_scatter(ax, x_col, y_col, x_lbl, y_lbl):
        for idx, asset in enumerate(assets):
            subset = df[df['Asset'] == asset]
            ax.scatter(subset[x_col], subset[y_col], 
                       s=10, alpha=0.6, 
                       color=colors[idx % len(colors)], label=asset)
        ax.set_xlabel(x_lbl)
        ax.set_ylabel(y_lbl)
        ax.grid(True, alpha=0.3)

    # 1 vs 2
    ax7 = fig.add_subplot(3, 3, 7)
    add_scatter(ax7, 'F1', 'F2', labels[0], labels[1])
    ax7.set_title(f"{labels[0]} vs {labels[1]}")

    # 1 vs 3
    ax8 = fig.add_subplot(3, 3, 8)
    add_scatter(ax8, 'F1', 'F3', labels[0], labels[2])
    ax8.set_title(f"{labels[0]} vs {labels[2]}")
    
    # 2 vs 3
    ax9 = fig.add_subplot(3, 3, 9)
    add_scatter(ax9, 'F2', 'F3', labels[1], labels[2])
    ax9.set_title(f"{labels[1]} vs {labels[2]}")

    # --- Final Legend ---
    # Grab handles from one of the scatter plots
    handles, legend_lbls = ax7.get_legend_handles_labels()
    axLegend.legend(handles, legend_lbls, loc='center', title="Assets", 
                    ncol=2, frameon=False, fontsize='small')

    plt.suptitle(f"{title_prefix} Analysis", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92) # Make room for suptitle
    plt.show()

def plot_single_asset_scores(df, symbol):
    # Filter for the specific asset
    asset_df = df[df['Symbol'] == symbol].sort_values('Date')
    
    if asset_df.empty:
        print(f"Symbol '{symbol}' not found in the dataset.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    factors = ['Level', 'Skew', 'Curvature']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, factor in enumerate(factors):
        axes[i].plot(asset_df['Date'], asset_df[factor], color=colors[i], lw=1.5)
        axes[i].set_ylabel(f'{factor} Score')
        axes[i].set_title(f'{symbol} - Interpretable {factor} Evolution')
        axes[i].grid(alpha=0.3)
        
    plt.xlabel('Date')
    plt.tight_layout()
    plt.show()

def plot_surfaces_for_latex(data, folder_prefix="res/fpca_fit/global", color_multiplier=0.7):

    X, Y = np.meshgrid(g.K_GRID, g.T_GRID)

    z_min, z_max = data.min(), data.max()
    
    norm = TwoSlopeNorm(vmin=min(z_min, -1e-9)*color_multiplier,
                        vcenter=0,
                        vmax=max(z_max, 1e-9)*color_multiplier)

    ax_zmin = z_min - abs(z_min) * 0.05
    ax_zmax = z_max + abs(z_max) * 0.05

    plt.rcParams.update({'font.size': 22})

    for i in range(data.shape[0]):
        fig = plt.figure(figsize=(14, 10)) 
        ax = fig.add_subplot(111, projection='3d')


        if i == data.shape[0] - 1:
            fig.text(
                .101, 0.5, 'Volatility',
                rotation='vertical',
                va='center',
                fontsize=30,
            )
        else:


            fig.text(
                0.18, 0.5, 'Volatility',
                rotation='vertical',
                va='center',
                fontsize=30,
            )


        Z = data[i]
        
        surf = ax.plot_surface(X, Y, Z, 
                               cmap=cm.RdBu_r, 
                               norm=norm,
                               linewidth=0, 
                               antialiased=True,
                               alpha=0.8)
        

        ax.set_xlabel('Moneyness (k)', fontsize=26, labelpad=25)
        ax.set_ylabel('Maturity (T)', fontsize=26, labelpad=25)
        

        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.zaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        
        ax.tick_params(axis='both', which='major', labelsize=22, pad=10)

        ax.set_zlim(ax_zmin, ax_zmax)
        
        if i == data.shape[0] - 1:
            # Colorbar with larger ticks
            cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=12, pad=0.1)
            cbar.ax.tick_params(labelsize=22)
        
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.view_init(elev=30, azim=225) 
        folder_path = Path(folder_prefix)
        folder_path.mkdir(parents=True, exist_ok=True)
        filename = f"{folder_prefix}/PC{i+1}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=True)

        plt.close()
        plt.close(fig)
