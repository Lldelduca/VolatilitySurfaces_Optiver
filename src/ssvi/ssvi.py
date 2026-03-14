# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import math
from sklearn.isotonic import IsotonicRegression
from scipy.stats import norm
from scipy.optimize import brentq
import plotly.graph_objects as go
from scipy.signal import savgol_filter
from scipy.optimize import differential_evolution

# %% Smile
class SSVI:

    def __init__(self, df, symbol, quote_datetime):
        self.symbol = symbol
        self.quote_datetime = pd.to_datetime(quote_datetime)

        df = df[df['underlying_symbol'] == symbol]

        df = df[df['quote_datetime'] == self.quote_datetime].copy()

        if df.empty:
            raise ValueError(f"No data for {symbol} on {quote_datetime}")

        self.df = df.reset_index(drop=True)
        # print(f"length df {len(self.df)} for day : {self.quote_datetime}")

    def __phi_power_law(self, theta, eta, gamma):

        theta_safe = np.maximum(theta, 1e-8)
        return eta / (np.power(theta_safe, gamma) * np.power(1 + theta_safe, 1 - gamma))
    
    def __ssvi_surface(self, k, theta, rho, phi):

        p = phi * k
        inner = np.maximum(np.square(p + rho) + (1 - rho**2), 0)

        return (theta / 2.0) * (1 + rho * p + np.sqrt(inner))

    def __no_butterfly_constraint(self, theta, params):
        """
        Combined butterfly arbitrage condition (Gatheral-Jacquier):
        max(
            theta * phi(theta) * (1 + |rho|),
            theta * phi(theta)^2 * (1 + |rho|)
        ) <= 4
        """
        rho, eta, gamma = params
        phi = self.__phi_power_law(theta, eta, gamma)

        term1 = theta * phi * (1.0 + abs(rho))
        term2 = theta * phi**2 * (1.0 + abs(rho))

        return 4.0 - max(term1, term2)

    def __no_calendar_shape_constraint(self, params):
        """
        Calendar spread arbitrage constraint (Gatheral-Jacquier):
        rho^2 (1 - gamma) <= 1 + sqrt(1 - rho^2)
        """
        rho, eta, gamma = params

        # avoid numerical issues close to |rho| = 1
        if abs(rho) >= 1.0:
            return -1.0

        return (1.0 + np.sqrt(1.0 - rho**2)) - rho**2 * (1.0 - gamma)

    def __objective_smile(self, params, theta, k, T, iv_mkt):
        rho, eta, gamma = params
        phi = self.__phi_power_law(theta, eta, gamma)
        w_model = self.__ssvi_surface(k, theta, rho, phi)
        
        
        iv_model = np.sqrt(np.maximum(w_model, 1e-9) / T)
        
        return np.sum((iv_model - iv_mkt)**2)*10e3

    def fit(self, max_iter_smile=10000, plot_diagnostics=False, smooth_params=False, save=None):
        k_all = self.df['log_moneyness'].values
        T_all = self.df['tau'].values
        iv_all = self.df['implied_volatility'].values # We use this directly now
        
        unique_T = np.sort(self.df['tau'].unique())
        
        theta_map, rho_map, eta_map, gamma_map = {}, {}, {}, {}
        last_params = np.array([-0.5, 0.5, 0.5])

        fitted_maturities = []

        for i, T in enumerate(unique_T):

            mask = T_all == T
            k = k_all[mask]

            if len(k)>5:
                iv_mkt = iv_all[mask]
                
                # Calculate theta (ATM variance) for the slice
                # We still need w_mkt here just to get the ATM intercept
                w_mkt_slice = (iv_mkt**2) * T
                if (k <= 0).any() and (k > 0).any():
                    theta = np.interp(0.0, k, w_mkt_slice)
                else:
                    theta = w_mkt_slice[np.argmin(np.abs(k))]

                bounds = [(-0.999, 0.999), (1e-6, 4.0), (0.0, 1.0)]
                constraints = [
                    {"type": "ineq", "fun": lambda p, theta=theta: self.__no_butterfly_constraint(theta, p)},
                    {"type": "ineq", "fun": lambda p: self.__no_calendar_shape_constraint(p)}
                ]

                if i == 0:

                    res = differential_evolution(
                        self.__objective_smile, 
                        bounds=bounds, 
                        args=(theta, k, T, iv_mkt)
                    )
                else:

                    res = minimize(
                        self.__objective_smile,
                        last_params,
                        args=(theta, k, T, iv_mkt), # Pass T and iv_mkt here
                        method="SLSQP", 
                        bounds=bounds, 
                        constraints=constraints,
                        options={"ftol": 1e-12, "maxiter": max_iter_smile}
                    )

                #if not res.success:

                #    res = differential_evolution(
                #        self.__objective_smile, 
                #        bounds=bounds, 
                #        args=(theta, k, T, iv_mkt)
                #    )

                if res.success and abs(res.x[0])<.99:
                    last_params = res.x
                    theta_map[T] = theta
                    rho_map[T], eta_map[T], gamma_map[T] = res.x
                    fitted_maturities.append(T) 



        # No Calendar Arbitrage - Lema 5.1
        Ts = np.array(sorted(theta_map.keys()))
        thetas = np.array([theta_map[T] for T in Ts])
        iso = IsotonicRegression(increasing=True)
        thetas_mono = iso.fit_transform(Ts, thetas)
        theta_map = {T: thetas_mono[i] for i, T in enumerate(Ts)}

        self.res = {"theta": theta_map, "rho": rho_map, "eta": eta_map, "gamma": gamma_map, "maturities": Ts}
        
        Ts = np.array(sorted(rho_map.keys()))
        if len(Ts) > 4 and smooth_params:
            rhos = np.array([rho_map[t] for t in Ts])
            etas = np.array([eta_map[t] for t in Ts])
            gammas = np.array([gamma_map[t] for t in Ts])


            window = min(len(Ts) // 2 * 2 - 1, 5) 
            if window >= 3:
                rhos_smooth = savgol_filter(rhos, window, 2)
                etas_smooth = savgol_filter(etas, window, 2)
                gammas_smooth = savgol_filter(gammas, window, 2)

                for i, t in enumerate(Ts):
                    rho_map[t] = rhos_smooth[i]
                    eta_map[t] = etas_smooth[i]
                    gamma_map[t] = gammas_smooth[i]
        
        
        self._compile_surface()

        if plot_diagnostics or (save is not None):
            w_market_all = (iv_all ** 2) * T_all 
            self._run_diagnostics(fitted_maturities, T_all, k_all, w_market_all, theta_map, rho_map, eta_map, gamma_map, save, plot_diagnostics)

    def _compile_surface(self):
        Ts = self.res["maturities"]
        self._surface = {
            "Ts": Ts,
            "theta": np.array([self.res["theta"][T] for T in Ts]),
            "rho": np.array([self.res["rho"][T] for T in Ts]),
            "eta": np.array([self.res["eta"][T] for T in Ts]),
            "gamma": np.array([self.res["gamma"][T] for T in Ts])
        }
        self._surface["sqrt_theta"] = np.sqrt(self._surface["theta"])
        
        # Corba Forward
        F_map = self.df.groupby("tau")["underlying_mid_price"].mean().loc[Ts].values
        self._surface["F"] = F_map
        self._forward_interp = interp1d(Ts, F_map, kind="linear", fill_value="extrapolate")

        # Phi pre-calculat
        self._surface["phi"] = np.array([
            self.__phi_power_law(self._surface["theta"][i], self._surface["eta"][i], self._surface["gamma"][i])
            for i in range(len(Ts))
        ])

    def bs_call(self, F, K, T, w):
        """Black-Scholes Call Formula."""
        if w < 1e-12: 
            return max(F - K, 0.0)
        vol_sqrt_t = np.sqrt(w)
        d1 = (np.log(F / K) + 0.5 * w) / vol_sqrt_t
        d2 = d1 - vol_sqrt_t
        return F * norm.cdf(d1) - K * norm.cdf(d2)

    def total_variance(self, T, k):
        """
        Calculates the total variance w(T, k) free of arbitrage.
        Uses linear interpolation in prices according to Lemma 5.1 of Gatheral 2014.
        """
        s = self._surface
        Ts = s["Ts"]

        if T in Ts:
            idx = np.where(Ts == T)[0][0]
            return self.__ssvi_surface(k, s["theta"][idx], s["rho"][idx], s["phi"][idx])

        i = np.searchsorted(Ts, T)
        if i == 0 or i == len(Ts):
            idx = 0 if i == 0 else -1
            return self.__ssvi_surface(k, s["theta"][idx], s["rho"][idx], s["phi"][idx])

        T1, T2 = Ts[i - 1], Ts[i]
        theta_t = np.interp(T, [T1, T2], [s["theta"][i - 1], s["theta"][i]])

        denom = s["sqrt_theta"][i] - s["sqrt_theta"][i - 1]
        if abs(denom) < 1e-10:
            alpha = (T2 - T) / (T2 - T1)
        else:
            alpha = (s["sqrt_theta"][i] - np.sqrt(theta_t)) / denom

        def get_slice_info(idx):
            F = s["F"][idx]
            K = F * np.exp(k)
            w = self.__ssvi_surface(k, s["theta"][idx], s["rho"][idx], s["phi"][idx])
            return self.bs_call(F, K, Ts[idx], w), K

        C1, K1 = get_slice_info(i - 1)
        C2, K2 = get_slice_info(i)
        
        F_t = float(self._forward_interp(T))
        K_t = F_t * np.exp(k)
        
        Ct = K_t * (alpha * (C1 / K1) + (1 - alpha) * (C2 / K2))
        
        price_min = max(F_t - K_t, 0.0)
        Ct = np.clip(Ct, price_min + 1e-9, F_t - 1e-9)
        
        def obj(w):
            return self.bs_call(F_t, K_t, T, w) - Ct
        
        return brentq(obj, 1e-9, 5.0) 

    def implied_vol(self, T, k):
        w = self.total_variance(T, k)
        return np.sqrt(w / T)


    def _run_diagnostics(self, fitted_maturities, T_all, k_all, w_market_all, 
                     theta_map, rho_map, eta_map, gamma_map, 
                     save=None, show_ylabel=True, show_yticklabels=True):

        TITLE_SIZE = 30
        LABEL_SIZE = 18
        TICK_SIZE = 20
        LINE_WIDTH = 2.5
        MARKER_SIZE = 50

        # Selecció de fins a 6 maturitats repartides
        sorted_maturities = sorted(fitted_maturities)
        n_total = len(sorted_maturities)

        if n_total > 3:
            indices = np.linspace(0, n_total - 1, 3, dtype=int)
            selected_maturities = [sorted_maturities[i] for i in indices]
        else:
            selected_maturities = sorted_maturities

        n_plots = len(selected_maturities)

        # Graella 3 columnes x 2 files
        n_rows, n_cols = 3, 1
        fig, axes = plt.subplots(n_rows, n_cols, 
                                figsize=(10, 8), 
                                constrained_layout=True)

        axes = axes.flatten()

        for idx, T in enumerate(selected_maturities):
            ax = axes[idx]

            mask = T_all == T
            k_market = k_all[mask]
            w_market = w_market_all[mask]

            if len(k_market) > 0:
                k_min, k_max = k_market.min(), k_market.max()
                k_grid = np.linspace(k_min - 0.1, k_max + 0.1, 100)

                phi = self.__phi_power_law(theta_map[T], eta_map[T], gamma_map[T])
                w_model = self.__ssvi_surface(k_grid, theta_map[T], rho_map[T], phi)

                ax.scatter(
                    k_market, np.sqrt(w_market / T),
                    s=MARKER_SIZE,
                    alpha=0.7,
                    color='black',
                    marker='x',
                    label='Market'
                )

                ax.plot(
                    k_grid, np.sqrt(w_model / T),
                    color='red',
                    lw=LINE_WIDTH,
                    label='SSVI Fit'
                )

            # Títol individual per subfigura (sense negreta)
            ax.set_title(f"T = {T:.3f}", fontsize=TITLE_SIZE)

            ax.grid(True, alpha=0.3, linestyle='--')
            ax.tick_params(axis='both', labelsize=TICK_SIZE)

            if not show_yticklabels:
                ax.set_yticklabels([])

            if show_ylabel and idx % n_cols == 0:
                ax.set_ylabel('Implied Vol ($\\sigma_{BS}$)', fontsize=LABEL_SIZE)

            if idx >= (n_rows - 1) * n_cols:
                ax.set_xlabel('Log-Moneyness ($k$)', fontsize=LABEL_SIZE)

        # Eliminar eixos buits si hi ha menys de 6 plots
        for j in range(n_plots, n_rows * n_cols):
            fig.delaxes(axes[j])

        if save is not None:
            plt.savefig(save, dpi=300, bbox_inches='tight')
            print(f"Saved figure: {save}")
        plt.show()
        plt.close(fig)


    def __plot_simple_volatility_smile(self, df_smile):

        colors = ['blue' if t == 'C' else 'red' for t in df_smile['option_type']]

        plt.scatter(df_smile['log_moneyness'], df_smile['implied_volatility'],
                    c=colors, alpha=0.6, s=50, edgecolor='black', linewidth=0.5)

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Call'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Put')
        ]

        return legend_elements

    def __plot_volatility_smile_curve(self, df_smile):

        params = self.res['params']
        theta_map = self.res['theta_map']
            
        T = df_smile['tau'].iloc[0]

        closest_T = min(theta_map.keys(), key=lambda x: abs(x - T))
        theta = theta_map[closest_T]
            

        k_min, k_max = df_smile['log_moneyness'].min(), df_smile['log_moneyness'].max()
        k_grid = np.linspace(k_min - 0.1, k_max + 0.1, 200)
            

        phi = params['eta'] / (np.power(theta, params['gamma']) * np.power(1 + theta, 1 - params['gamma']))
        p = phi * k_grid
        w_model = np.maximum((theta / 2.0, 0) * (1 + params['rho'] * p + np.sqrt(np.square(p + params['rho']) + (1 - params['rho']**2))), 0.0)
        iv_model = np.sqrt(w_model / T)
        ssvi_line = (k_grid, iv_model)

        plt.plot(ssvi_line[0], ssvi_line[1], color='black', linewidth=2.5, 
                    label=f"SSVI Fit (rho:{params['rho']:.2f}, gamma:{params['gamma']:.2f})")

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Calls'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Puts'),
            Line2D([0], [0], color='black', lw=2, label='SSVI Curve')
        ]

        
        return legend_elements

    def plot_volatility_smile(self, expiration):

        if isinstance(expiration, list):
            expirations = sorted(expiration)
        else:
            expirations = [expiration]
            
        n_plots = len(expirations)
        if n_plots == 0:
            print("No expirations provided.")
            return

        # 2. Setup Grid
        ncols = min(n_plots, 3)
        nrows = math.ceil(n_plots / ncols)
        figsize = (6 * ncols, 5 * nrows)
        
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        
        if n_plots > 1:
            axes_flat = axes.flatten(order='F') 
        else:
            axes_flat = [axes]

        global_xlim = [float('inf'), float('-inf')]
        global_ylim = [float('inf'), float('-inf')]

        for i, ax in enumerate(axes_flat):
            if i < n_plots:
                current_exp = expirations[i]
                target_date = pd.to_datetime(current_exp)
                df_smile = self.df[self.df['expiration'] == target_date]

                plt.sca(ax)

                if df_smile.empty:
                    ax.text(0.5, 0.5, f"No data for\n{current_exp}", 
                            ha='center', va='center', transform=ax.transAxes)
                else:
                    if self.res is None:
                        legend_elements = self.__plot_simple_volatility_smile(df_smile)
                    else:
                        self.__plot_simple_volatility_smile(df_smile)
                        legend_elements = self.__plot_volatility_smile_curve(df_smile)
                    
                    if legend_elements:
                        ax.legend(handles=legend_elements)

                    curr_xlim = ax.get_xlim()
                    curr_ylim = ax.get_ylim()

                    global_xlim[0] = min(global_xlim[0], curr_xlim[0])
                    global_xlim[1] = max(global_xlim[1], curr_xlim[1])
                    global_ylim[0] = min(global_ylim[0], curr_ylim[0])
                    global_ylim[1] = max(global_ylim[1], curr_ylim[1])

                # Standard formatting
                ax.set_xlabel('log moneyness (log(K/F))')
                ax.set_ylabel('Implied Volatility')
                ax.set_title(f'Expiry: {current_exp}')
                ax.grid(True, alpha=0.3)
                ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
            else:
                ax.axis('off')

        
        for i, ax in enumerate(axes_flat):
            if i < n_plots:
                ax.set_xlim(global_xlim)
                ax.set_ylim(global_ylim)

        fig.suptitle(f'{self.symbol} Volatility Smile Analysis - Date: {self.quote_datetime}', y=1.02, fontsize=16)
        plt.tight_layout()
        plt.show()

    def plot_surface(
        self,
        k_range=(-0.5, 0.5),
        n_k=60,
        n_T=60,
        show_data=True,
        elev=25,
        azim=-60
    ):
        """
        Plot the SSVI total variance surface w(T, k).
        
        Parameters
        ----------
        k_range : tuple
            Min / max log-moneyness
        n_k : int
            Number of k grid points
        n_T : int
            Number of T grid points
        show_data : bool
            Overlay observed market points if True
        elev, azim : float
            View angles for 3D plot
        """

        if not hasattr(self, "_surface"):
            raise RuntimeError("Surface not compiled. Call _compile_surface() first.")

        # --- Build grid ---
        T_min, T_max = self._surface["Ts"][0], self._surface["Ts"][-1]
        T_grid = np.linspace(T_min, T_max, n_T)
        k_grid = np.linspace(k_range[0], k_range[1], n_k)

        TT, KK = np.meshgrid(T_grid, k_grid)

        W = np.zeros_like(TT)

        # --- Evaluate surface ---
        for i in range(n_T):
            T = T_grid[i]
            for j in range(n_k):
                W[j, i] = self.total_variance(T, KK[j, i])

        # --- Plot ---
        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(111, projection="3d")

        surf = ax.plot_surface(
            TT, KK, W,
            cmap="viridis",
            linewidth=0,
            antialiased=True,
            alpha=0.85
        )

        # --- Market data overlay ---
        if show_data:
            k_obs = self.df["log_moneyness"].values
            T_obs = self.df["tau"].values
            iv_obs = self.df["implied_volatility"].values
            w_obs = iv_obs**2 * T_obs

            ax.scatter(
                T_obs, k_obs, w_obs,
                color="red",
                s=10,
                alpha=0.6,
                label="Market data"
            )

        # --- Formatting ---
        ax.set_xlabel("Time to expiry T")
        ax.set_ylabel("Log-moneyness k")
        ax.set_zlabel("Total variance w")

        ax.view_init(elev=elev, azim=azim)

        fig.colorbar(surf, ax=ax, shrink=0.6, aspect=12, label="Total variance")

        if show_data:
            ax.legend()

        plt.tight_layout()
        plt.show()

    def plot_surface_plotly(self, k_range=(-0.5, 0.5), n_k=100, n_T=100, show_data=True):
        """Interactive Plotly surface of the SSVI total variance w(T,k)."""
        if not hasattr(self, "_surface"):
            raise RuntimeError("Surface not compiled. Call fit() first.")

        T_grid = np.linspace(self._surface["Ts"][0], self._surface["Ts"][-1], n_T)
        k_grid = np.linspace(k_range[0], k_range[1], n_k)

        # Use the vectorized method for speed
        W = self.total_variance_vectorized(T_grid, k_grid)

        fig = go.Figure()
        fig.add_trace(go.Surface(
            x=T_grid, y=k_grid, z=W,
            colorscale="Viridis", opacity=0.85, name="SSVI Surface"
        ))

        if show_data:
            k_obs = self.df["log_moneyness"].values
            T_obs = self.df["tau"].values
            w_obs = self.df["implied_volatility"].values**2 * T_obs
            fig.add_trace(go.Scatter3d(
                x=T_obs, y=k_obs, z=w_obs,
                mode="markers", marker=dict(size=2, color="red"), name="Market Data"
            ))

        fig.update_layout(
            title=f"",
            scene=dict(xaxis_title="T", yaxis_title="k", zaxis_title="w"),
            width=1000, height=800
        )
        fig.show()

    def evaluate_fit(self):
        results = {}
        total_rmse = 0
        
        Ts = self.res["maturities"]
        for T in Ts:

            mask = self.df['tau'] == T
            k = self.df[mask]['log_moneyness'].values
            iv_mkt = self.df[mask]['implied_volatility'].values
            
            w_model = np.array([self.total_variance(T, ki) for ki in k])
            iv_model = np.sqrt(w_model / T)
            
            rmse = np.sqrt(np.mean((iv_mkt - iv_model)**2))
            total_rmse += rmse
            
        results['avg_rmse'] = total_rmse / len(Ts)
        
        rhos = np.array([self.res["rho"][t] for t in Ts])
        results['rho_stability'] = np.std(np.diff(rhos))
        

        results['final_score'] = 1.0 / (results['avg_rmse'] + results['rho_stability'])
        results["quote_datetime"] = self.quote_datetime
        return results
    
    def bs_call_vectorized(self, F, K, T, w):
        """Vectorized Black-Scholes for surface generation."""
        w_safe = np.maximum(w, 1e-12)
        vol_sqrt_t = np.sqrt(w_safe)
        d1 = (np.log(F / K) + 0.5 * w_safe) / vol_sqrt_t
        d2 = d1 - vol_sqrt_t
        return F * norm.cdf(d1) - K * norm.cdf(d2)

    def total_variance_vectorized(self, T_grid, k_grid):
        """
        Calculates a 2D mesh of total variance w(T, k).
        Used by plotting functions for high-speed rendering.
        """
        s = self._surface
        Ts = s["Ts"]
        
        # Interpolate parameters across the T_grid
        theta_interp = interp1d(Ts, s["theta"], kind="linear", fill_value="extrapolate")
        rho_interp   = interp1d(Ts, s["rho"], kind="linear", fill_value="extrapolate")
        phi_interp   = interp1d(Ts, s["phi"], kind="linear", fill_value="extrapolate")

        TT, KK = np.meshgrid(T_grid, k_grid)
        theta_vals = theta_interp(TT)
        rho_vals   = rho_interp(TT)
        phi_vals   = phi_interp(TT)

        p = phi_vals * KK
        inner = np.square(p + rho_vals) + (1 - rho_vals**2)
        W = (theta_vals / 2.0) * (1 + rho_vals * p + np.sqrt(np.maximum(inner, 0.0)))
        return W

    def get_variance_grid(self, t_flat, k_flat):
        """
        Direct SSVI formula application for arbitrary 1D arrays of T and k.
        Useful for backtesting or bulk-calculating IV on a dataframe.
        """
        s = self._surface
        theta_t = np.interp(t_flat, s["Ts"], s["theta"])
        rho_t   = np.interp(t_flat, s["Ts"], s["rho"])
        phi_t   = np.interp(t_flat, s["Ts"], s["phi"])
        
        p = phi_t * k_flat
        inner = np.square(p + rho_t) + (1 - rho_t**2)
        return (theta_t / 2.0) * (1 + rho_t * p + np.sqrt(np.maximum(inner, 0.0)))
    
    def get_iv(self, K, T, S_current, r=0.0):
        """
        Returns IV for a specific Strike K and Expiry T,
        re-calculating moneyness based on the new Spot Price S_current.
        """
        F = S_current * np.exp(r * T)
        
        if F <= 1e-8 or K <= 1e-8: return 0.0
        k = np.log(K / F)
        
        w = self.total_variance(T, k)
        
        if T < 1e-6: return 0.0
        return np.sqrt(max(w, 0.0) / T)

    def get_iv_fast_vectorized(self, K_array, T_array, S_array, r=0.0):
        # Ensure minimum safe values
        S_safe = np.maximum(S_array, 1e-8)
        T_safe = np.maximum(T_array, 1e-5)
        K_safe = np.maximum(K_array, 1e-8)
        
        F = S_safe * np.exp(r * T_safe)
        k_log = np.log(K_safe / F)
        
        s = self._surface
        Ts = s["Ts"]
        
        # Fast 1D interpolation of parameters for all scenarios at once
        theta_t = np.interp(T_safe, Ts, s["theta"])
        rho_t   = np.interp(T_safe, Ts, s["rho"])
        phi_t   = np.interp(T_safe, Ts, s["phi"])
        
        # Vectorized SSVI core formula
        p = phi_t * k_log
        inner = np.square(p + rho_t) + (1.0 - rho_t**2)
        inner_safe = np.maximum(inner, 0.0) # Prevent sqrt of negative
        
        W = (theta_t / 2.0) * (1.0 + rho_t * p + np.sqrt(inner_safe))
        W_safe = np.maximum(W, 0.0)
        
        iv = np.sqrt(W_safe / T_safe)
        
        # Clean up extreme edge cases
        iv = np.where(T_array < 1e-5, 0.0, iv)
        return iv
    
    def update_surface_params(self, theta_new, rho_new, eta_new, gamma_new):
        """
        Updates internal parameters.
        """
        # 1. Enforce Basic Bounds (Prevent Crashes)
        self._surface["theta"] = np.maximum(theta_new, 1e-6)  # Variance must be positive
        self._surface["rho"]   = np.clip(rho_new, -0.999, 0.999) # Correlation must be < 1
        self._surface["eta"]   = np.maximum(eta_new, 1e-6)    # Curvature must be positive
        self._surface["gamma"] = np.clip(gamma_new, 0.0, 1.0) # Gamma usually in [0,1]
        
        # 2. Recalculate Dependents
        self._surface["sqrt_theta"] = np.sqrt(self._surface["theta"])
        
        # 3. Recalculate Phi
        self._surface["phi"] = np.array([
            self.__phi_power_law(t, e, g) 
            for t, e, g in zip(self._surface["theta"], self._surface["eta"], self._surface["gamma"])
        ])
        
# %% matplotlib widget

if __name__ == "__main__":
    from src.get_data import load_symbol_data, load_data_symbol_polaris

    symbol = "AAPL"
    quote_date = "2020-05-26 12:00:00"
    expiry_date = "2026-12-18 00:00:00"
    expiry_date_2 = "2025-04-17 00:00:00"

    df = load_symbol_data(symbol=symbol, splits={"2020-08-31 12:00:00": 4})
    df2 = load_data_symbol_polaris(symbol=symbol)
    # %% 
    ssvi = SSVI(df2, symbol=symbol, quote_datetime=quote_date)
    ssvi.fit(plot_diagnostics=True, smooth_params=False)
    ssvi.evaluate_fit()
    # ssvi.total_variance(0.5, 0.02)
    # ssvi.plot_surface(show_data=True, elev=40, azim=-45)
    # ssvi.plot_surface_plotly(show_data=True)

# %%