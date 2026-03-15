import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution, brentq
from scipy.interpolate import interp1d
from sklearn.isotonic import IsotonicRegression
from scipy.stats import norm
import warnings

# Suppress optimization warnings during mass-fitting
warnings.filterwarnings("ignore", category=RuntimeWarning)

class SSVI:
    """
    Surface Stochastic Volatility Inspired (SSVI) Model.
    Implements the Power-Law parameterization by Gatheral & Jacquier (2014)
    with strict enforcement of Butterfly and Calendar no-arbitrage constraints.
    """
    def __init__(self, df, symbol, quote_datetime):
        self.symbol = symbol
        self.quote_datetime = pd.to_datetime(quote_datetime)
        
        # Filter data for the specific asset and day
        df_filtered = df[(df['underlying_symbol'] == symbol) & 
                         (pd.to_datetime(df['quote_datetime']) == self.quote_datetime)].copy()
        
        if df_filtered.empty:
            raise ValueError(f"No data for {symbol} on {quote_datetime}")
            
        self.df = df_filtered.reset_index(drop=True)
        self.res = None
        self._surface = None

    # --- 1. CORE SSVI MATHEMATICS ---
    def _phi_power_law(self, theta, eta, gamma):
        """Power-law specification for the phi function."""
        theta_safe = np.maximum(theta, 1e-8)
        return eta / (np.power(theta_safe, gamma) * np.power(1 + theta_safe, 1 - gamma))
    
    def _ssvi_surface(self, k, theta, rho, phi):
        """Total variance w(k, theta) formulation."""
        p = phi * k
        inner = np.maximum(np.square(p + rho) + (1 - rho**2), 0)
        return (theta / 2.0) * (1 + rho * p + np.sqrt(inner))

    # --- 2. NO-ARBITRAGE CONSTRAINTS ---
    def _no_butterfly_constraint(self, theta, params):
        """Gatheral-Jacquier condition to prevent Butterfly arbitrage."""
        rho, eta, gamma = params
        phi = self._phi_power_law(theta, eta, gamma)
        term1 = theta * phi * (1.0 + abs(rho))
        term2 = theta * phi**2 * (1.0 + abs(rho))
        return 4.0 - max(term1, term2)

    def _no_calendar_shape_constraint(self, params):
        """Gatheral-Jacquier condition to prevent Calendar arbitrage."""
        rho, eta, gamma = params
        if abs(rho) >= 1.0: return -1.0 # Invalid correlation
        return (1.0 + np.sqrt(1.0 - rho**2)) - rho**2 * (1.0 - gamma)

    def _objective_smile(self, params, theta, k, T, iv_mkt):
        """Loss function for the SLSQP optimizer."""
        rho, eta, gamma = params
        phi = self._phi_power_law(theta, eta, gamma)
        w_model = self._ssvi_surface(k, theta, rho, phi)
        iv_model = np.sqrt(np.maximum(w_model, 1e-9) / T)
        return np.sum((iv_model - iv_mkt)**2) * 1e4 # Scale up for optimizer sensitivity

    # --- 3. CALIBRATION ENGINE ---
    def fit(self, max_iter=10000):
        """Fits the SSVI surface slice-by-slice across maturities."""
        k_all = self.df['log_moneyness'].values
        T_all = self.df['tau'].values
        iv_all = self.df['implied_volatility'].values
        
        unique_T = np.sort(self.df['tau'].unique())
        
        theta_map, rho_map, eta_map, gamma_map = {}, {}, {}, {}
        last_params = np.array([-0.5, 0.5, 0.5]) # Initial guess [rho, eta, gamma]

        for i, T in enumerate(unique_T):
            mask = T_all == T
            k = k_all[mask]

            # Require at least 5 strikes to fit a reliable smile
            if len(k) > 5:
                iv_mkt = iv_all[mask]
                w_mkt_slice = (iv_mkt**2) * T
                
                # Interpolate ATM variance (theta)
                if (k <= 0).any() and (k > 0).any():
                    theta = np.interp(0.0, k, w_mkt_slice)
                else:
                    theta = w_mkt_slice[np.argmin(np.abs(k))]

                bounds = [(-0.999, 0.999), (1e-6, 4.0), (0.0, 1.0)]
                constraints = [
                    {"type": "ineq", "fun": lambda p, t=theta: self._no_butterfly_constraint(t, p)},
                    {"type": "ineq", "fun": lambda p: self._no_calendar_shape_constraint(p)}
                ]

                # Use Differential Evolution for the first slice to avoid local minima
                if i == 0:
                    res = differential_evolution(self._objective_smile, bounds=bounds, args=(theta, k, T, iv_mkt))
                else:
                    res = minimize(
                        self._objective_smile, last_params, args=(theta, k, T, iv_mkt),
                        method="SLSQP", bounds=bounds, constraints=constraints,
                        options={"ftol": 1e-12, "maxiter": max_iter}
                    )

                if res.success and abs(res.x[0]) < 0.99:
                    last_params = res.x
                    theta_map[T] = theta
                    rho_map[T], eta_map[T], gamma_map[T] = res.x

        if not theta_map:
            raise ValueError(f"SSVI Fit failed entirely for {self.symbol} on {self.quote_datetime}")

        # Enforce strict Monotonicity of ATM Variance (No Calendar Arbitrage)
        Ts = np.array(sorted(theta_map.keys()))
        thetas = np.array([theta_map[T] for T in Ts])
        iso = IsotonicRegression(increasing=True)
        thetas_mono = iso.fit_transform(Ts, thetas)
        theta_map = {T: thetas_mono[i] for i, T in enumerate(Ts)}

        self.res = {"theta": theta_map, "rho": rho_map, "eta": eta_map, "gamma": gamma_map, "maturities": Ts}
        self._compile_surface()
        return self

    def _compile_surface(self):
        """Pre-computes grids and interpolators for ultra-fast pricing."""
        Ts = self.res["maturities"]
        self._surface = {
            "Ts": Ts,
            "theta": np.array([self.res["theta"][T] for T in Ts]),
            "rho": np.array([self.res["rho"][T] for T in Ts]),
            "eta": np.array([self.res["eta"][T] for T in Ts]),
            "gamma": np.array([self.res["gamma"][T] for T in Ts])
        }
        self._surface["sqrt_theta"] = np.sqrt(self._surface["theta"])
        
        # Forward Price Interpolator
        F_map = self.df.groupby("tau")["underlying_price"].mean().loc[Ts].values # Ensure underlying_price maps correctly
        self._surface["F"] = F_map
        self._forward_interp = interp1d(Ts, F_map, kind="linear", fill_value="extrapolate")

        self._surface["phi"] = np.array([
            self._phi_power_law(self._surface["theta"][i], self._surface["eta"][i], self._surface["gamma"][i])
            for i in range(len(Ts))
        ])

    # --- 4. EXPOSED PRICING & VOLATILITY METHODS ---
    def total_variance(self, T, k):
        """Calculates total variance w(T, k) using Lemma 5.1 linear interpolation."""
        s = self._surface
        Ts = s["Ts"]

        if T in Ts:
            idx = np.where(Ts == T)[0][0]
            return self._ssvi_surface(k, s["theta"][idx], s["rho"][idx], s["phi"][idx])

        i = np.searchsorted(Ts, T)
        if i == 0 or i == len(Ts):
            idx = 0 if i == 0 else -1
            return self._ssvi_surface(k, s["theta"][idx], s["rho"][idx], s["phi"][idx])

        # Linear interpolation in variance space
        T1, T2 = Ts[i - 1], Ts[i]
        theta_t = np.interp(T, [T1, T2], [s["theta"][i - 1], s["theta"][i]])

        denom = s["sqrt_theta"][i] - s["sqrt_theta"][i - 1]
        alpha = (T2 - T) / (T2 - T1) if abs(denom) < 1e-10 else (s["sqrt_theta"][i] - np.sqrt(theta_t)) / denom

        def get_slice_price(idx):
            F = s["F"][idx]
            K = F * np.exp(k)
            w = self._ssvi_surface(k, s["theta"][idx], s["rho"][idx], s["phi"][idx])
            
            # Internal BS Call
            vol_sqrt_t = np.sqrt(np.maximum(w, 1e-12))
            d1 = (np.log(F / K) + 0.5 * w) / vol_sqrt_t
            d2 = d1 - vol_sqrt_t
            return F * norm.cdf(d1) - K * norm.cdf(d2), K

        C1, K1 = get_slice_price(i - 1)
        C2, K2 = get_slice_price(i)
        
        F_t = float(self._forward_interp(T))
        K_t = F_t * np.exp(k)
        Ct = np.clip(K_t * (alpha * (C1 / K1) + (1 - alpha) * (C2 / K2)), max(F_t - K_t, 0.0) + 1e-9, F_t - 1e-9)
        
        def obj(w):
            vol_sqrt_t = np.sqrt(np.maximum(w, 1e-12))
            d1 = (np.log(F_t / K_t) + 0.5 * w) / vol_sqrt_t
            d2 = d1 - vol_sqrt_t
            return (F_t * norm.cdf(d1) - K_t * norm.cdf(d2)) - Ct
        
        return brentq(obj, 1e-9, 5.0) 

    def get_iv(self, K, T, S_current, r=0.0):
        """Exposed method: Returns Interpolated Implied Volatility."""
        F = S_current * np.exp(r * T)
        if F <= 1e-8 or K <= 1e-8 or T < 1e-6: return 0.0
        k = np.log(K / F)
        w = self.total_variance(T, k)
        return np.sqrt(max(w, 0.0) / T)
