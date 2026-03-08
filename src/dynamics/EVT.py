import numpy as np
from scipy.stats import genpareto, gaussian_kde
from scipy.interpolate import interp1d
import random
seed = 42
random.seed(seed)
np.random.seed(seed)

# Three-region semi-parametric Extreme Value Theory (GDP and KDE)
class EVT:
    def __init__(self):
        # Threshold values that separate lower tail, body, and upper tail
        self.u_lower = None 
        self.u_upper = None
        
        # GPD parameters for lower and upper tails (shape, location, scale)
        self.params_lower = None
        self.params_upper = None 
        
        # Gaussian KDE object fitted to the body region
        self.body_kde = None
        
        # Inverse interpolator for the body region (u -> z)
        self.body_interp = None
        
        # Tail probabilities (quantile levels used for thresholding)
        self.eta_l = None 
        self.eta_u = None
        
        # CDF values at body boundaries (used for normalization)
        self.body_min_cdf = 0.0
        self.body_max_cdf = 1.0 
        
        self._is_fitted = False

    # Fits GPD to tails and Gaussian KDE to the body.
    def fit(self, z, lower_quantile=0.10, upper_quantile=0.10):
        self.eta_l = lower_quantile
        self.eta_u = upper_quantile
        
        # Sort data for threshold identification
        sorted_z = np.sort(z)
        n = len(z)
        
        # 1. Determine Thresholds
        idx_lower = int(self.eta_l * n)
        idx_upper = int((1 - self.eta_u) * n)
        
        # Robust check: ensure indices don't cross and dataset is large enough
        if idx_lower >= idx_upper:
            raise ValueError("Tail thresholds overlap or dataset too small.")

        self.u_lower = sorted_z[idx_lower]
        self.u_upper = sorted_z[idx_upper]
        
        # 2. Fit Lower Tail (GPD)
        lower_data = sorted_z[sorted_z < self.u_lower]
        if len(lower_data) >= 10:
            excess_lower = self.u_lower - lower_data
            self.params_lower = genpareto.fit(excess_lower, floc=0)
        else:
            self.params_lower = None

        # 3. Fit Upper Tail (GPD)
        upper_data = sorted_z[sorted_z > self.u_upper]
        if len(upper_data) >= 10:
            excess_upper = upper_data - self.u_upper
            self.params_upper = genpareto.fit(excess_upper, floc=0)
        else:
            self.params_upper = None

        # 4. Fit Body (Gaussian KDE)
        mask_body = (sorted_z >= self.u_lower) & (sorted_z <= self.u_upper)
        body_data = sorted_z[mask_body]
        
        if len(body_data) > 2:
            self.body_kde = gaussian_kde(body_data)
            self.body_min_cdf = self.body_kde.integrate_box_1d(-np.inf, self.u_lower)
            self.body_max_cdf = self.body_kde.integrate_box_1d(-np.inf, self.u_upper)
            
            # Pre-compute inverse interpolation for the body to speed up simulation
            self._setup_body_interpolation()
        else:
            self.body_kde = None
            self.body_interp = None

        self._is_fitted = True
        return self

    def _setup_body_interpolation(self, n_points=1000):
        """
        Creates a lookup table (interpolator) for the body region.
        This maps u (probability) back to z (quantile) efficiently.
        """
        # Create a grid of z values within the body bounds
        z_grid = np.linspace(self.u_lower, self.u_upper, n_points)
        
        # Calculate raw KDE CDF values for this grid
        raw_cdf = np.array([self.body_kde.integrate_box_1d(-np.inf, x) for x in z_grid])
        
        # Normalize raw CDF to the target Uniform range [eta_l, 1 - eta_u]
        target_range = (1 - self.eta_u) - self.eta_l
        raw_range = self.body_max_cdf - self.body_min_cdf
        
        if raw_range > 1e-9:
            u_grid = self.eta_l + (raw_cdf - self.body_min_cdf) * (target_range / raw_range)
        else:
            u_grid = np.linspace(self.eta_l, 1 - self.eta_u, n_points)
            
        # Create interpolator: u -> z
        self.body_interp = interp1d(u_grid, z_grid, kind='linear', bounds_error=False, fill_value="extrapolate")

    def transform(self, z):
        """
        Probability Integral Transform (PIT): Maps z -> u in [0, 1]
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before transform.")
            
        u = np.zeros_like(z, dtype=float)
        
        # --- Lower Tail Transformation ---
        mask_l = z < self.u_lower
        
        if np.any(mask_l):
            if self.params_lower:
                # Extract GPD parameters
                xi, _, sigma = self.params_lower
                excess = self.u_lower - z[mask_l]
                cdf_gpd = genpareto.cdf(excess, xi, 0, sigma)
                
                # Map [0, 1] to [0, eta_l]
                u[mask_l] = self.eta_l * (1 - cdf_gpd)
            else:
                u[mask_l] = self.eta_l * 0.5 

        # --- Upper Tail Transformation ---
        mask_u = z > self.u_upper
        
        if np.any(mask_u):
            if self.params_upper:
                # Extract GPD parameters
                xi, _, sigma = self.params_upper
                excess = z[mask_u] - self.u_upper
                cdf_gpd = genpareto.cdf(excess, xi, 0, sigma)

                # Map GPD [0, 1] -> Uniform [1-eta_u, 1]
                u[mask_u] = (1 - self.eta_u) + self.eta_u * cdf_gpd
            else:
                u[mask_u] = 1.0 - (self.eta_u * 0.5)

        # --- Body Transformation (KDE) ---
        mask_b = (~mask_l) & (~mask_u)
        
        if np.any(mask_b) and self.body_kde:
            # Calculate CDF at each body point using the fitted KDE
            raw_cdf = np.array([self.body_kde.integrate_box_1d(-np.inf, x) for x in z[mask_b]])
            
            # Normalize raw_cdf from [min_cdf, max_cdf] to [eta_l, 1-eta_u]
            target_range = (1 - self.eta_u) - self.eta_l
            raw_range = self.body_max_cdf - self.body_min_cdf
            
            if raw_range > 1e-9:
                u[mask_b] = self.eta_l + (raw_cdf - self.body_min_cdf) * (target_range / raw_range)
            else:
                u[mask_b] = 0.5
                
        elif np.any(mask_b):
            u[mask_b] = 0.5

        return np.clip(u, 1e-6, 1-1e-6)

    def inverse_transform(self, u):
        """
        Inverse Probability Integral Transform (Inverse PIT): Maps u -> z
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before inverse_transform.")

        z = np.zeros_like(u, dtype=float)
        
        # --- Lower Tail (Inverse GPD) ---
        mask_l = u < self.eta_l
        if np.any(mask_l):
            if self.params_lower:
                xi, _, sigma = self.params_lower
                # GPD PPF logic:
                # u_scaled = u / eta_l  (prob within lower tail sector)
                # target_cdf = 1 - u_scaled (since lower tail CDF integrates from right to left in excess terms)
                # excess = GPD_PPF(target_cdf)
                val_gpd = genpareto.ppf(1 - u[mask_l]/self.eta_l, xi, 0, sigma)
                z[mask_l] = self.u_lower - val_gpd
            else:
                # Fallback to Normal
                from scipy.stats import norm
                z[mask_l] = norm.ppf(u[mask_l])

        # --- Upper Tail (Inverse GPD) ---
        mask_u = u > (1 - self.eta_u)
        if np.any(mask_u):
            if self.params_upper:
                xi, _, sigma = self.params_upper
                # GPD PPF logic:
                # u_scaled = (u - (1 - eta_u)) / eta_u
                # excess = GPD_PPF(u_scaled)
                p_gpd = (u[mask_u] - (1 - self.eta_u)) / self.eta_u
                val_gpd = genpareto.ppf(p_gpd, xi, 0, sigma)
                z[mask_u] = self.u_upper + val_gpd
            else:
                from scipy.stats import norm
                z[mask_u] = norm.ppf(u[mask_u])

        # --- Body (Inverse Interpolation) ---
        mask_b = (~mask_l) & (~mask_u)
        if np.any(mask_b):
            if self.body_interp:
                z[mask_b] = self.body_interp(u[mask_b])
            else:
                # Fallback if body fit failed (rare)
                from scipy.stats import norm
                z[mask_b] = norm.ppf(u[mask_b])
                
        return z
