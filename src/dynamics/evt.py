import numpy as np
from scipy.stats import genpareto, gaussian_kde
from scipy.interpolate import interp1d

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
        else:
            self.body_kde = None

        self._is_fitted = True
        return self

    def transform(self, z):
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
            # Formula: Scaled = Target_Min + (Raw - Raw_Min) * (Target_Range / Raw_Range)
            
            target_range = (1 - self.eta_u) - self.eta_l
            raw_range = self.body_max_cdf - self.body_min_cdf
            
            if raw_range > 1e-9:
                u[mask_b] = self.eta_l + (raw_cdf - self.body_min_cdf) * (target_range / raw_range)
            else:
                u[mask_b] = 0.5
                
        elif np.any(mask_b):
            u[mask_b] = 0.5

        return np.clip(u, 1e-6, 1-1e-6)
