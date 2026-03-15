import numpy as np
from scipy.stats import norm

def bs_price_vec(S, K, T, sigma, r, option_type):
    """Vectorized Black-Scholes Pricing."""
    S, sigma = np.asarray(S, dtype=float), np.asarray(sigma, dtype=float)
    price = np.zeros_like(S)
    mask = (T >= 1e-6) & (sigma >= 1e-8) & (S >= 1e-8)
    
    if not np.any(mask): return price
    
    sqrt_T = np.sqrt(T) if isinstance(T, float) else np.sqrt(T[mask])
    sig_mask = sigma[mask]
    
    d1 = (np.log(S[mask] / K) + (r + 0.5 * sig_mask**2) * T) / (sig_mask * sqrt_T)
    d2 = d1 - sig_mask * sqrt_T
    
    if option_type == 'C':
        val = S[mask] * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    else:
        val = K * np.exp(-r*T) * norm.cdf(-d2) - S[mask] * norm.cdf(-d1)
        
    price[mask] = np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
    return price

def bs_delta_vec(S, K, T, sigma, r, option_type):
    """Vectorized Black-Scholes Delta."""
    S, sigma = np.asarray(S, dtype=float), np.asarray(sigma, dtype=float)
    delta = np.zeros_like(S)
    mask = (T >= 1e-6) & (sigma >= 1e-8) & (S >= 1e-8)
    
    if not np.any(mask): return delta
    
    d1 = (np.log(S[mask] / K) + (r + 0.5 * sigma[mask]**2) * T) / (sigma[mask] * np.sqrt(T))
    val = norm.cdf(d1) if option_type == 'C' else norm.cdf(d1) - 1.0
    delta[mask] = np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
    return delta

def bilinear_interp_vec(grid, mat, mon, tau, kappa):
    """Fast Bilinear Interpolation for extracted IV grids."""
    tau, kappa = np.clip(tau, mat[0], mat[-1]), np.clip(kappa, mon[0], mon[-1])
    i1 = np.clip(np.searchsorted(mat, tau, side='right') - 1, 0, len(mat) - 2)
    j1 = np.clip(np.searchsorted(mon, kappa, side='right') - 1, 0, len(mon) - 2).astype(int)
    
    t = (tau - mat[i1]) / (mat[i1+1] - mat[i1] + 1e-12)
    u = (kappa - mon[j1]) / (mon[j1+1] - mon[j1] + 1e-12)
    b = np.arange(len(kappa))
    
    return ((1-t)*(1-u)*grid[b, i1, j1] + t*(1-u)*grid[b, i1+1, j1] + 
            (1-t)*u*grid[b, i1, j1+1] + t*u*grid[b, i1+1, j1+1])
