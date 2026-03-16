import numpy as np
from scipy.stats import norm

def bs_price_vec(S, K, T, sigma, r, option_type):
    """Vectorized Black-Scholes Pricing."""
    S, sigma = np.asarray(S, dtype=float), np.asarray(sigma, dtype=float)
    price = np.zeros_like(S)
    mask = (T >= 1e-6) & (sigma >= 1e-8) & (S >= 1e-8)
    
    if not np.any(mask): return price
    
    # FIX: Safely check if T is a scalar using np.ndim
    T_mask = T if np.ndim(T) == 0 else T[mask]
    sqrt_T = np.sqrt(T_mask)
    sig_mask = sigma[mask]
    
    d1 = (np.log(S[mask] / K) + (r + 0.5 * sig_mask**2) * T_mask) / (sig_mask * sqrt_T)
    d2 = d1 - sig_mask * sqrt_T
    
    if option_type == 'C':
        val = S[mask] * norm.cdf(d1) - K * np.exp(-r * T_mask) * norm.cdf(d2)
    else:
        val = K * np.exp(-r * T_mask) * norm.cdf(-d2) - S[mask] * norm.cdf(-d1)
        
    price[mask] = np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
    return price

def bs_delta_vec(S, K, T, sigma, r, option_type):
    """Vectorized Black-Scholes Delta."""
    S, sigma = np.asarray(S, dtype=float), np.asarray(sigma, dtype=float)
    delta = np.zeros_like(S)
    mask = (T >= 1e-6) & (sigma >= 1e-8) & (S >= 1e-8)
    
    if not np.any(mask): return delta
    
    T_mask = T if np.ndim(T) == 0 else T[mask]
    sig_mask = sigma[mask]
    
    d1 = (np.log(S[mask] / K) + (r + 0.5 * sig_mask**2) * T_mask) / (sig_mask * np.sqrt(T_mask))
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

def vega_bs_vec(S, K, T, sigma, r):
    S, sigma = np.asarray(S, dtype=float), np.asarray(sigma, dtype=float)
    vega = np.zeros_like(S)
    mask = (T >= 1e-6) & (sigma >= 1e-8) & (S >= 1e-8)
    if not np.any(mask): return vega
    
    S_m, sig_m = S[mask], sigma[mask]
    sqrt_T = np.sqrt(T[mask] if isinstance(T, np.ndarray) else T)
    
    d1 = (np.log(S_m / K) + (r + 0.5 * sig_m**2) * T) / (sig_m * sqrt_T)
    val = S_m * norm.pdf(d1) * sqrt_T
    vega[mask] = np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
    return vega

def vanna_bs_vec(S, K, T, sigma, r):
    S, sigma = np.asarray(S, dtype=float), np.asarray(sigma, dtype=float)
    vanna = np.zeros_like(S)
    mask = (T >= 1e-6) & (sigma >= 1e-8) & (S >= 1e-8)
    if not np.any(mask): return vanna
    
    S_m, sig_m = S[mask], sigma[mask]
    sqrt_T = np.sqrt(T[mask] if isinstance(T, np.ndarray) else T)
    
    d1 = (np.log(S_m / K) + (r + 0.5 * sig_m**2) * T) / (sig_m * sqrt_T)
    d2 = d1 - sig_m * sqrt_T
    val = -norm.pdf(d1) * d2 / sig_m
    vanna[mask] = np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
    return vanna

def gamma_bs_vec(S, K, T, sigma, r):
    S, sigma = np.asarray(S, dtype=float), np.asarray(sigma, dtype=float)
    gamma = np.zeros_like(S)
    mask = (T >= 1e-6) & (sigma >= 1e-8) & (S >= 1e-8)
    if not np.any(mask): return gamma
    
    S_m, sig_m = S[mask], sigma[mask]
    sqrt_T = np.sqrt(T[mask] if isinstance(T, np.ndarray) else T)
    
    d1 = (np.log(S_m / K) + (r + 0.5 * sig_m**2) * T) / (sig_m * sqrt_T)
    val = norm.pdf(d1) / (S_m * sig_m * sqrt_T)
    gamma[mask] = np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
    return gamma

def bilinear_interp_vec(grid_3d, mat_arr, mon_arr, tau, kappa_vec):
    """Vectorized 3D bilinear interpolation across 10,000 scenarios."""
    N = len(kappa_vec)
    tau = float(np.clip(tau, mat_arr[0], mat_arr[-1]))
    kappa_vec = np.clip(kappa_vec, mon_arr[0], mon_arr[-1])
    
    i1 = int(np.clip(np.searchsorted(mat_arr, tau, side='right') - 1, 0, len(mat_arr)-2))
    i2 = i1 + 1
    
    j1 = np.searchsorted(mon_arr, kappa_vec, side='right') - 1
    j1 = np.clip(j1, 0, len(mon_arr)-2).astype(int)
    j2 = j1 + 1
    
    t = (tau - mat_arr[i1]) / (mat_arr[i2] - mat_arr[i1] + 1e-12)
    u = (kappa_vec - mon_arr[j1]) / (mon_arr[j2] - mon_arr[j1] + 1e-12)
    
    batch_idx = np.arange(N)
    return (1-t)*(1-u)*grid_3d[batch_idx, i1, j1] + \
           t*(1-u)*grid_3d[batch_idx, i2, j1] + \
           (1-t)*u*grid_3d[batch_idx, i1, j2] + \
           t*u*grid_3d[batch_idx, i2, j2]
