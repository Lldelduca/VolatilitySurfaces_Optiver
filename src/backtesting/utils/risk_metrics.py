import numpy as np
from scipy.stats import norm, chi2

def kupiec_pof_test(hits, alpha=0.05):
    N, x = len(hits), int(np.sum(hits))
    if x == 0 or x == N: return 1.0
    p_hat = x / N
    lr = -2.0 * (x * np.log(alpha / p_hat) + (N - x) * np.log((1 - alpha) / (1 - p_hat)))
    return float(1 - chi2.cdf(lr, df=1))

def christoffersen_test(hits, alpha=0.05):
    hits = np.asarray(hits, dtype=int)
    N, x = len(hits), int(np.sum(hits))
    
    T00 = int(np.sum((hits[:-1] == 0) & (hits[1:] == 0)))
    T01 = int(np.sum((hits[:-1] == 0) & (hits[1:] == 1)))
    T10 = int(np.sum((hits[:-1] == 1) & (hits[1:] == 0)))
    T11 = int(np.sum((hits[:-1] == 1) & (hits[1:] == 1)))
    n0, n1 = T00 + T01, T10 + T11
    
    if x == 0 or x == N or n0 == 0 or n1 == 0: return 1.0, 1.0, 1.0
    
    pi_01, pi_11, pi_hat = np.clip(T01 / n0, 1e-9, 1 - 1e-9), np.clip(T11 / n1, 1e-9, 1 - 1e-9), np.clip(x / N, 1e-9, 1 - 1e-9)
    
    ll_h0 = T00 * np.log(1 - pi_hat) + T01 * np.log(pi_hat) + T10 * np.log(1 - pi_hat) + T11 * np.log(pi_hat)
    ll_h1 = T00 * np.log(1 - pi_01)  + T01 * np.log(pi_01) + T10 * np.log(1 - pi_11)  + T11 * np.log(pi_11)
    lr_ind = max(-2.0 * (ll_h0 - ll_h1), 0.0)
    
    ll_uc_h0 = x * np.log(alpha) + (N - x) * np.log(1 - alpha)
    ll_uc_h1 = x * np.log(pi_hat) + (N - x) * np.log(1 - pi_hat)
    lr_uc = max(-2.0 * (ll_uc_h0 - ll_uc_h1), 0.0)
    
    return float(1 - chi2.cdf(lr_uc + lr_ind, df=2)), float(1 - chi2.cdf(lr_ind, df=1)), float(1 - chi2.cdf(lr_uc, df=1))

def mcneil_frey_test(realized_pnl, es_forecast, pnl_std, hits, n_bootstrap=10000):
    x = np.asarray(realized_pnl)
    es = np.asarray(es_forecast)
    vol = np.asarray(pnl_std)
    h = np.asarray(hits, dtype=bool)

    if h.sum() < 5:
        return np.nan, np.nan, int(h.sum())

    residuals = (x[h] - es[h]) / (vol[h] + 1e-10)
    mf_stat = float(np.mean(residuals))
    n_hits = len(residuals)

    idx = np.random.randint(0, n_hits, size=(n_bootstrap, n_hits))
    boot_means = residuals[idx].mean(axis=1)
    
    centered_boot = boot_means - np.mean(boot_means)
    mf_pval = float(np.mean(np.abs(centered_boot) >= np.abs(mf_stat)))

    return mf_stat, mf_pval, n_hits

def diebold_mariano_test(loss_a, loss_b, h=1):
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    T = len(d)
    nw_var = np.var(d, ddof=1)
    for lag in range(1, h + 1): 
        nw_var += 2 * (1 - lag / (h + 1)) * np.cov(d[lag:], d[:-lag], ddof=1)[0, 1]
        
    dm_stat = d.mean() / np.sqrt(max(nw_var, 1e-12) / T)
    return float(dm_stat), float(2 * (1 - norm.cdf(np.abs(dm_stat)))), float(d.mean())

def tick_loss(pnl_real, var_forecast, alpha=0.05):
    e = pnl_real - var_forecast
    return np.where(e < 0, (1 - alpha) * np.abs(e), alpha * np.abs(e))