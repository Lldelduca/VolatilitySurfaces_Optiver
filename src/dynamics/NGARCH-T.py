import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import t as student_t
from scipy.special import gammaln
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import probplot, kstest, norm, uniform
import os
import pickle

from src.dynamics.HAR_GARCH import HAR_GARCH_EVT

class NGARCH_T:
    def __init__(self):
        self.params = None
        self.train_uniforms = None
        self.test_uniforms = None
        self.vol = None
        self.test_vol = None
        self.resids = None
        self.test_resids = None
        
    def _garch_vol(self, r, mu, omega, alpha, beta, theta):
        T = len(r)
        sig2 = np.zeros(T)
        sig2[0] = np.var(r)
        
        for t in range(1, T):
            eps = r[t-1] - mu
            prev_sig = np.sqrt(sig2[t-1]) if sig2[t-1] > 1e-12 else 1e-6
            z = eps / prev_sig
            sig2[t] = max(omega + alpha * ((z - theta)**2) * sig2[t-1] + beta * sig2[t-1], 1e-6)
        
        return np.sqrt(sig2)
    
    def _loglik(self, params, r):
        mu, omega, alpha, beta, theta, nu = params
        
        if omega <= 0 or alpha < 0 or beta < 0 or nu <= 2.01:
            return 1e10
        if alpha * (1 + theta**2) + beta >= 0.999:
            return 1e10
        
        sig = self._garch_vol(r, mu, omega, alpha, beta, theta)
        z = (r - mu) / sig
        
        try:
            ll = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * nu)
            ll = np.sum(ll - ((nu + 1) / 2) * np.log(1 + z**2 / nu) - np.log(sig))
            return -ll
        except:
            return 1e10
    
    def predict_volatility(self, returns):
        if self.params is None: raise ValueError("Model not fitted.")
        mu_unscaled, omega_unscaled, alpha, beta, theta, nu = self.params
        
        scale = 100.0
        r_scaled = returns * scale
        mu_scaled = mu_unscaled * scale
        omega_scaled = omega_unscaled * (scale**2)
        
        vol_scaled = self._garch_vol(r_scaled, mu_scaled, omega_scaled, alpha, beta, theta)
        return vol_scaled / scale

    def evaluate(self, test_returns, test_vol):
        mu, _, _, _, _, nu = self.params
        z = (test_returns - mu) / test_vol
        ll_const = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * nu)
        loglik = np.sum(ll_const - ((nu + 1) / 2) * np.log(1 + z**2 / nu) - np.log(test_vol))
        mse = np.mean(((test_returns - mu)**2 - test_vol**2)**2)
        return loglik, mse

    def fit_and_predict(self, returns, holdout_days=248):
        r_full = returns.values if isinstance(returns, pd.Series) else np.array(returns)
        idx_full = returns.index if isinstance(returns, pd.Series) else None

        r_train = r_full[:-holdout_days]
        idx_train = idx_full[:-holdout_days]
        idx_test = idx_full[-holdout_days:]

        # Scaling for optimizer stability
        scale = 100.0
        r_train_scaled = r_train * scale

        x0 = [np.mean(r_train_scaled), 0.05 * np.var(r_train_scaled), 0.05, 0.90, 0.5, 6.0]
        bounds = [(-10, 10), (1e-6, 10), (0.0, 0.5), (0.5, 0.99), (-5, 5), (2.1, 50)]

        res = minimize(self._loglik, x0, args=(r_train_scaled,), method='SLSQP', bounds=bounds, tol=1e-6)
        
        # Save state (Keeping them in the SCALED space internally for predict_volatility consistency, 
        # but reporting the unscaled version in self.params)
        mu_s, omega_s, alpha, beta, theta, nu = res.x
        self.params = [mu_s/scale, omega_s/(scale**2), alpha, beta, theta, nu]

        # --- THE FIX: Filter in the SCALED space, then downscale ---
        r_full_scaled = r_full * scale
        vol_full_scaled = self._garch_vol(r_full_scaled, mu_s, omega_s, alpha, beta, theta)
        
        # Downscale vol back to original return space
        vol_full = vol_full_scaled / scale
        
        # z-scores remain the same because the scale cancels out (r*100 / vol*100 = r/vol)
        mu_unscaled = mu_s / scale
        z_full = (r_full - mu_unscaled) / vol_full
        
        u_full = np.clip(student_t.cdf(z_full, df=nu), 1e-6, 1-1e-6)
        
        self.train_uniforms = pd.Series(u_full[:-holdout_days], index=idx_train)
        self.test_uniforms = pd.Series(u_full[-holdout_days:], index=idx_test)
        self.vol = vol_full[:-holdout_days]
        self.test_vol = vol_full[-holdout_days:]
        self.resids = z_full[:-holdout_days]
        self.test_resids = z_full[-holdout_days:]
        
        return self
    
    def diagnostics(self, name="Asset", is_test=False, save=None):
        # Swap between Train and Test data
        mu, omega, alpha, beta, theta, nu = self.params
        z = self.test_resids if is_test else self.resids
        u_series = self.test_uniforms if is_test else self.train_uniforms
        u = u_series.values
        vol_series = self.test_vol if is_test else self.vol
        
        phase = "Out-of-Sample (Test)" if is_test else "In-Sample (Train)"
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        fig.suptitle(f'NGARCH-t Diagnostics [{phase}]: {name}', fontsize=14, fontweight='bold')
        
        # 1. Realized Returns vs Predicted Volatility Bands
        ax0 = fig.add_subplot(gs[0, :])
        ax0.plot(z, lw=0.5, alpha=0.7)
        ax0.set_title(f'Residuals (Alpha={alpha:.3f}, Beta={beta:.3f}, Theta={theta:.3f})')
        
        # 2. Histogram
        ax1 = fig.add_subplot(gs[1, 0])
        ax1.hist(z, bins=50, density=True, alpha=0.6, color='steelblue', ec='black')
        x_range = np.linspace(z.min(), z.max(), 200)
        ax1.plot(x_range, student_t.pdf(x_range, df=nu), 'r-', lw=2, label=f't(ν={nu:.1f})')
        ax1.plot(x_range, norm.pdf(x_range), 'g--', lw=2, label='N(0,1)', alpha=0.7)
        ax1.set_title('Distribution')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 3. QQ plot (Student-t)
        ax2 = fig.add_subplot(gs[1, 1])
        probplot(z, dist=student_t, sparams=(nu,), plot=ax2)
        ax2.set_title(f'Q-Q Plot (ν={nu:.1f})')
        ax2.grid(alpha=0.3)
        
        # 4. ACF residuals
        ax3 = fig.add_subplot(gs[1, 2])
        plot_acf(z, lags=40, ax=ax3, alpha=0.05)
        ax3.set_title('ACF Residuals')
        ax3.grid(alpha=0.3)
        
        # 5. ACF squared residuals
        ax4 = fig.add_subplot(gs[2, 0])
        plot_acf(z**2, lags=40, ax=ax4, alpha=0.05)
        ax4.set_title('ACF Squared Residuals')
        ax4.grid(alpha=0.3)
        
        # 6. Uniform histogram (PIT)
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.hist(u, bins=50, density=True, alpha=0.6, color='purple', ec='black')
        ax5.axhline(1.0, color='r', ls='--', lw=2, label='Uniform')
        ks_stat, ks_p = kstest(u, 'uniform')
        ax5.text(0.05, 0.95, f'KS: {ks_stat:.3f}\np={ks_p:.3f}',
                transform=ax5.transAxes, va='top',
                bbox=dict(boxstyle='round', fc='wheat', alpha=0.5))
        ax5.set_title('Uniform Transform')
        ax5.legend()
        ax5.grid(alpha=0.3)
        
        # 7. QQ uniform
        ax6 = fig.add_subplot(gs[2, 2])
        probplot(u, dist=uniform, plot=ax6)
        ax6.set_title('Q-Q Uniform')
        ax6.grid(alpha=0.3)
        
        plt.subplots_adjust(top=0.93, bottom=0.07, left=0.07, right=0.93, hspace=0.5, wspace=0.3)
        if save:
            plt.savefig(save, dpi=300, bbox_inches='tight')
        return fig

def plot_test_prediction(test_returns, test_vol, mu, nu, asset_name, save_path=None):
    plt.figure(figsize=(12, 5))
    
    # Calculate the 95% CI bounds using the Student-t inverse CDF
    # Student-t has fatter tails than Normal, so q will be > 1.96
    t_quantile = student_t.ppf(0.975, df=nu)
    
    # Plot realized returns in the background
    plt.plot(test_returns.index, test_returns, color='grey', alpha=0.4, label='Realized Returns', linewidth=1)
    
    # Plot predicted mean (mu)
    point_pred = np.full(len(test_returns), mu)
    plt.plot(test_returns.index, point_pred, color='blue', linewidth=1.5, label=f'Mean={mu:.4f})')
    
    # Calculate and plot the dynamic volatility bands
    upper_bound = point_pred + t_quantile * test_vol
    lower_bound = point_pred - t_quantile * test_vol
    
    plt.plot(test_returns.index, upper_bound, color='red', linestyle='--', alpha=0.8, label=f'95% CI (t-dist, q={t_quantile:.2f})')
    plt.plot(test_returns.index, lower_bound, color='red', linestyle='--')
    
    # Shade the confidence area
    plt.fill_between(test_returns.index, lower_bound, upper_bound, color='red', alpha=0.05)

    plt.title(f"{asset_name} - Out-of-Sample Volatility Envelope (dof={nu:.2f})", fontsize=12, fontweight='bold')    
    plt.legend(loc='upper left')
    plt.grid(alpha=0.2)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

    script_dir = os.path.join(project_root, "data", "processed")
    data_path = os.path.join(script_dir, "returns.csv")

    res_dir = os.path.join(project_root, "results", "dynamics", "NGARCH-T")
    os.makedirs(res_dir, exist_ok=True)
        
    diag_dir = os.path.join(res_dir, "plots")
    os.makedirs(diag_dir, exist_ok=True)

    pred_dir = os.path.join(diag_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    SPLIT_DATE = pd.Timestamp("2025-01-02")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True).apply(pd.to_numeric, errors='coerce')
    holdout_days = len(df[df.index >= SPLIT_DATE])
    
    new_models = {}
    train_u, test_u, params = {}, {}, []
    for col in df.columns:  
        print(f"Fitting: {col}")      
        m = NGARCH_T()
        m.fit_and_predict(df[col].dropna(), holdout_days=holdout_days)

        new_models[col] = m
        train_u[col] = m.train_uniforms
        test_u[col] = m.test_uniforms

        mu, omega, alpha, beta, theta, nu = m.params
        persistence = alpha * (1 + theta**2) + beta
        params.append({'asset': col, 'mu': mu, 'omega': omega, 'alpha': alpha,
                       'beta': beta, 'theta': theta, 'nu': nu, 'persistence': persistence})

        # Evaluaiton
        mu_val, _, _, _, _, nu_val = m.params
        test_series = df[col].tail(holdout_days)
        plot_test_prediction(test_returns=test_series, test_vol=m.test_vol, mu=mu_val, nu=nu_val, 
                             asset_name=col, save_path=os.path.join(pred_dir, f"{col}_vol_envelope.png"))

        # Plots
        m.diagnostics(name=col, is_test=False, save=os.path.join(diag_dir, f"{col}_train_diag.png"))
        m.diagnostics(name=col, is_test=True,  save=os.path.join(diag_dir, f"{col}_test_diag.png"))
        plt.close('all')
        
    # Save Model
    model_pkl_path = os.path.join(res_dir, "fitted_marginals.pkl")
    
    if os.path.exists(model_pkl_path):
        with open(model_pkl_path, "rb") as f:
            all_models = pickle.load(f)
        print(f"[-] Loaded existing pickle. Merging {len(new_models)} new models.")
    else:
        all_models = {}

    all_models.update(new_models) # NGARCH models will overwrite/add to HAR models

    with open(model_pkl_path, "wb") as f:
        pickle.dump(all_models, f)
    print(f"[-] Hybrid Model Library updated at: {model_pkl_path}")

    # CSVs
    train_df, test_df = pd.DataFrame(train_u), pd.DataFrame(test_u)
    train_df.index, test_df.index = pd.to_datetime(train_df.index).date, pd.to_datetime(test_df.index).date
    train_df.index.name, test_df.index.name = "Date", "Date"

    train_df.to_csv(os.path.join(res_dir, "train_uniforms_ngarch_t.csv"))
    test_df.to_csv(os.path.join(res_dir, "test_uniforms_ngarch_t.csv"))

    p_df = pd.DataFrame(params)
    p_df.to_csv(os.path.join(res_dir, "ngarch_t_params.csv"), index=False)

    # Uniformity Report
    alpha = 0.05
    failed_train = []
    failed_test = []

    for col in df.columns:
        # Run KS test on the generated uniforms
        _, p_train = kstest(train_u[col].dropna(), 'uniform')
        _, p_test = kstest(test_u[col].dropna(), 'uniform')
        
        if p_train < alpha:
            failed_train.append((col, p_train))
        if p_test < alpha:
            failed_test.append((col, p_test))

    total_cols = len(df.columns)

    print("\n" + "="*50)
    print("📊 FINAL UNIFORMITY REPORT (NGARCH-t)")
    print("="*50)

    print("--- IN-SAMPLE (Train 2020-2024) ---")
    if not failed_train:
        print(f"✅ All {total_cols} assets passed uniformity (p > {alpha}).")
    else:
        print(f"❌ {len(failed_train)}/{total_cols} assets failed (p < {alpha}):")
        for c, p in failed_train: 
            print(f"   - {c:.<20} p: {p:.5f}")
            
    print("\n--- OUT-OF-SAMPLE (Test 2025) ---")
    if not failed_test:
        print(f"✅ All {total_cols} assets passed uniformity (p > {alpha}).")
    else:
        print(f"❌ {len(failed_test)}/{total_cols} assets failed (p < {alpha}):")
        for c, p in failed_test: 
            print(f"   - {c:.<20} p: {p:.5f}")
    print("="*50)
    
    print(f"\n\nParameters:\n{p_df[['asset', 'alpha', 'beta', 'persistence', 'theta', 'nu']]}")
