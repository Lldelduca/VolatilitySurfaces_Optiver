import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import t as student_t
from scipy.special import gammaln
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import probplot, kstest, norm, uniform
import os

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
            z = eps / np.sqrt(sig2[t-1])
            sig2[t] = max(omega + alpha * ((z - theta)**2) * sig2[t-1] + beta * sig2[t-1], 1e-6)
        
        return np.sqrt(sig2)
    
    def _loglik(self, params, r):
        mu, omega, alpha, beta, theta, nu = params
        
        if omega <= 0 or alpha <= 0 or beta < 0 or alpha + beta >= 1.0 or nu <= 2.0:
            return 1e10
        
        sig = self._garch_vol(r, mu, omega, alpha, beta, theta)
        z = (r - mu) / sig
        
        try:
            ll = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * nu)
            ll = np.sum(ll - ((nu + 1) / 2) * np.log(1 + z**2 / nu) - np.log(sig))
            return -ll
        except:
            return 1e10
    
    def fit_and_predict(self, returns, holdout_days=248):
        r_full = returns.values if isinstance(returns, pd.Series) else np.array(returns)
        idx_full = returns.index if isinstance(returns, pd.Series) else None

        r_train, idx_train = r_full[:-holdout_days], idx_full[:-holdout_days]
        r_test, idx_test = r_full[-holdout_days:], idx_full[-holdout_days:]
        
        mu0, omega0 = np.mean(r_train), 0.01 * np.var(r_train)
        nu0 = max(2.1, 6.0 / max(0.1, pd.Series(r_train).kurtosis()) + 4.0)
        x0 = [mu0, omega0, 0.08, 0.90, 0.5, nu0]
        bounds = [(None, None), (1e-8, None), (1e-6, 0.5), (0.0, 0.999), (-10, 10), (2.1, 100)]

        # Fit
        res = minimize(self._loglik, x0, args=(r_train,), method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 2000, 'ftol': 1e-10})
        self.params = res.x
        mu, omega, alpha, beta, theta, nu = self.params

        # Filter
        vol_full = self._garch_vol(r_full, mu, omega, alpha, beta, theta)
        z_full = (r_full - mu) / vol_full
        z_train, z_test = z_full[:-holdout_days], z_full[-holdout_days:]

        # Map 
        u_train = np.clip(student_t.cdf(z_train, df=nu), 1e-6, 1-1e-6)
        u_test = np.clip(student_t.cdf(z_test, df=nu), 1e-6, 1-1e-6)

        self.train_uniforms = pd.Series(u_train, index=idx_train)
        self.test_uniforms = pd.Series(u_test, index=idx_test)

        # Diagnostics
        self.vol = vol_full[:-holdout_days]
        self.test_vol = vol_full[-holdout_days:]
        self.resids = z_train
        self.test_resids = z_test
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

        actual_returns = z * vol_series + mu
        q95 = student_t.ppf(0.975, df=nu)

        ax0.plot(u_series.index, actual_returns, lw=1, alpha=0.6, color='black', label='Realized Returns')
        ax0.plot(u_series.index, mu + q95 * vol_series, 'r--', lw=1.5, label='Upper 95% Volatility Band')
        ax0.plot(u_series.index, mu - q95 * vol_series, 'g--', lw=1.5, label='Lower 95% Volatility Band (VaR)')
        ax0.set_title('Realized Returns vs. Predicted Conditional Volatility')
        ax0.legend(loc='upper left')
        ax0.grid(alpha=0.3)
        
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


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

    script_dir = os.path.join(project_root, "data", "processed")
    data_path = os.path.join(script_dir, "returns.csv")

    res_dir = os.path.join(project_root, "outputs", "dynamics")
    os.makedirs(res_dir, exist_ok=True)
        
    diag_dir = os.path.join(res_dir, "plots", "ngarch")
    os.makedirs(diag_dir, exist_ok=True)

    HOLDOUT_DAYS = 248 # 2025 Split

    df = pd.read_csv(data_path, index_col=0, parse_dates=True).apply(pd.to_numeric, errors='coerce')
    print(f"Shape: {df.shape}, Assets: {df.columns.tolist()}\n")
    
    train_u, test_u, params = {}, {}, []
    for col in df.columns:        
        m = NGARCH_T()
        m.fit_and_predict(df[col].dropna(), holdout_days=HOLDOUT_DAYS)
        
        train_u[col] = m.train_uniforms
        test_u[col] = m.test_uniforms

        # Plots
        m.diagnostics(name=col, is_test=False, save=os.path.join(diag_dir, f"{col}_train_diag.png"))
        m.diagnostics(name=col, is_test=True,  save=os.path.join(diag_dir, f"{col}_test_diag.png"))
        plt.close('all')

        mu, omega, alpha, beta, theta, nu = m.params
        params.append({
            'asset': col, 'mu': mu, 'omega': omega, 'alpha': alpha,
            'beta': beta, 'theta': theta, 'nu': nu, 'persistence': alpha + beta
        })
        
    train_df, test_df = pd.DataFrame(train_u), pd.DataFrame(test_u)
    train_df.index, test_df.index = pd.to_datetime(train_df.index).date, pd.to_datetime(test_df.index).date
    train_df.index.name, test_df.index.name = "Date", "Date"

    train_df.to_csv(os.path.join(res_dir, "train_uniforms_ngarch_t.csv"))
    test_df.to_csv(os.path.join(res_dir, "test_uniforms_ngarch_t.csv"))

    p_df = pd.DataFrame(params)
    p_df.to_csv(os.path.join(res_dir, "ngarch_t_params.csv"), index=False)
    
    print(f"\n\nParameters:\n{p_df[['asset', 'alpha', 'beta', 'persistence', 'theta', 'nu']]}")
