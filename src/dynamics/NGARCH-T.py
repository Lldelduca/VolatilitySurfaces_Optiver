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
        self.uniforms = None
        self.vol = None
        self.resids = None
        
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
    
    def fit(self, returns):
        r = returns.values if isinstance(returns, pd.Series) else np.array(returns)
        idx = returns.index if isinstance(returns, pd.Series) else None
        
        # Initial values
        mu0 = np.mean(r)
        omega0 = 0.01 * np.var(r)
        nu0 = max(2.1, 6.0 / max(0.1, pd.Series(r).kurtosis()) + 4.0)
        x0 = [mu0, omega0, 0.08, 0.90, 0.5, nu0]
        
        bounds = [(None, None), (1e-8, None), (1e-6, 0.5), (0.0, 0.999), (-10, 10), (2.1, 100)]
        
        res = minimize(self._loglik, x0, args=(r,), method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 2000, 'ftol': 1e-10})
        
        self.params = res.x
        mu, omega, alpha, beta, theta, nu = self.params
        
        self.vol = self._garch_vol(r, mu, omega, alpha, beta, theta)
        self.resids = (r - mu) / self.vol
        
        u = student_t.cdf(self.resids, df=nu)
        self.uniforms = pd.Series(np.clip(u, 1e-6, 1-1e-6), index=idx) if idx is not None else np.clip(u, 1e-6, 1-1e-6)
        
        return self
    
    def diagnostics(self, name="Asset", save=None):
        # Unpack parameters
        mu, omega, alpha, beta, theta, nu = self.params
        z = self.resids
        u = self.uniforms.values if isinstance(self.uniforms, pd.Series) else self.uniforms
        
        fig = plt.figure(figsize=(16, 12))
        # Create a 3x3 grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        fig.suptitle(f'NGARCH-t Diagnostics: {name}', fontsize=14, fontweight='bold')
        
        # 1. Residuals time series (Spanning the whole top row)
        ax0 = fig.add_subplot(gs[0, :])
        ax0.plot(z, lw=0.5, alpha=0.7)
        ax0.axhline(0, color='r', ls='--', lw=0.8)
        ax0.axhline(2, color='orange', ls=':', lw=0.8)
        ax0.axhline(-2, color='orange', ls=':', lw=0.8)
        ax0.set_title('Standardized Residuals')
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
    
    df = pd.read_csv(data_path, index_col=0).apply(pd.to_numeric, errors='coerce')
    print(f"Shape: {df.shape}, Assets: {df.columns.tolist()}\n")
    
    models = {}
    uniforms = {}
    params = []
    
    for col in df.columns:        
        m = NGARCH_T()
        m.fit(df[col].dropna())
        
        models[col] = m
        uniforms[col] = m.uniforms
        
        m.diagnostics(name=col, save=os.path.join(diag_dir, f"{col}_diag.png"))
        plt.close()

        mu, omega, alpha, beta, theta, nu = models[col].params
        params.append({
            'asset': col, 'mu': mu, 'omega': omega, 'alpha': alpha,
            'beta': beta, 'theta': theta, 'nu': nu, 'persistence': alpha + beta
        })
    
    u_df = pd.DataFrame(uniforms)
    u_df.index = pd.to_datetime(u_df.index).date
    u_df.index.name = "Date"
    u_df.to_csv(os.path.join(res_dir, "uniforms_ngarch_t.csv"))

    p_df = pd.DataFrame(params)
    p_df.to_csv(os.path.join(res_dir, "ngarch_t_params.csv"), index=False)
    
    print(f"\n\nParameters:\n{p_df[['asset', 'alpha', 'beta', 'persistence', 'theta', 'nu']]}")