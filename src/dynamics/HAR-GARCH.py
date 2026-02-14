import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from arch import arch_model
from EVT import EVT
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import probplot, kstest, norm, uniform
import os


class HAR_GARCH_EVT:
    def __init__(self, use_iterative_wls=True, max_iter=3):
        self.use_iterative_wls = use_iterative_wls
        self.max_iter = max_iter
        self.har_model = None
        self.garch_model = None
        self.evt_model = None
        self.params = {}
        self.uniforms = None
        self.resids = None
        self.vol = None
        self.har_features = None
        
    def _compute_har_features(self, series):
        s = pd.Series(series)
        
        avg_daily = s
        avg_weekly = s.rolling(window=5, min_periods=5).mean()
        avg_monthly = s.rolling(window=22, min_periods=22).mean()
        
        feat_d = avg_daily.shift(1)
        feat_w = avg_weekly.shift(1)
        feat_m = avg_monthly.shift(1)
        
        df_feats = pd.concat([feat_d, feat_w, feat_m], axis=1)
        df_feats.columns = ['daily', 'weekly', 'monthly']
        
        df_combined = pd.concat([df_feats, s.rename('target')], axis=1).dropna()
        
        X = df_combined[['daily', 'weekly', 'monthly']].values
        Y = df_combined['target'].values
        valid_idx = df_combined.index
        
        return X, Y, valid_idx
    
    def fit(self, returns):
        r = returns.values if isinstance(returns, pd.Series) else np.array(returns)
        idx = returns.index if isinstance(returns, pd.Series) else None
        
        X, Y, valid_idx = self._compute_har_features(r if idx is None else pd.Series(r, index=idx))
        
        weights = np.ones(len(Y))
        iterations = self.max_iter if self.use_iterative_wls else 1
        
        for iteration in range(iterations):
            har_model = LinearRegression()
            har_model.fit(X, Y, sample_weight=weights)
            
            pred_mean = har_model.predict(X)
            residuals = Y - pred_mean
            
            garch = arch_model(residuals, vol='GARCH', p=1, q=1, mean='Zero', dist='normal', rescale=True)
            garch_res = garch.fit(disp='off', show_warning=False)
            
            cond_vol = garch_res.conditional_volatility
            weights = 1.0 / (cond_vol**2 + 1e-6)
        
        self.har_model = har_model
        self.garch_model = garch_res
        
        self.params = {
            'har_intercept': har_model.intercept_,
            'har_daily': har_model.coef_[0],
            'har_weekly': har_model.coef_[1],
            'har_monthly': har_model.coef_[2],
            'garch_omega': garch_res.params['omega'],
            'garch_alpha': garch_res.params['alpha[1]'],
            'garch_beta': garch_res.params['beta[1]'],
            'loglikelihood': garch_res.loglikelihood
        }
        
        # print(f"HAR: intercept={self.params['har_intercept']:.4f}, "
        #       f"daily={self.params['har_daily']:.4f}, "
        #       f"weekly={self.params['har_weekly']:.4f}, "
        #       f"monthly={self.params['har_monthly']:.4f}")
        # print(f"GARCH: omega={self.params['garch_omega']:.6f}, "
        #       f"alpha={self.params['garch_alpha']:.4f}, "
        #       f"beta={self.params['garch_beta']:.4f}, "
        #       f"LL={self.params['loglikelihood']:.2f}")
        
        self.vol = garch_res.conditional_volatility
        self.resids = residuals / self.vol
        
        evt = EVT()
        evt.fit(self.resids, lower_quantile=0.10, upper_quantile=0.10)
        self.evt_model = evt
                
        u = evt.transform(self.resids)
        self.uniforms = pd.Series(u, index=valid_idx) if valid_idx is not None else u
        
        self.har_features = pd.DataFrame(X, columns=['daily', 'weekly', 'monthly'], index=valid_idx)
        
        return self
    
    def diagnostics(self, name="Factor", save=None):
        z = self.resids
        u = self.uniforms.values if isinstance(self.uniforms, pd.Series) else self.uniforms
        
        fig = plt.figure(figsize=(16, 14))
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
        fig.suptitle(f'HAR-GARCH-EVT Diagnostics: {name}', fontsize=14, fontweight='bold')
        
        # 1. HAR fitted values vs actual
        ax0 = fig.add_subplot(gs[0, :])
        actual = self.har_features.index
        fitted = self.har_model.predict(self.har_features.values)
        if isinstance(self.uniforms, pd.Series):
            ax0.plot(self.uniforms.index, self.uniforms.index.map(dict(zip(actual, fitted))), 
                    'b-', lw=1, alpha=0.7, label='HAR Fitted')
            ax0.plot(self.uniforms.index, 
                    self.uniforms.index.map(dict(zip(actual, fitted + self.resids * self.vol))),
                    'red', lw=0.5, alpha=0.5, label='Actual')
        ax0.set_title('HAR Model Fit')
        ax0.legend()
        ax0.grid(alpha=0.3)
        
        # 2. Standardized residuals time series
        ax1 = fig.add_subplot(gs[1, :])
        ax1.plot(z, lw=0.5, alpha=0.7)
        ax1.axhline(0, color='r', ls='--', lw=0.8)
        ax1.axhline(2, color='orange', ls=':', lw=0.8)
        ax1.axhline(-2, color='orange', ls=':', lw=0.8)
        ax1.set_title('Standardized Residuals')
        ax1.grid(alpha=0.3)
        
        # 3. Histogram with EVT fit
        ax2 = fig.add_subplot(gs[2, 0])
        ax2.hist(z, bins=50, density=True, alpha=0.6, color='steelblue', ec='black')
        x_range = np.linspace(z.min(), z.max(), 200)
        ax2.plot(x_range, norm.pdf(x_range), 'r--', lw=2, label='N(0,1)')
        ax2.axvline(self.evt_model.u_lower, color='orange', ls='--', lw=1.5, 
                   label='EVT thresholds')
        ax2.axvline(self.evt_model.u_upper, color='orange', ls='--', lw=1.5)
        ax2.set_title('Distribution with EVT Thresholds')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 4. QQ plot (Normal)
        ax3 = fig.add_subplot(gs[2, 1])
        probplot(z, dist=norm, plot=ax3)
        ax3.set_title('Q-Q Plot (Normal)')
        ax3.grid(alpha=0.3)
        
        # 5. ACF residuals
        ax4 = fig.add_subplot(gs[2, 2])
        plot_acf(z, lags=40, ax=ax4, alpha=0.05)
        ax4.set_title('ACF Residuals')
        ax4.grid(alpha=0.3)
        
        # 6. ACF squared residuals
        ax5 = fig.add_subplot(gs[3, 0])
        plot_acf(z**2, lags=40, ax=ax5, alpha=0.05)
        ax5.set_title('ACF Squared Residuals')
        ax5.grid(alpha=0.3)
        
        # 7. Uniform histogram (PIT)
        ax6 = fig.add_subplot(gs[3, 1])
        ax6.hist(u, bins=50, density=True, alpha=0.6, color='purple', ec='black')
        ax6.axhline(1.0, color='r', ls='--', lw=2, label='Uniform')
        ks_stat, ks_p = kstest(u, 'uniform')
        ax6.text(0.05, 0.95, f'KS: {ks_stat:.3f}\np={ks_p:.3f}',
                transform=ax6.transAxes, va='top',
                bbox=dict(boxstyle='round', fc='wheat', alpha=0.5))
        ax6.set_title('Uniform Transform (EVT)')
        ax6.legend()
        ax6.grid(alpha=0.3)
        
        ax7 = fig.add_subplot(gs[3, 2])
        probplot(u, dist=uniform, plot=ax7)
        ax7.set_title('Q-Q Uniform')
        ax7.grid(alpha=0.3)
        
        plt.subplots_adjust(top=0.93, bottom=0.07, left=0.07, right=0.93, hspace=0.5, wspace=0.3)
        if save:
            plt.savefig(save, dpi=300, bbox_inches='tight')
        return fig


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

    script_dir = os.path.join(project_root, "outputs", "factors")
    file_path = os.path.join(script_dir, "factors.csv")

    res_dir = os.path.join(project_root, "outputs", "dynamics")
    os.makedirs(res_dir, exist_ok=True)
        
    diag_dir = os.path.join(res_dir, "plots", "HAR-GARCH-EVT")
    os.makedirs(diag_dir, exist_ok=True)
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col=0).apply(pd.to_numeric, errors='coerce')
        print(f"Processing {df.shape[1]} factors from {file_path}...\n")
        
        models = {}
        uniforms = {}
        params = []
        
        for col in df.columns:
            print(f"\nFitting: {col}")
            m = HAR_GARCH_EVT(use_iterative_wls=True, max_iter=3)
            m.fit(df[col].dropna())
            
            models[col] = m
            uniforms[col] = m.uniforms
            m.diagnostics(name=col, save=os.path.join(diag_dir, f"{col}_diag.png"))
            plt.close()

            p = m.params.copy()
            p['factor'] = col
            p['persistence'] = p['garch_alpha'] + p['garch_beta']
            params.append(p)
        
        u_df = pd.DataFrame(uniforms)
        u_df.index = pd.to_datetime(u_df.index).date
        u_df.index.name = "Date"
        u_df.to_csv(os.path.join(res_dir, "uniforms_har_garch_evt.csv"))
        
        p_df = pd.DataFrame(params)
        cols = ['factor'] + [c for c in p_df.columns if c != 'factor']
        p_df = p_df[cols]
        p_df.to_csv(os.path.join(res_dir, "har_garch_evt_params.csv"), index=False)
        
    else:
        print(f"Error: Could not find factors.csv at {file_path}")