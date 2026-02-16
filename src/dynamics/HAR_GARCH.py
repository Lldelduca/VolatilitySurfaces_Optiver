import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from arch import arch_model
from EVT import EVT
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import probplot, kstest, norm, uniform
import pickle
import os


class HAR_GARCH_EVT:
    def __init__(self, use_iterative_wls=True, max_iter=3):
        self.use_iterative_wls = use_iterative_wls
        self.max_iter = max_iter
        self.har_model = None
        self.garch_model = None
        self.evt_model = None
        self.params = {}
        
        self.train_uniforms = None
        self.test_uniforms = None
        self.resids = None 
        self.test_resids = None
        self.vol = None
        self.test_vol = None
        self.har_features = None 
        self.test_har_features = None
    
    def fit_and_predict(self, series, holdout_days=248):
        s = pd.Series(series)
        
        # Feature Engineering
        avg_daily = s
        avg_weekly = s.rolling(window=5, min_periods=5).mean()
        avg_monthly = s.rolling(window=22, min_periods=22).mean()
        
        df_feats = pd.concat([avg_daily.shift(1), avg_weekly.shift(1), avg_monthly.shift(1)], axis=1)
        df_feats.columns = ['daily', 'weekly', 'monthly']
        df_combined = pd.concat([df_feats, s.rename('target')], axis=1).dropna()
                
        X_full = df_combined[['daily', 'weekly', 'monthly']].values
        Y_full = df_combined['target'].values
        idx_full = df_combined.index

        X_train, Y_train, idx_train = X_full[:-holdout_days], Y_full[:-holdout_days], idx_full[:-holdout_days]
        X_test, Y_test, idx_test = X_full[-holdout_days:], Y_full[-holdout_days:], idx_full[-holdout_days:]

        # Fit HAR
        weights = np.ones(len(Y_train))
        iterations = self.max_iter if self.use_iterative_wls else 1
        for _ in range(iterations):
            har_model = LinearRegression()
            har_model.fit(X_train, Y_train, sample_weight=weights)
            res_train = Y_train - har_model.predict(X_train)
            
            garch = arch_model(res_train, vol='GARCH', p=1, q=1, mean='Zero', dist='normal', rescale=True)
            garch_res = garch.fit(disp='off', show_warning=False)
            weights = 1.0 / (garch_res.conditional_volatility**2 + 1e-6)
        
        self.har_model = har_model
        self.garch_model = garch_res
        self.params = {
            'har_intercept': har_model.intercept_, 'har_daily': har_model.coef_[0],
            'har_weekly': har_model.coef_[1], 'har_monthly': har_model.coef_[2],
            'garch_omega': garch_res.params['omega'], 'garch_alpha': garch_res.params['alpha[1]'],
            'garch_beta': garch_res.params['beta[1]'], 'loglikelihood': garch_res.loglikelihood
        }

        # OOS + Filtering
        res_test = Y_test - har_model.predict(X_test)
        res_full = np.concatenate([res_train, res_test])

        omega, alpha, beta = self.params['garch_omega'], self.params['garch_alpha'], self.params['garch_beta']
        vol_full = np.zeros(len(res_full))
        vol_full[0] = np.var(res_train)
        for t in range(1, len(res_full)):
            vol_full[t] = omega + alpha * res_full[t-1]**2 + beta * vol_full[t-1]
        vol_full = np.sqrt(vol_full)
        
        z_train = res_train / vol_full[:-holdout_days]
        z_test = res_test / vol_full[-holdout_days:]
        
        # EVT
        evt = EVT()
        evt.fit(z_train, lower_quantile=0.10, upper_quantile=0.10)
        self.evt_model = evt
                
        self.train_uniforms = pd.Series(evt.transform(z_train), index=idx_train)
        self.test_uniforms = pd.Series(evt.transform(z_test), index=idx_test)
        
        self.resids = z_train
        self.test_resids = z_test
        self.vol = vol_full[:-holdout_days]
        self.test_vol = vol_full[-holdout_days:]
        self.har_features = pd.DataFrame(X_train, columns=['daily', 'weekly', 'monthly'], index=idx_train)
        self.test_har_features = pd.DataFrame(X_test, columns=['daily', 'weekly', 'monthly'], index=idx_test)
        
        return self
    
    def diagnostics(self, name="Factor", is_test=False, save=None):
        # Swap between Train and Test data
        z = self.test_resids if is_test else self.resids
        u_series = self.test_uniforms if is_test else self.train_uniforms
        u = u_series.values
        features = self.test_har_features if is_test else self.har_features
        vol_series = self.test_vol if is_test else self.vol
        
        phase = "Out-of-Sample (Test)" if is_test else "In-Sample (Train)"
        
        fig = plt.figure(figsize=(16, 14))
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
        fig.suptitle(f'HAR-GARCH-EVT Diagnostics [{phase}]: {name}', fontsize=16, fontweight='bold')
        
        # 1. HAR fitted values vs actual
        ax0 = fig.add_subplot(gs[0, :])
        fitted = self.har_model.predict(features.values)
        actual = fitted + z * vol_series
        
        ax0.plot(u_series.index, fitted, 'b-', lw=1, alpha=0.7, label='HAR Fitted')
        ax0.plot(u_series.index, actual, 'red', lw=0.5, alpha=0.5, label='Actual')
        ax0.set_title('HAR Model Fit')
        ax0.legend()
        ax0.grid(alpha=0.3)
        
        # 2. Standardized residuals time series
        ax1 = fig.add_subplot(gs[1, :])
        ax1.plot(u_series.index, z, lw=0.5, alpha=0.7)
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
        ax2.axvline(self.evt_model.u_lower, color='orange', ls='--', lw=1.5, label='EVT thresholds')
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

    file_path = os.path.join(project_root, "outputs", "factors", "factors.csv")
    res_dir = os.path.join(project_root, "outputs", "dynamics")
    diag_dir = os.path.join(res_dir, "plots", "HAR-GARCH-EVT")
    os.makedirs(diag_dir, exist_ok=True)

    SPLIT_DATE = pd.Timestamp("2025-01-02")

    if os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col=0, parse_dates=True).apply(pd.to_numeric, errors='coerce')
        holdout_days = len(df[df.index >= SPLIT_DATE])
        print(f"Processing {df.shape[1]} factors. Holdout: {holdout_days} days (from {SPLIT_DATE.date()})...\n")
        
        fitted_models, train_u, test_u, params = {}, {}, {}, []
        
        for col in df.columns:
            print(f"Fitting: {col}")
            m = HAR_GARCH_EVT(use_iterative_wls=True, max_iter=3)
            m.fit_and_predict(df[col].dropna(), holdout_days=holdout_days)

            fitted_models[col] = m 
            train_u[col] = m.train_uniforms
            test_u[col] = m.test_uniforms
            
            m.diagnostics(name=col, is_test=False, save=os.path.join(diag_dir, f"{col}_train_diag.png"))
            m.diagnostics(name=col, is_test=True,  save=os.path.join(diag_dir, f"{col}_test_diag.png"))
            plt.close('all')

            p = m.params.copy()
            p['factor'] = col
            p['persistence'] = p['garch_alpha'] + p['garch_beta']
            params.append(p)
        
        # Save Model
        model_pkl_path = os.path.join(res_dir, "fitted_marginals.pkl")
        with open(model_pkl_path, "wb") as f:
            pickle.dump(fitted_models, f)

        print(f"[-] LIVE MODELS saved to: {model_pkl_path}")

        # CSVs
        train_df, test_df = pd.DataFrame(train_u), pd.DataFrame(test_u)
        train_df.index, test_df.index = pd.to_datetime(train_df.index).date, pd.to_datetime(test_df.index).date
        train_df.index.name, test_df.index.name = "Date", "Date"
        
        train_df.to_csv(os.path.join(res_dir, "train_uniforms_har_garch_evt.csv"))
        test_df.to_csv(os.path.join(res_dir, "test_uniforms_har_garch_evt.csv"))
        
        p_df = pd.DataFrame(params)
        p_df = p_df[['factor'] + [c for c in p_df.columns if c != 'factor']]
        p_df.to_csv(os.path.join(res_dir, "har_garch_evt_params.csv"), index=False)

        print(f"\nSuccess! Results and models saved in: {res_dir}")
    else:
        print(f"Error: Could not find {file_path}")
