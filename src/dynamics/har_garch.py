import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from arch import arch_model
from evt import EVT

class HAR:
    def __init__(self, use_iterative_wls=True):
        self.models = []
        self.evts = []
        self.garch_models = []
        self.uniforms_df = None
        self.use_iterative_wls = use_iterative_wls
        
    def _compute_har_features(self, series):
        s = pd.Series(series)
        
        # Calculate Averages (Daily, Weekly, Monthly)
        avg_daily = s
        avg_weekly = s.rolling(window=5, min_periods=5).mean()
        avg_monthly = s.rolling(window=22, min_periods=22).mean()
        
        # Lag features by 1 day to prevent look-ahead bias
        feat_d = avg_daily.shift(1)
        feat_w = avg_weekly.shift(1)
        feat_m = avg_monthly.shift(1)
        
        df_feats = pd.concat([feat_d, feat_w, feat_m], axis=1)
        df_feats.columns = ['d', 'w', 'm']
        df_target = s
        
        # Alignment
        df_combined = pd.concat([df_feats, df_target.rename('target')], axis=1).dropna()
        
        X = df_combined[['d', 'w', 'm']].values
        Y = df_combined['target'].values
        valid_idx = df_combined.index
        
        return X, Y, valid_idx

    # HAR-GARCH-EVT pipeline
    def fit(self, scores, max_iter=3):
        if not isinstance(scores, pd.DataFrame):
            scores = pd.DataFrame(scores)

        self.models = []
        self.evts = []
        self.garch_models = []
        
        # Dictionary to collect results aligned by index
        uniforms_dict = {}
        
        for i, col in enumerate(scores.columns):
            factor_data = scores[col]
            X, Y, valid_idx = self._compute_har_features(factor_data)
            
            # 1: Iterative WLS (HAR + GARCH)
            weights = np.ones(len(Y))
            
            har_model = LinearRegression()
            garch_res = None
            residuals = None
            
            iterations = max_iter if self.use_iterative_wls else 1
            for _ in range(iterations):
                # 1. Fit Mean Model (HAR) with current weights
                har_model.fit(X, Y, sample_weight=weights)
                
                # 2. Get Residuals
                pred_mean = har_model.predict(X)
                residuals = Y - pred_mean
                
                # 3. Fit Volatility Model (GARCH) on residuals
                garch = arch_model(residuals, vol='GARCH', p=1, q=1, mean='Zero', dist='normal')
                garch_res = garch.fit(disp='off')
                
                # 4. Update weights for next loop (Inverse Variance)
                cond_vol = garch_res.conditional_volatility
                weights = 1.0 / (cond_vol**2 + 1e-6)

            self.models.append(har_model)
            self.garch_models.append(garch_res)

            # 2: Standardized Residuals & EVT
            z = residuals / garch_res.conditional_volatility
            
            # Fit EVT
            evt = EVT()
            evt.fit(z, lower_quantile=0.10, upper_quantile=0.10)
            self.evts.append(evt)
            
            # Transform to Uniforms
            u = evt.transform(z)
            uniforms_dict[col] = pd.Series(u, index=valid_idx)

        self.uniforms_df = pd.DataFrame(uniforms_dict)
        return self

    def get_uniforms(self):
        return self.uniforms_df
