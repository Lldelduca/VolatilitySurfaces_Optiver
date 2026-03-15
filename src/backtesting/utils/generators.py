import numpy as np
import torch
from scipy.stats import norm, t as student_t

class UniversalScenarioGenerator:
    def __init__(self, factor_order, copula_model, model_id):
        self.factor_order = factor_order
        self.copula = copula_model 
        self.model_id = model_id
        
        self._ng_idx, self._har_idx, self._nsde_idx = [], [], []

    def classify_marginals(self, marginals):
        """Detects the classes of the imported marginals to route the simulation logic."""
        for i, n in enumerate(self.factor_order):
            m = marginals[n]
            # Detect NSDE
            if hasattr(m, 'pi_drift') and hasattr(m, 'pi_diff'): 
                self._nsde_idx.append(i)
            # Detect NGARCH (Spot)
            elif hasattr(m, 'params') and isinstance(m.params, list): 
                self._ng_idx.append(i)
            # Default to HAR-GARCH-EVT
            else: 
                self._har_idx.append(i)

    def simulate_1day_dual(self, n_scenarios, init_states, marginals):
        """
        Simulates one step forward.
        Computes both Joint (Vine) and Independent baseline simultaneously for efficiency.
        Handles M0 (Static) and M1-M4 (Dynamic) gracefully.
        """
        if not self._ng_idx and not self._har_idx and not self._nsde_idx: 
            self.classify_marginals(marginals)
            
        dim = len(self.factor_order)
        paths_j = np.zeros((n_scenarios, dim))
        paths_i = np.zeros((n_scenarios, dim))
        
        # --- 1. COPULA SIMULATION ---
        if self.model_id == 'M0':
            # Static Vine (pyvinecopulib)
            U_j = np.clip(self.copula.simulate(n_scenarios), 1e-6, 1 - 1e-6)
        else:
            # Dynamic Vines (GAS or Neural)
            U_j = np.clip(self.copula.simulate(n_scenarios), 1e-6, 1 - 1e-6)
            
        U_i = np.random.uniform(1e-6, 1 - 1e-6, size=(n_scenarios, dim))

        # --- 2. MARGINAL INVERSION ---
        # NGARCH Marginals
        for d_idx in self._ng_idx:
            n_name, m = self.factor_order[d_idx], marginals[self.factor_order[d_idx]]
            mu, omega, alpha, beta, theta, nu = m.params
            prev_sig = max(np.sqrt(init_states[n_name]['sigma2']), 1e-6)
            prev_z = init_states[n_name]['resid'] / prev_sig
            
            next_sig2 = omega + alpha * ((prev_z - theta)**2) * init_states[n_name]['sigma2'] + beta * init_states[n_name]['sigma2']
            sig_term = np.sqrt(next_sig2)
            
            paths_j[:, d_idx] = mu + sig_term * student_t.ppf(U_j[:, d_idx], df=nu)
            paths_i[:, d_idx] = mu + sig_term * student_t.ppf(U_i[:, d_idx], df=nu)

        # HAR-GARCH Marginals
        for d_idx in self._har_idx:
            n_name, m = self.factor_order[d_idx], marginals[self.factor_order[d_idx]]
            p, h = m.params, init_states[n_name]['history']
            mean = p['har_intercept'] + p['har_daily']*h[-1] + p['har_weekly']*h[-5:].mean() + p['har_monthly']*h.mean()
            sig_term = np.sqrt(p['garch_omega'] + p['garch_alpha']*(init_states[n_name]['resid']**2) + p['garch_beta']*init_states[n_name]['sigma2'])
            
            if hasattr(m, 'evt_model') and m.evt_model is not None:
                paths_j[:, d_idx] = mean + sig_term * m.evt_model.inverse_transform(U_j[:, d_idx])
                paths_i[:, d_idx] = mean + sig_term * m.evt_model.inverse_transform(U_i[:, d_idx])
            else:
                paths_j[:, d_idx] = mean + sig_term * norm.ppf(U_j[:, d_idx])
                paths_i[:, d_idx] = mean + sig_term * norm.ppf(U_i[:, d_idx])

        # Neural SDE Marginals
        if self._nsde_idx:
            dt = 1.0 / 252.0
            for d_idx in self._nsde_idx:
                n_name, m = self.factor_order[d_idx], marginals[self.factor_order[d_idx]]
                window = torch.tensor(init_states[n_name]['history'], dtype=torch.float64, device=m.device).unsqueeze(0).unsqueeze(-1)
                with torch.no_grad():
                    mu_pred, sig_pred, nu = m.pi_drift(window).item(), m.pi_diff(window).item(), m.nu.item()
                
                paths_j[:, d_idx] = mu_pred * dt + sig_pred * np.sqrt(dt) * student_t.ppf(U_j[:, d_idx], df=nu)
                paths_i[:, d_idx] = mu_pred * dt + sig_pred * np.sqrt(dt) * student_t.ppf(U_i[:, d_idx], df=nu)

        return paths_j, paths_i

    def calculate_realized_uniforms(self, realized_row, init_states, marginals):
        """Back-calculates the PIT uniforms required for the RNN/GAS hidden state update."""
        u_realized = np.zeros(len(self.factor_order))
        
        for d_idx in self._ng_idx:
            n_name, m = self.factor_order[d_idx], marginals[self.factor_order[d_idx]]
            mu, omega, alpha, beta, theta, nu = m.params
            prev_sig = max(np.sqrt(init_states[n_name]['sigma2']), 1e-6)
            next_sig2 = omega + alpha * ((init_states[n_name]['resid'] / prev_sig - theta)**2) * init_states[n_name]['sigma2'] + beta * init_states[n_name]['sigma2']
            u_realized[d_idx] = student_t.cdf((realized_row[d_idx] - mu) / np.sqrt(next_sig2), df=nu)

        for d_idx in self._har_idx:
            n_name, m = self.factor_order[d_idx], marginals[self.factor_order[d_idx]]
            p, h = m.params, init_states[n_name]['history']
            mean = p['har_intercept'] + p['har_daily']*h[-1] + p['har_weekly']*h[-5:].mean() + p['har_monthly']*h.mean()
            sig2 = p['garch_omega'] + p['garch_alpha']*(init_states[n_name]['resid']**2) + p['garch_beta']*init_states[n_name]['sigma2']
            z_real = (realized_row[d_idx] - mean) / np.sqrt(sig2)
            u_realized[d_idx] = m.evt_model.transform(np.array([z_real]))[0] if hasattr(m, 'evt_model') else norm.cdf(z_real)

        if self._nsde_idx:
            dt = 1.0 / 252.0
            for d_idx in self._nsde_idx:
                n_name, m = self.factor_order[d_idx], marginals[self.factor_order[d_idx]]
                window = torch.tensor(init_states[n_name]['history'], dtype=torch.float64, device=m.device).unsqueeze(0).unsqueeze(-1)
                with torch.no_grad():
                    mu_pred, sig_pred, nu = m.pi_drift(window).item(), m.pi_diff(window).item(), m.nu.item()
                u_realized[d_idx] = student_t.cdf((realized_row[d_idx] - mu_pred * dt) / (sig_pred * np.sqrt(dt)), df=nu)
            
        return np.clip(u_realized, 1e-6, 1 - 1e-6)