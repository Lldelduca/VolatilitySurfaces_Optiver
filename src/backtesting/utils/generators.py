import numpy as np
import torch
import pyvinecopulib as pv
from scipy.stats import norm, t as student_t

# Import your PyTorch Copula modules
from src.dependence.gas_mixed_vine import GASPairCopula
from src.dependence.neural_mixed_vine import NeuralPairCopula

class DynamicGASVine:
    def __init__(self, pth_path, static_json_path):
        self.base_copula = pv.Vinecop.from_file(static_json_path)
        self.matrix = np.array(self.base_copula.matrix, dtype=np.int64)
        self.N = self.matrix.shape[0]
        self.models = {}
        
        gas_dict = torch.load(pth_path, map_location='cpu', weights_only=False)
        for tree in range(self.N - 1):
            for edge in range(self.N - 1 - tree):
                edge_key = f"T{tree}_E{edge}"
                info = gas_dict.get(edge_key, None)
                if not info or info['family'] == 'indep':
                    self.models[edge_key] = None
                    continue
                    
                model = GASPairCopula(info['family'], info['rotation'])
                model.omega.data = torch.tensor(info['omega'])
                model.A.data = torch.tensor(info['A'])
                b_val = info['B']
                model.B_logit.data = torch.tensor(np.log(b_val / (1 - b_val + 1e-8)) if b_val < 1.0 else 10.0)
                if model.nu_param is not None and not np.isnan(info.get('nu', np.nan)):
                    model.nu_param.data = torch.tensor(np.log(np.exp(max(info['nu'] - 2.01, 1e-6)) - 1))
                
                oos = float(info['oos_forecast'])
                if 'gaussian' in info['family'] or 'student' in info['family']: f_init = np.arctanh(np.clip(oos / 0.999, -0.99, 0.99))
                elif 'clayton' in info['family']: f_init = np.log(np.exp(max(oos - 1e-5, 1e-6)) - 1)
                elif 'gumbel' in info['family']: f_init = np.log(np.exp(max(oos - 1.0001, 1e-6)) - 1)
                else: f_init = oos
                    
                model.f_t = torch.tensor(f_init)
                self.models[edge_key] = model
        self._push_to_cpp()
        
    def _push_to_cpp(self):
        pcs_list = []
        trunc_lvl = len(self.base_copula.pair_copulas)
        for tree in range(trunc_lvl):
            tree_list = []
            for edge in range(self.N - 1 - tree):
                pc = self.base_copula.get_pair_copula(tree, edge)
                model = self.models[f"T{tree}_E{edge}"]
                if model:
                    # FIX: GAS natively stores its forecast in model.f_t
                    theta = float(model.transform_parameter(model.f_t).item())
                    nu = model.get_nu()
                    
                    if pc.parameters.shape[0] == 2:
                        nu_val = float(nu.item()) if nu is not None else 5.0
                        params = np.array([[np.clip(theta, -0.999, 0.999)], [nu_val]])
                    else:
                        if any(f in model.family for f in ['gaussian', 'student']): theta = np.clip(theta, -0.999, 0.999)
                        elif 'frank' in model.family: theta = np.clip(theta, -40.0, 40.0); theta = 1e-4 if abs(theta) < 1e-4 else theta
                        elif 'clayton' in model.family: theta = np.clip(theta, 1e-5, 28.0)
                        elif 'gumbel' in model.family: theta = np.clip(theta, 1.001, 28.0)
                        params = np.array([[theta]])
                    pc.parameters = params
                tree_list.append(pc)
            pcs_list.append(tree_list)
        self.base_copula = pv.Vinecop.from_structure(structure=self.base_copula.structure, pair_copulas=pcs_list)
        
    def simulate(self, n_scenarios): return np.clip(self.base_copula.simulate(n_scenarios), 1e-6, 1 - 1e-6)

    def update_states(self, u_realized_np):
        u_tensor = torch.tensor(u_realized_np, dtype=torch.float64).view(1, -1)
        M = self.matrix
        if M.max() == self.N: M -= 1 
        if np.sum(M[0] >= 0) > np.sum(M[-1] >= 0): M = np.flipud(M)
            
        h_storage = {(i, -1): u_tensor[:, i] for i in range(self.N)}
        for tree in range(self.N - 1):
            for edge in range(self.N - 1 - tree):
                row, col = self.N - 1 - tree, edge
                u_vec = h_storage[(M[row, col], -1)] if tree == 0 else h_storage[(col, tree-1)]
                var_2, partner_col = M[col, col], -1
                
                if tree == 0: v_vec = h_storage[(var_2, -1)]
                else:
                    for k in range(self.N):
                        if M[row+1, k] == var_2: partner_col = k; break
                    v_vec = h_storage[(partner_col, tree-1)]
                    
                model = self.models[f"T{tree}_E{edge}"]
                if model is None:
                    h_storage[(edge, tree)] = u_vec
                    if tree < self.N - 2: h_storage[(partner_col, tree)] = v_vec
                else:
                    _, _, h_dir, h_indir = model.update_step(u_vec, v_vec)
                    h_storage[(edge, tree)] = h_dir
                    if tree < self.N - 2: h_storage[(partner_col, tree)] = h_indir
        self._push_to_cpp()

class DynamicNeuralVine:
    def __init__(self, pth_path, static_json_path, history_window):
        self.base_copula = pv.Vinecop.from_file(static_json_path)
        self.matrix = np.array(self.base_copula.matrix, dtype=np.int64)
        self.N = self.matrix.shape[0]
        self.models = {}
        
        neural_dict = torch.load(pth_path, map_location='cpu', weights_only=False)

        for tree in range(self.N - 1):
            for edge in range(self.N - 1 - tree):
                edge_key = f"T{tree}_E{edge}"
                info = neural_dict.get(edge_key, None)

                if not info or info.get('family') == 'indep':
                    self.models[edge_key] = None
                    continue
                
                model = NeuralPairCopula(
                    family=info['family'], 
                    rotation=info.get('rotation', 0),
                    hidden_dim=info.get('hidden_dim', 8),
                    num_layers=info.get('num_layers', 1),
                    dropout=info.get('dropout', 0.0)
                )
                
                if 'state_dict' in info:
                    model.load_state_dict(info['state_dict'], strict=True)
                
                model.eval()
                self.models[edge_key] = model
                
        self._warm_up(history_window)
        self._push_to_cpp()

    def _warm_up(self, history_window):
        for t in range(history_window.shape[0]):
            self.update_states(history_window[t:t+1])
            
    def _push_to_cpp(self):
        pcs_list = []
        trunc_lvl = len(self.base_copula.pair_copulas)
        for tree in range(trunc_lvl):
            tree_list = []
            for edge in range(self.N - 1 - tree):
                pc = self.base_copula.get_pair_copula(tree, edge)
                model = self.models[f"T{tree}_E{edge}"]
                if model:
                    with torch.no_grad():
                        dummy_in = torch.zeros(1, 1, 2)
                        out, _ = model.rnn(dummy_in, model.hidden_state)
                        f_t = model.head(out).squeeze()
                        theta = float(model.transform_parameter(f_t).item())
                        nu = model.get_nu()
                    
                    if pc.parameters.shape[0] == 2:
                        nu_val = float(nu.item()) if nu is not None else 5.0
                        params = np.array([[np.clip(theta, -0.999, 0.999)], [nu_val]])
                    else:
                        if any(f in model.family for f in ['gaussian', 'student']): theta = np.clip(theta, -0.999, 0.999)
                        elif 'frank' in model.family: theta = np.clip(theta, -40.0, 40.0); theta = 1e-4 if abs(theta) < 1e-4 else theta
                        elif 'clayton' in model.family: theta = np.clip(theta, 1e-5, 28.0)
                        elif 'gumbel' in model.family: theta = np.clip(theta, 1.001, 28.0)
                        params = np.array([[theta]])
                    pc.parameters = params
                tree_list.append(pc)
            pcs_list.append(tree_list)
        self.base_copula = pv.Vinecop.from_structure(structure=self.base_copula.structure, pair_copulas=pcs_list)
                
    def simulate(self, n_scenarios): return np.clip(self.base_copula.simulate(n_scenarios), 1e-6, 1 - 1e-6)

    def update_states(self, u_realized_np):
        u_tensor = torch.tensor(u_realized_np, dtype=torch.float64).view(1, -1)
        M = self.matrix
        if M.max() == self.N: M -= 1 
        if np.sum(M[0] >= 0) > np.sum(M[-1] >= 0): M = np.flipud(M)
            
        h_storage = {(i, -1): u_tensor[:, i] for i in range(self.N)}
        for tree in range(self.N - 1):
            for edge in range(self.N - 1 - tree):
                row, col = self.N - 1 - tree, edge
                u_vec = h_storage[(M[row, col], -1)] if tree == 0 else h_storage[(col, tree-1)]
                var_2, partner_col = M[col, col], -1
                
                if tree == 0: v_vec = h_storage[(var_2, -1)]
                else:
                    for k in range(self.N):
                        if M[row+1, k] == var_2: partner_col = k; break
                    v_vec = h_storage[(partner_col, tree-1)]
                    
                model = self.models[f"T{tree}_E{edge}"]
                if model is None:
                    h_storage[(edge, tree)] = u_vec
                    if tree < self.N - 2: h_storage[(partner_col, tree)] = v_vec
                else:
                    _, _, h_dir, h_indir = model.step_forward(u_vec, v_vec)
                    h_storage[(edge, tree)] = h_dir
                    if tree < self.N - 2: h_storage[(partner_col, tree)] = h_indir
        self._push_to_cpp()

class UniversalScenarioGenerator:
    def __init__(self, factor_order, copula_model, model_id):
        self.factor_order = factor_order
        self.copula = copula_model 
        self.model_id = model_id
        self._ng_idx, self._har_idx, self._nsde_idx = [], [], []

    def classify_marginals(self, marginals):
        for i, n in enumerate(self.factor_order):
            m = marginals[n]
            if hasattr(m, 'pi_drift') and hasattr(m, 'pi_diff'): self._nsde_idx.append(i)
            elif hasattr(m, 'params') and isinstance(m.params, list): self._ng_idx.append(i)
            else: self._har_idx.append(i)

    def simulate_1day_dual(self, n_scenarios, init_states, marginals):
        if not self._ng_idx and not self._har_idx and not self._nsde_idx: 
            self.classify_marginals(marginals)
            
        dim = len(self.factor_order)
        paths_j = np.zeros((n_scenarios, dim))
        paths_i = np.zeros((n_scenarios, dim))
        
        U_j = np.clip(self.copula.simulate(n_scenarios), 1e-6, 1 - 1e-6)
        U_i = np.random.uniform(1e-6, 1 - 1e-6, size=(n_scenarios, dim))

        for d_idx in self._ng_idx:
            n_name, m = self.factor_order[d_idx], marginals[self.factor_order[d_idx]]
            mu, omega, alpha, beta, theta, nu = m.params
            prev_sig = max(np.sqrt(init_states[n_name]['sigma2']), 1e-6)
            prev_z = init_states[n_name]['resid'] / prev_sig
            next_sig2 = omega + alpha * ((prev_z - theta)**2) * init_states[n_name]['sigma2'] + beta * init_states[n_name]['sigma2']
            sig_term = np.sqrt(next_sig2)
            paths_j[:, d_idx] = mu + sig_term * student_t.ppf(U_j[:, d_idx], df=nu)
            paths_i[:, d_idx] = mu + sig_term * student_t.ppf(U_i[:, d_idx], df=nu)

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
    
    # Add this method to UniversalScenarioGenerator in generators.py
    def simulate_multiday(self, n_scenarios, horizon, init_states, marginals, use_copula=True):
        if not self._ng_idx and not self._har_idx and not self._nsde_idx: 
            self.classify_marginals(marginals)
            
        dim = len(self.factor_order)
        paths = np.zeros((n_scenarios, horizon, dim))
        n_total = n_scenarios * horizon
        
        # 1. Draw all uniforms at once from the frozen t=0 Copula
        if use_copula and self.copula is not None:
            U_all = np.clip(self.copula.simulate(n_total), 1e-6, 1 - 1e-6)
        else:
            U_all = np.random.uniform(1e-6, 1 - 1e-6, size=(n_total, dim))
            
        U_all = U_all.reshape((horizon, n_scenarios, dim))

        # 2. Extract initial states
        sig2_n = np.array([init_states[self.factor_order[i]]['sigma2'] for i in self._ng_idx]) if self._ng_idx else None
        eps_n = np.array([init_states[self.factor_order[i]]['resid'] for i in self._ng_idx]) if self._ng_idx else None
        
        sig2_h = np.array([init_states[self.factor_order[i]]['sigma2'] for i in self._har_idx]) if self._har_idx else None
        resid_h = np.array([init_states[self.factor_order[i]]['resid'] for i in self._har_idx]) if self._har_idx else None
        
        hist_h = np.zeros((n_scenarios, len(self._har_idx), 22)) if self._har_idx else None
        if self._har_idx:
            for j, idx in enumerate(self._har_idx):
                hist_h[:, j, :] = np.tile(init_states[self.factor_order[idx]]['history'][-22:], (n_scenarios, 1))

        if self._ng_idx:
            sig2_n = np.tile(sig2_n, (n_scenarios, 1))
            eps_n = np.tile(eps_n, (n_scenarios, 1))
            p_n = np.array([marginals[self.factor_order[i]].params for i in self._ng_idx])
            mu_n, om_n, al_n, be_n, th_n, nu_n = [p_n[:, k].reshape(1, -1) for k in range(6)]

        if self._har_idx:
            sig2_h = np.tile(sig2_h, (n_scenarios, 1))
            resid_h = np.tile(resid_h, (n_scenarios, 1))
            p_h = np.array([[marginals[self.factor_order[i]].params[k] for k in ['har_intercept', 'har_daily', 'har_weekly', 'har_monthly', 'garch_omega', 'garch_alpha', 'garch_beta']] for i in self._har_idx])
            h_int, h_d, h_w, h_m, g_om, g_al, g_be = [p_h[:, k].reshape(1, -1) for k in range(7)]

        dt = 1.0 / 252.0

        # 3. Project paths forward sequentially
        for t in range(horizon):
            U = U_all[t]
            
            if self._ng_idx:
                U_n = U[:, self._ng_idx]
                z_n = student_t.ppf(U_n, df=nu_n)
                prev_sig = np.sqrt(sig2_n)
                prev_z = np.where(prev_sig > 1e-6, eps_n / np.maximum(prev_sig, 1e-6), 0.0)
                next_sig2 = om_n + al_n * ((prev_z - th_n)**2) * sig2_n + be_n * sig2_n
                shock_n = np.sqrt(next_sig2) * z_n
                paths[:, t, self._ng_idx] = mu_n + shock_n
                sig2_n, eps_n = next_sig2, shock_n

            if self._har_idx:
                U_h = U[:, self._har_idx]
                z_h = np.zeros_like(U_h)
                for j, idx in enumerate(self._har_idx):
                    m = marginals[self.factor_order[idx]]
                    z_h[:, j] = m.evt_model.inverse_transform(U_h[:, j]) if hasattr(m, 'evt_model') else norm.ppf(U_h[:, j])
                
                next_sig2_h = g_om + g_al * (resid_h**2) + g_be * sig2_h
                mean_h = h_int + h_d * hist_h[:, :, -1] + h_w * hist_h[:, :, -5:].mean(axis=2) + h_m * hist_h.mean(axis=2)
                shock_h = np.sqrt(next_sig2_h) * z_h
                val_h = mean_h + shock_h
                paths[:, t, self._har_idx] = val_h
                sig2_h, resid_h = next_sig2_h, shock_h
                hist_h = np.concatenate([hist_h[:, :, 1:], val_h[:, :, np.newaxis]], axis=2)

            if self._nsde_idx:
                for d_idx in self._nsde_idx:
                    n_name, m = self.factor_order[d_idx], marginals[self.factor_order[d_idx]]
                    # Use actual projected history for t > 0
                    if t == 0:
                        window_data = init_states[n_name]['history'][-m.n_lags:]
                        window_data = np.tile(window_data, (n_scenarios, 1))
                    else:
                        window_data = np.concatenate([window_data[:, 1:], paths[:, t-1, d_idx:d_idx+1]], axis=1)
                    
                    window = torch.tensor(window_data, dtype=torch.float64, device=m.device).unsqueeze(-1)
                    with torch.no_grad():
                        mu_p = m.pi_drift(window).squeeze(-1).numpy()
                        sig_p = m.pi_diff(window).squeeze(-1).numpy()
                        nu_v = m.nu.item()
                        
                    paths[:, t, d_idx] = mu_p * dt + sig_p * np.sqrt(dt) * student_t.ppf(U[:, d_idx], df=nu_v)

        return paths

    def calculate_realized_uniforms(self, realized_row, init_states, marginals):
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