import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.special import stdtrit, stdtr
from scipy.stats import kendalltau, norm
from tqdm import tqdm

torch.set_default_dtype(torch.float64)

class InverseStudentT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, nu):
        u_cpu = u.detach().cpu().numpy()
        nu_cpu = nu.detach().cpu().numpy()
        u_cpu = np.clip(u_cpu, 1e-12, 1 - 1e-12)
        x = stdtrit(nu_cpu, u_cpu)
        x_tensor = torch.from_numpy(x).to(u.device, dtype=u.dtype)
        ctx.save_for_backward(x_tensor, u, nu)
        return x_tensor

    @staticmethod
    def backward(ctx, grad_output):
        x, u, nu = ctx.saved_tensors
        pi = torch.tensor(3.141592653589793, device=x.device, dtype=x.dtype)
        log_const = torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2) - 0.5 * torch.log(nu * pi)
        log_kernel = -((nu + 1) / 2) * torch.log(1 + (x**2) / nu)
        pdf = torch.exp(log_const + log_kernel)
        grad_u = grad_output / torch.clamp(pdf, min=1e-100)
        grad_nu = None # Simplified for speed
        return grad_u, grad_nu

def inverse_t_cdf(u, nu):
    return InverseStudentT.apply(u, nu)


# NEURAL PAIR COPULA
class NeuralPairCopula(nn.Module):
    def __init__(self, family, rotation=0, hidden_dim=10):
        super().__init__()
        self.family = str(family).split('.')[-1].lower()
        self.rotation = int(rotation)
        
        # --- ARCHITECTURE: GRU ---
        # Input: 2 dims (Transformed u_t, v_t)
        # Hidden: 10 dims (Market State / Regime)
        self.rnn = nn.GRU(input_size=2, hidden_size=hidden_dim, batch_first=True)
        
        # Output Head: Maps Hidden State -> Parameter f_t
        self.head = nn.Linear(hidden_dim, 1)
        
        # Student-t Degrees of Freedom (Learned Static)
        if 'student' in self.family:
            self.nu_param = nn.Parameter(torch.tensor(2.0))
        else:
            self.register_parameter('nu_param', None)

    def get_nu(self):
        if self.nu_param is None: return None
        return torch.nn.functional.softplus(self.nu_param) + 2.01

    def rotate_data(self, u, v):
        if self.rotation == 90: return 1-u, v
        if self.rotation == 180: return 1-u, 1-v
        if self.rotation == 270: return u, 1-v
        return u, v
    
    def transform_parameter(self, f_t):
        """Maps Neural Output to Valid Parameter Space"""
        if 'gaussian' in self.family or 'student' in self.family:
            return torch.tanh(f_t) * 0.999 # (-1, 1)
        elif 'clayton' in self.family:
            return torch.nn.functional.softplus(f_t) + 1e-5 # (>0)
        elif 'gumbel' in self.family:
            return torch.nn.functional.softplus(f_t) + 1.0001 # (>1)
        elif 'frank' in self.family:
            val = f_t 
            mask = torch.abs(val) < 1e-4
            val = torch.where(mask, torch.sign(val) * 1e-4, val)
            return val
        return f_t

    def log_likelihood_pair(self, u, v, theta, nu=None):
        # EXACTLY SAME LOGIC AS GAS_MIXED_VINE.PY
        u_rot, v_rot = self.rotate_data(u, v)
        eps = 1e-9
        u_rot = torch.clamp(u_rot, eps, 1 - eps)
        v_rot = torch.clamp(v_rot, eps, 1 - eps)

        if 'gaussian' in self.family:
            rho = theta
            n = torch.distributions.Normal(0, 1)
            x, y = n.icdf(u_rot), n.icdf(v_rot)
            z = x**2 + y**2 - 2*rho*x*y
            log_det = 0.5 * torch.log(1 - rho**2 + 1e-8)
            log_exp = -0.5 * (z / (1 - rho**2 + 1e-8) - (x**2 + y**2))
            return -log_det + log_exp

        elif 'student' in self.family:
            rho = theta
            x = inverse_t_cdf(u_rot, nu)
            y = inverse_t_cdf(v_rot, nu)
            zeta = (x**2 + y**2 - 2*rho*x*y) / (1 - rho**2)
            term1 = -((nu + 2)/2) * torch.log(1 + zeta/nu)
            term2 = ((nu + 1)/2) * (torch.log(1 + x**2/nu) + torch.log(1 + y**2/nu))
            log_det = 0.5 * torch.log(1 - rho**2)
            lgamma = torch.lgamma
            const = lgamma((nu + 2)/2) + lgamma(nu/2) - 2*lgamma((nu+1)/2)
            return const - log_det + term1 + term2

        elif 'clayton' in self.family:
            t = theta
            a = torch.log(1 + t) - (1 + t) * (torch.log(u_rot) + torch.log(v_rot))
            b = torch.pow(u_rot, -t) + torch.pow(v_rot, -t) - 1
            return a - (2 + 1/t) * torch.log(torch.clamp(b, min=eps))

        elif 'gumbel' in self.family:
            t = theta
            x = -torch.log(u_rot)
            y = -torch.log(v_rot)
            A = torch.pow(x**t + y**t, 1/t)
            term1 = torch.log(A + t - 1); term2 = -A
            term3 = (t - 1) * (torch.log(x) + torch.log(y))
            term4 = (1/t - 2) * torch.log(x**t + y**t)
            jacobian = -torch.log(u_rot) - torch.log(v_rot)
            return term1 + term2 + term3 + term4 + jacobian
        
        elif 'frank' in self.family:
            t = theta
            exp_t = torch.exp(-t)
            exp_tu = torch.exp(-t * u_rot)
            exp_tv = torch.exp(-t * v_rot)
            log_num = torch.log(torch.abs(t) + eps) + torch.log(torch.abs(1 - exp_t) + eps) - t*(u_rot + v_rot)
            denom_inner = (1 - exp_t) - (1 - exp_tu) * (1 - exp_tv)
            log_denom = 2.0 * torch.log(torch.abs(denom_inner) + eps)
            return log_num - log_denom
        
        return torch.zeros_like(u)

    def compute_h_func(self, u, v, theta, nu=None):
        # EXACTLY SAME LOGIC AS GAS_MIXED_VINE.PY
        u_rot, v_rot = self.rotate_data(u, v)
        eps = 1e-9
        u_rot = torch.clamp(u_rot, eps, 1-eps)
        v_rot = torch.clamp(v_rot, eps, 1-eps)
        h_val = torch.zeros_like(u_rot)
        
        if 'gaussian' in self.family:
            n = torch.distributions.Normal(0, 1)
            x, y = n.icdf(u_rot), n.icdf(v_rot)
            h_val = n.cdf((x - theta*y) / torch.sqrt(1 - theta**2))
        elif 'student' in self.family:
            x = inverse_t_cdf(u_rot, nu)
            y = inverse_t_cdf(v_rot, nu)
            factor = torch.sqrt((nu + 1) / (nu + y**2) / (1 - theta**2))
            arg = (x - theta * y) * factor
            h_val = torch.tensor(stdtr((nu+1).detach().cpu().numpy(), arg.detach().cpu().numpy()))
        elif 'clayton' in self.family:
            t = theta
            term = torch.pow(v_rot, -t-1) * torch.pow(torch.pow(u_rot, -t) + torch.pow(v_rot, -t) - 1, -1/t - 1)
            h_val = term
        elif 'gumbel' in self.family:
            t = theta
            x = -torch.log(u_rot); y = -torch.log(v_rot)
            A = torch.pow(x**t + y**t, 1/t)
            h_val = torch.exp(-A) * torch.pow(y, t-1) / v_rot * torch.pow(x**t + y**t, 1/t - 1)
        elif 'frank' in self.family:
            t = theta
            et = torch.exp(-t); eu = torch.exp(-t*u_rot); ev = torch.exp(-t*v_rot)
            num = (eu - 1) * ev
            den = (et - 1) + (eu - 1) * (ev - 1)
            h_val = num / (den + 1e-20)

        if self.rotation in [90, 270]: h_val = 1 - h_val
        return torch.clamp(h_val, eps, 1 - eps)

    def forward(self, u_vec, v_vec):
        """
        Runs the GRU on the full sequence.
        """
        # 1. Probit Transform (Uniform[0,1] -> Normal(-inf, inf))
        # Neural nets learn better from Gaussian inputs than Uniform inputs.
        eps = 1e-6
        u_clamped = torch.clamp(u_vec, eps, 1-eps)
        v_clamped = torch.clamp(v_vec, eps, 1-eps)
        
        x_in = torch.erfinv(2 * u_clamped - 1) * 1.414
        y_in = torch.erfinv(2 * v_clamped - 1) * 1.414
        
        # 2. Prepare for RNN (Batch=1, Seq=T, Feat=2)
        # We assume u_vec, v_vec are (T,) tensors
        inputs = torch.stack([x_in, y_in], dim=1).unsqueeze(0) 
        
        # 3. Run GRU
        # out shape: (1, T, hidden_dim)
        rnn_out, _ = self.rnn(inputs)
        
        # 4. Project to Parameter Space
        # Shape: (T, 1) -> (T,)
        f_t_seq = self.head(rnn_out).squeeze(0).squeeze(1)
        
        # 5. Transform to Valid Theta
        thetas = self.transform_parameter(f_t_seq)
        
        # 6. Calculate Loss (Negative Log Likelihood)
        # Note: In a true forecasting setting, we would shift: rnn_out[t] predicts theta[t+1].
        # For this in-sample fitting, we model contemporaneous dependence.
        nu = self.get_nu()
        loss_vec = self.log_likelihood_pair(u_vec, v_vec, thetas, nu)
        
        return -torch.sum(loss_vec), thetas

# NEURAL VINE FITTER
def fit_neural_vine(u_matrix, structure, hidden_dim=10):
    T, N = u_matrix.shape
    device = torch.device("cpu")
    print(f"Fitting Neural GRU Vine on {device}...")
    u_tensor = torch.tensor(u_matrix, dtype=torch.float64).to(device)
    
    # --- Matrix Processing (Same as GAS) ---
    M = np.array(structure.matrix, dtype=np.int64)
    if M.max() == N: M -= 1 
    top_density = np.sum(M[0] >= 0)
    bot_density = np.sum(M[-1] >= 0)
    if top_density > bot_density: M = np.flipud(M)
    
    fams = structure.pair_copulas
    h_storage = {} 
    
    # Initialize Tree -1
    for i in range(N):
        h_storage[(i, -1)] = u_tensor[:, i]

    fitted_models = {}
    
    for tree in range(N - 1):
        edges = N - 1 - tree
        pbar = tqdm(range(edges), desc=f"Tree {tree+1}/{N-1}")
        
        for edge in pbar:
            row = N - 1 - tree; col = edge
            var_1 = M[row, col]
            u_vec = h_storage[(var_1, -1)] if tree == 0 else h_storage[(col, tree-1)]
            
            var_2 = M[col, col]
            partner_col = -1
            if tree == 0:
                v_vec = h_storage[(var_2, -1)]
            else:
                for k in range(N):
                    if M[row+1, k] == var_2: partner_col = k; break
                v_vec = h_storage[(partner_col, tree-1)]

            u_vec = torch.nan_to_num(u_vec, 0.5)
            v_vec = torch.nan_to_num(v_vec, 0.5)

            pc = fams[tree][edge]
            fam_str = str(pc.family)

            if 'indep' in fam_str.lower():
                h_direct = u_vec; h_indirect = v_vec
                fitted_models[f"T{tree}_E{edge}"] = {'path': np.zeros(T), 'family': 'indep'}
            else:
                # --- NEURAL MODEL HERE ---
                model = NeuralPairCopula(fam_str, rotation=pc.rotation, hidden_dim=hidden_dim).to(device)
                
                # AdamW helps regularize neural weights
                optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-3)
                loss_hist = []

                # Neural Training Loop
                for _ in range(50): # 50 epochs is usually enough for GRU on small data
                    optimizer.zero_grad()
                    loss, _ = model(u_vec, v_vec) # Batch forward
                    if torch.isnan(loss): break
                    loss.backward()
                    
                    # Gradient Clipping (Essential for RNN stability)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    loss_hist.append(loss.item())
                
                # Extract Path
                _, theta_path = model(u_vec, v_vec)
                theta_path = theta_path.detach()
                
                nu_val = model.get_nu()
                if nu_val is not None: nu_val = nu_val.detach()
                    
                h_direct = model.compute_h_func(u_vec, v_vec, theta_path, nu_val)
                h_indirect = model.compute_h_func(v_vec, u_vec, theta_path, nu_val)
                
                fitted_models[f"T{tree}_E{edge}"] = {
                    'loss_history': loss_hist,
                    'path': theta_path.numpy(),
                    'family': fam_str,
                    'rotation': pc.rotation
                }

            h_storage[(col, tree)] = h_direct
            if tree < N - 2: h_storage[(partner_col, tree)] = h_indirect
                
    return fitted_models
