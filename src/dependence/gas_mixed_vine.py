import numpy as np
import pyvinecopulib as pv
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Bounds helper for copula families
def get_parameter_bounds(family):
    bounds = {
        pv.BicopFamily.gaussian: (-0.9999, 0.9999),
        pv.BicopFamily.student: (-0.9999, 0.9999),
        pv.BicopFamily.clayton: (1e-4, 20.0),      # Must be > 0
        pv.BicopFamily.gumbel: (1.0001, 20.0),     # Must be >= 1
        pv.BicopFamily.frank: (-20.0, 20.0),       # Non-zero
    }
    return bounds.get(family, (-np.inf, np.inf))

# Robust parameter converter to ensure compatibility with pyvinecopulib's C++ backend
def to_pv_params(theta, fixed_params):
    th_val = float(theta)
    p_list = [th_val]

    if len(fixed_params) > 1:
        for i in range(1, len(fixed_params)):
            p_list.append(float(fixed_params[i]))
            
    return np.array([p_list], dtype=np.float64)

# Maps the unbounded GAS factor 'f' to the valid parameter space 'theta'
def transform_f_to_theta(f, family):
    if family in [pv.BicopFamily.gaussian, pv.BicopFamily.student]:
        return np.tanh(f) # Maps (-inf, inf) -> (-1, 1)
    
    elif family == pv.BicopFamily.clayton:
        return np.exp(f) + 1e-4 # Maps to (0, inf)
    
    elif family in pv.BicopFamily.gumbel:
        return np.exp(f) + 1.0001 # Maps to (1, inf)
    
    elif family == pv.BicopFamily.frank:
        return f if abs(f) > 1e-4 else 1e-4
             
    return f

#  Inverse mapping used to initialize 'f' from the Static model's parameter
def inverse_transform_theta_to_f(theta, family):
    th = float(theta)
    if family in [pv.BicopFamily.gaussian, pv.BicopFamily.student]:
        return np.arctanh(np.clip(th, -0.995, 0.995))
    elif family == pv.BicopFamily.clayton:
        return np.log(max(th, 1e-4))
    elif family in pv.BicopFamily.gumbel:
        return np.log(max(th - 1.001, 1e-4))
    return th

# Compute the Score (Gradient of Log-Likelihood) using a 5-Point Stencil for numerical differentiation
def score(u, v, family, theta, fixed_params, rotation=0, epsilon=1e-5):
    try:
        def get_ll(th):
            # Bounds Check
            lb, ub = get_parameter_bounds(family)
            if th <= lb or th >= ub: return -1e10

            try:
                params_arr = to_pv_params(th, fixed_params)
                bc = pv.Bicop(family=family, parameters=params_arr, rotation=rotation)
                ll = bc.loglik(np.column_stack([u, v]))
                return -1e10 if (np.isnan(ll) or np.isinf(ll)) else ll
            except RuntimeError:
                return -1e10

        # Richardson Extrapolation (5 points)
        ll_p2 = get_ll(theta + 2*epsilon)
        ll_p1 = get_ll(theta + 1*epsilon)
        ll_m1 = get_ll(theta - 1*epsilon)
        ll_m2 = get_ll(theta - 2*epsilon)
        
        # Stability check
        if ll_p1 <= -1e9 or ll_m1 <= -1e9: return 0.0
        
        grad = (-ll_p2 + 8*ll_p1 - 8*ll_m1 + ll_m2) / (12 * epsilon)
        
        # Clip gradients to prevent GAS explosion during shocks
        return np.clip(grad, -50.0, 50.0)
        
    except:
        return 0.0

# Fits the GAS process for a single pair of variables
def fit_gas_edge(u, v, family, rotation=0):
    T = len(u)
    
    # Warm Start: Fit Static Copula to get initial parameters
    bc_static = pv.Bicop(family=family, rotation=rotation)
    bc_static.fit(np.column_stack([u, v]))
    static_params = np.array(bc_static.parameters).flatten()
    
    # Initial theta and GAS Factors
    theta_static = static_params[0]
    f_init = inverse_transform_theta_to_f(theta_static, family)

    # Tune A and B
    def objective(hyperparams):
        A, B = hyperparams
        
        # Constraints: Stationarity (A + B < 1) and Positivity (A > 0, B > 0) 
        if A <= 0.001 or B <= 0.001: return 1e12
        if A + B >= 0.999: return 1e12
        
        # omega = f_long_run * (1 - B) - A * expectation_of_score can be simplified to omega = f_init * (1 - B) 
        omega = f_init * (1 - B)

        f_t = f_init
        total_nll = 0.0
        for t in range(T):
            theta_t = transform_f_to_theta(f_t, family)
            
            # Bounds check
            lb, ub = get_parameter_bounds(family)
            if theta_t <= lb + 1e-4 or theta_t >= ub - 1e-4:
                total_nll += 1e6
                # Mean reversion if out of bounds
                f_t = omega + A * 0 + B * f_t 
                continue

            # Log Likelihood
            try:
                params = to_pv_params(theta_t, static_params)
                bc = pv.Bicop(family=family, parameters=params, rotation=rotation)
                ll = bc.loglik(np.column_stack([u[t], v[t]]))
                if np.isnan(ll) or np.isinf(ll):
                    total_nll += 1e6
                else:
                    total_nll -= ll
            except:
                total_nll += 1e6
            
            # Update Score
            st = score(np.array([u[t]]), np.array([v[t]]), family, theta_t, static_params, rotation)
            f_t = omega + A * st + B * f_t
            
        return total_nll

    best_res = None
    best_fun = 1e20
    guesses = [[0.05, 0.90], [0.02, 0.97], [0.1, 0.8]]

    for guess in guesses:
        res = minimize(objective, guess, method='Nelder-Mead', tol=1e-2)
        if res.fun < best_fun:
            best_fun = res.fun
            best_res = res
            
    best_A, best_B = best_res.x

    msg = f"Optimized Edge ({family}): A={best_A:.3f}, B={best_B:.3f}"
    tqdm.write(msg)

    # Final Run with Best Params to generate path
    omega_best = f_init * (1 - best_B)
    theta_path = np.zeros(T)
    f_t = f_init
    
    for t in range(T):
        theta_t = transform_f_to_theta(f_t, family)
        theta_path[t] = theta_t
        st = score(np.array([u[t]]), np.array([v[t]]), family, theta_t, static_params, rotation)
        f_t = omega_best + best_A * st + best_B * f_t
        
    return theta_path, static_params

# Fits the Mixed GAS Vine
def fit_gas_vine(u_data, static_model):
    T, N = u_data.shape
    M = np.array(static_model.matrix)
    
    # Orient Matrix to Lower Triangular if needed
    top_nonzeros = np.count_nonzero(M[0, :])
    bot_nonzeros = np.count_nonzero(M[N-1, :])
    if top_nonzeros > bot_nonzeros:
        M = np.flip(M).T # Standardize orientation
        print(">> Re-oriented Matrix.")

    dynamic_results = {}
    tree_outputs = { -1: {} }
    
    # Init Tree -1 (Raw Data)
    for i in range(N):
        tree_outputs[-1][i] = {
            'direct_var': i,
            'direct_h': u_data[:, i],
            'indirect_var': i,
            'indirect_h': u_data[:, i]
        }
        
    print(f"Fitting GAS Vine on {N} assets (T={T})...")
    
    for tree in range(N - 1):
        tree_outputs[tree] = {}
        pbar = tqdm(range(N - 1 - tree), desc=f"Tree {tree+1}")
        
        for edge_col in pbar:

            row_idx = N - 1 - tree
            a_idx = int(M[row_idx, edge_col]) - 1
            b_idx = int(M[edge_col, edge_col]) - 1
            
            prev_level = tree_outputs[tree - 1]
            u_vec = None
            v_vec = None
            
            # Search helper
            def find_input(target_var, prev_outputs):
                for k, out in prev_outputs.items():
                    if out['direct_var'] == target_var:
                        return out['direct_h']
                    if out['indirect_var'] == target_var:
                        return out['indirect_h']
                return None
            
            if tree == 0:
                u_vec = u_data[:, a_idx]
                v_vec = u_data[:, b_idx]
            else:
                u_vec = find_input(a_idx, prev_level)
                v_vec = find_input(b_idx, prev_level)

            if u_vec is None or v_vec is None:
                continue

            pc = static_model.pair_copulas[tree][edge_col]
            fam, rot = pc.family, pc.rotation
            
            if fam == pv.BicopFamily.indep:
                theta_path = np.zeros(T)
                h_a_next = u_vec
                h_b_next = v_vec
            else:
                theta_path, fixed_params = fit_gas_edge(u_vec, v_vec, fam, rot)
                
                h_a_next = np.zeros(T)
                h_b_next = np.zeros(T)
                
                for t in range(T):
                    try:
                        p_arr = to_pv_params(theta_path[t], fixed_params)
                        bc_t = pv.Bicop(family=fam, parameters=p_arr, rotation=rot)
                        
                        pt = np.array([[u_vec[t], v_vec[t]]])

                        h_a_next[t] = bc_t.hfunc2(pt).item()
                        h_b_next[t] = bc_t.hfunc1(pt).item()
                        
                    except:
                        h_a_next[t] = u_vec[t]
                        h_b_next[t] = v_vec[t]

            # Store for next tree
            tree_outputs[tree][edge_col] = {
                'direct_var': a_idx,
                'direct_h': h_a_next,  
                'indirect_var': b_idx,
                'indirect_h': h_b_next  
            }
            
            key = f"T{tree}_{a_idx}-{b_idx}"
            dynamic_results[key] = {
                'theta': theta_path,
                'family': str(fam).split('.')[-1],
                'rotation': rot
            }

    return dynamic_results
