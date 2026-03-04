import numpy as np
from scipy import stats

class Reconstruction:
    """
    Surface-β Hierarchical FPCA reconstruction for a single asset.
    """

    def __init__(self, asset, grand_mean, asset_bias,
                 global_scores, B_j,
                 local_components, local_scores,
                 residuals=None,
                 maturity_labels=None,
                 moneyness_labels=None):

        self.__asset   = asset
        self.__M, self.__T = grand_mean.shape
        self.__S_PTS   = self.__M * self.__T

        self.__grand_mean = grand_mean
        self.__asset_bias = asset_bias

        self.__global_scores = np.asarray(global_scores)   # (N_OBS, K)
        self.__B_j           = np.asarray(B_j)             # (K, S_PTS)
        self.__n_global      = self.__B_j.shape[0]         # K  (no intercept row)

        N_OBS            = self.__global_scores.shape[0]
        self.__X_reg     = self.__global_scores            # (N_OBS, K) — no ones
        self.__n_obs     = N_OBS

        self.__local_components = np.asarray(local_components)
        self.__local_scores     = np.asarray(local_scores)
        self.__n_local          = self.__local_components.shape[0]

        self.__residuals  = np.asarray(residuals) if residuals is not None else None
        self.__mat_labels = maturity_labels
        self.__mon_labels = moneyness_labels

        # Pre-compute (X'X)^{-1} for inference — X has no intercept column
        XtX              = self.__X_reg.T @ self.__X_reg   # (K, K)
        self.__XtX_inv   = np.linalg.inv(XtX)
        self.__df        = N_OBS - self.__n_global

    # ─── private helpers ────────────────────────────────────────────────

    def __global_contribution(self, x_rows):
        """x_rows : (n, K) → (n, M, T)"""
        return (x_rows @ self.__B_j).reshape(-1, self.__M, self.__T)

    def __local_contribution(self, l_scores):
        """l_scores : (n, n_local) → (n, M, T)"""
        local_flat = self.__local_components.reshape(self.__n_local, -1)
        return (l_scores @ local_flat).reshape(-1, self.__M, self.__T)

    def __resolve_indices(self, time_idx):
        if time_idx is None:
            return self.__X_reg, self.__local_scores
        if isinstance(time_idx, int):
            return (self.__X_reg[time_idx:time_idx + 1],
                    self.__local_scores[time_idx:time_idx + 1])
        return self.__X_reg[time_idx], self.__local_scores[time_idx]

    @staticmethod
    def __maybe_squeeze(arr, was_single):
        return arr[0] if was_single else arr

    # ─── reconstruction ─────────────────────────────────────────────────

    def reconstruct(self, global_param=None, local_param=None, time_idx=None):
        base = self.__grand_mean + self.__asset_bias

        if global_param is None and local_param is None:
            x_rows, l_scores = self.__resolve_indices(time_idx)
            was_single = isinstance(time_idx, int)
        else:
            if global_param is None:
                global_param = self.__global_scores
            if local_param is None:
                local_param = self.__local_scores

            global_param = np.asarray(global_param)
            local_param  = np.asarray(local_param)
            was_single   = (global_param.ndim == 1)

            if was_single:
                global_param = global_param.reshape(1, -1)
                local_param  = local_param.reshape(1, -1)

            x_rows   = global_param          # (n, K) — no intercept
            l_scores = local_param

        global_contrib = self.__global_contribution(x_rows)
        local_contrib  = self.__local_contribution(l_scores)
        result = base[np.newaxis] + global_contrib + local_contrib
        return self.__maybe_squeeze(result, was_single)

    def reconstruct_full_series(self):
        """Returns (N_OBS, M, T)."""
        return self.reconstruct()

    def reconstruct_decomposed(self, time_idx=None):
        x_rows, l_scores = self.__resolve_indices(time_idx)
        was_single = isinstance(time_idx, int)
        base = self.__grand_mean + self.__asset_bias

        global_surf   = self.__global_contribution(x_rows)
        local_contrib = self.__local_contribution(l_scores)
        total = base[np.newaxis] + global_surf + local_contrib

        return {
            'base'   : base,
            'global' : self.__maybe_squeeze(global_surf,   was_single),
            'local'  : self.__maybe_squeeze(local_contrib, was_single),
            'total'  : self.__maybe_squeeze(total,         was_single),
        }

    def loading_surface(self, k):
        if k < 0 or k >= self.__n_global:
            raise IndexError(f"k={k} out of range [0, {self.__n_global})")
        return self.__B_j[k].reshape(self.__M, self.__T)

    # ─── accessors ──────────────────────────────────────────────────────

    def get_global_scores(self):    return self.__global_scores
    def get_B_j(self):              return self.__B_j
    def get_local_components(self): return self.__local_components
    def get_local_scores(self):     return self.__local_scores
    def get_residuals(self):        return self.__residuals
    def get_asset(self):            return self.__asset
    def n_global(self):             return self.__n_global
    def n_local(self):              return self.__n_local
    def n_obs(self):                return self.__n_obs

    def __str__(self):
        return (f"HfPCA_Reconstruction(asset={self.__asset}, "
                f"n_global={self.__n_global}, n_local={self.__n_local}, "
                f"n_obs={self.__n_obs})")

    def __repr__(self):
        return self.__str__()
    