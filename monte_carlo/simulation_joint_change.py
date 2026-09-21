"""
Block-preserving PCA-CUSUM engine for the JBES manuscript revision.

Design (per ImplementPlan.md):
  - Z_t = (X_t', Y_t)', X_t in R^p shape block, Y_t in R price increment.
  - PCA is applied to the SHAPE BLOCK ONLY; the price component is never
    discarded by PCA.
  - Reduced detection vector W_t = (U_q' X_t, Y_t) in R^{q+1}; all test
    dimensions and critical values use d = q + 1.
  - PCA is used only for detection. Classification estimates the FULL
    break vector directly from pre- and post-break sample means of X and Y.
  - No Cholesky whitening. The long-run covariance matrix is used directly
    with a Bartlett kernel and bandwidth floor(T^{1/3}) matching the theory.
  - p is fixed in the asymptotics; q is fixed.

The previous joint-PCA engine is preserved in git history (commit b0a8662)
and as simulation_joint_change_jointpca.py in this directory.

"""
import numpy as np
from typing import Optional, Tuple
from numpy import linalg as linalg
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
#  Long-Run Covariance (Bartlett kernel, bandwidth = floor(T^{1/3}))
# ============================================================================

def newey_west_lrv(X: np.ndarray, bandwidth: Optional[int] = None) -> np.ndarray:
    """Newey-West long-run variance with the Bartlett kernel.

    Bandwidth matches the theory: bandwidth = floor(T^{1/3}).
    The sensitivity analysis may use 0.5 / 1 / 2 times this value.
    """
    T, _ = X.shape
    if bandwidth is None:
        bandwidth = max(1, int(np.floor(T ** (1 / 3))))

    X_dm = X - X.mean(axis=0)
    Gamma_0 = (X_dm.T @ X_dm) / T
    Omega = Gamma_0.copy()

    for h in range(1, bandwidth + 1):
        Gh = (X_dm[h:].T @ X_dm[:-h]) / T
        w = 1 - h / (bandwidth + 1)
        Omega += w * (Gh + Gh.T)

    return (Omega + Omega.T) / 2


# ============================================================================
#  Critical Values for sup||B^0_d(v)||^2  (d = q + 1)
# ============================================================================

# Simulated with N = 20000 grid + M = 50000 reps.
# The 2.5% quantiles for the separate tests are simulated directly.
CRITICAL_VALUES = {
    1: {0.10: 1.49, 0.05: 1.84, 0.025: 2.17508900832957, 0.01: 2.63},
    2: {0.10: 2.11, 0.05: 2.49, 0.01: 3.37},
    3: {0.10: 2.62, 0.05: 3.06, 0.01: 4.04},
    4: {0.10: 3.06, 0.05: 3.54, 0.01: 4.60},
    5: {0.10: 3.46, 0.05: 3.97, 0.025: 4.43149016326379, 0.01: 5.10},
    6: {0.10: 3.83, 0.05: 4.37, 0.01: 5.57},
    7: {0.10: 4.18, 0.05: 4.74, 0.01: 6.00},
    8: {0.10: 4.50, 0.05: 5.08, 0.01: 6.40},
    9: {0.10: 4.80, 0.05: 5.40, 0.01: 6.78},
    10: {0.10: 5.10, 0.05: 5.70, 0.01: 7.13},
    # calibrate_full_vector_cv.py: 50,000 paths, 20,000 grid, seed 20260916.
    11: {0.10: 5.785398673345621, 0.05: 6.388606030395973,
         0.025: 6.949642031613014, 0.01: 7.681818805423302},
}


def get_critical_value(dim: int, alpha: float = 0.05) -> float:
    """Critical value of sup||B^0_dim(v)||^2 for dimension dim (= q+1).

    Quantiles are simulated at the tabulated levels. The 0.025 quantiles
    used by Bonferroni-split tests are calibrated directly.
    """
    keys = sorted(CRITICAL_VALUES.keys())

    def quantile_at(dk, a):
        alphas = sorted(CRITICAL_VALUES[dk])
        if a in CRITICAL_VALUES[dk]:
            return CRITICAL_VALUES[dk][a]
        if a <= alphas[0]:
            return CRITICAL_VALUES[dk][alphas[0]]
        if a >= alphas[-1]:
            return CRITICAL_VALUES[dk][alphas[-1]]
        alo = max([x for x in alphas if x < a])
        ahi = min([x for x in alphas if x > a])
        w = (a - alo) / (ahi - alo)
        return (1 - w) * CRITICAL_VALUES[dk][alo] + w * CRITICAL_VALUES[dk][ahi]

    if dim in CRITICAL_VALUES:
        return quantile_at(dim, alpha)
    if dim <= keys[0]:
        return quantile_at(keys[0], alpha)
    if dim > keys[-1]:
        raise ValueError(f'No calibrated critical values for dimension {dim}')
    lo = max([x for x in keys if x < dim])
    hi = min([x for x in keys if x > dim])
    w = (dim - lo) / (hi - lo)
    return (1 - w) * quantile_at(lo, alpha) + w * quantile_at(hi, alpha)


# ============================================================================
#  Shape-block PCA (no whitening)
# ============================================================================

def shape_pca(X: np.ndarray, q_method: str = 'variance',
              var_threshold: float = 0.85, max_q: Optional[int] = None,
              fixed_q: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, int]:
    """PCA on the shape block only (fixed p). Returns U_q, scores, q."""
    T, p = X.shape
    X_dm = X - X.mean(axis=0)
    cov = (X_dm.T @ X_dm) / (T - 1)
    eigvals, eigvecs = linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals_sorted = eigvals[idx]

    if q_method == 'fixed':
        q = min(fixed_q, p) if fixed_q is not None else p
    elif q_method == 'kaiser':
        mean_eig = np.mean(eigvals_sorted)
        q = max(int(np.sum(eigvals_sorted > mean_eig)), 1)
    elif q_method == 'ratio':
        ratios = eigvals_sorted[:-1] / eigvals_sorted[1:]
        q = int(np.argmax(ratios) + 1)
    else:  # 'variance'
        cum_var = np.cumsum(eigvals_sorted) / np.sum(eigvals_sorted)
        q = int(np.searchsorted(cum_var, var_threshold) + 1)
        if max_q is not None:
            q = min(q, max_q)
    q = min(q, p)

    U_q = eigvecs[:, idx][:, :q]
    # Sign convention: largest-magnitude entry of each column positive.
    for j in range(q):
        k = int(np.argmax(np.abs(U_q[:, j])))
        if U_q[k, j] < 0:
            U_q[:, j] = -U_q[:, j]
    return U_q, X @ U_q, q


def build_detection_vec(X: np.ndarray, Y: np.ndarray,
                        U_q: np.ndarray) -> np.ndarray:
    """W_t = (U_q' X_t, Y_t) in R^{q+1}. The price component is never
    discarded by PCA."""
    return np.column_stack([X @ U_q, Y])


# ============================================================================
#  Multivariate CUSUM test on W (dimension d = q + 1)
# ============================================================================

def cusum_test(W: np.ndarray, alpha: float = 0.05,
               bandwidth: Optional[int] = None) -> Tuple[float, bool, int]:
    """CUSUM test on the reduced detection vector W (dim d = q+1)."""
    T, d = W.shape
    cumsum = np.cumsum(W, axis=0)
    total = cumsum[-1, :]
    S_k = cumsum[:-1] - (np.arange(1, T) / T)[:, None] * total

    Omega_hat = newey_west_lrv(W, bandwidth)
    try:
        Omega_inv = linalg.inv(Omega_hat)
    except linalg.LinAlgError:
        Omega_hat = Omega_hat + 1e-8 * np.eye(d)
        Omega_inv = linalg.inv(Omega_hat)

    Q_k = np.sum((S_k @ Omega_inv) * S_k, axis=1) / T
    T_stat = np.max(Q_k)
    k_star = int(np.argmax(Q_k) + 1)
    reject = T_stat > get_critical_value(d, alpha)
    return T_stat, reject, k_star


# ============================================================================
#  Full break-vector estimation and classification (direct, no back-projection)
# ============================================================================

def estimate_full_break(X: np.ndarray, Y: np.ndarray,
                        tau: int) -> Optional[np.ndarray]:
    """Full (p+1)-dimensional break vector from pre/post sample means."""
    if tau <= 0 or tau >= len(X):
        return None
    dX = X[tau:].mean(axis=0) - X[:tau].mean(axis=0)
    dY = float(Y[tau:].mean(axis=0) - Y[:tau].mean(axis=0))
    return np.concatenate([dX, [dY]])


def classify_break_full(delta: np.ndarray) -> Tuple[float, float, str]:
    """omega = a/(a+|delta_Y|) with a = p^{-1/2} ||delta_X||_2 on the FULL
    break vector. Thresholds: >0.6 shape, [0.4,0.6] joint, <0.4 price."""
    shape_dim = len(delta) - 1
    shape_norm = np.sqrt(np.sum(delta[:-1] ** 2) / shape_dim) if shape_dim > 0 else 0.0
    price_norm = np.abs(delta[-1])
    total = shape_norm + price_norm
    if total == 0:
        return 0.0, 0.0, 'joint'
    omega_mu = shape_norm / total
    omega_theta = 1 - omega_mu
    if omega_mu > 0.6:
        btype = 'shape-driven'
    elif omega_mu < 0.4:
        btype = 'price-driven'
    else:
        btype = 'joint'
    return omega_mu, omega_theta, btype


# ============================================================================
#  Binary segmentation (per-segment CUSUM, LRV and bandwidth per segment)
# ============================================================================

def binary_segmentation(W: np.ndarray, alpha: float = 0.05,
                        min_len: int = 30, max_breaks: int = 10,
                        bandwidth: Optional[int] = None) -> list:
    """Recursive binary segmentation on W. Each recursion computes its own
    CUSUM statistic, LRV estimator and bandwidth on the segment."""
    breaks = []

    def _recursive(start: int, end: int, depth: int = 0):
        if depth >= max_breaks or end - start < min_len:
            return
        seg = W[start:end]
        _, reject, tau_hat = cusum_test(seg, alpha, bandwidth)
        if not reject or tau_hat <= 0 or tau_hat >= end - start:
            return
        abs_pos = start + tau_hat
        if not any(abs(abs_pos - b) < min_len // 2 for b in breaks):
            breaks.append(abs_pos)
            _recursive(start, abs_pos, depth + 1)
            _recursive(abs_pos, end, depth + 1)

    _recursive(0, len(W))
    return sorted(breaks)


# ============================================================================
#  Full Pipeline
# ============================================================================

def run_full_pipeline(Z: np.ndarray, alpha: float = 0.05,
                      q_method: str = 'variance',
                      var_threshold: float = 0.85,
                      max_q: Optional[int] = None,
                      fixed_q: Optional[int] = None,
                      binary_seg: bool = False,
                      min_seg_len: int = 30,
                      max_breaks: int = 10) -> dict:
    """Shape-block PCA -> reduced W -> CUSUM -> full-vector classification.

    Z : (T, p+1); the last column is the price component Y and the first p
    columns are the shape block X (fixed p)."""
    X = Z[:, :-1]
    Y = Z[:, -1]
    U_q, _, q = shape_pca(X, q_method=q_method, var_threshold=var_threshold,
                          max_q=max_q, fixed_q=fixed_q)
    W = build_detection_vec(X, Y, U_q)
    d = W.shape[1]  # q + 1

    if binary_seg:
        tau_hats = binary_segmentation(W, alpha, min_seg_len, max_breaks)
        result = {'T': len(Z), 'q': q, 'd': d, 'reject': len(tau_hats) > 0,
                  'tau_hats': tau_hats, 'num_breaks': len(tau_hats)}
        details = []
        for th in tau_hats:
            delta = estimate_full_break(X, Y, th)
            if delta is None:
                continue
            wm, wt, bt = classify_break_full(delta)
            details.append({
                'tau_hat': int(th),
                'omega_mu': wm,
                'omega_theta': wt,
                'break_type': bt,
                'delta_price': float(delta[-1]),
                'delta_shape_mean': float(np.mean(delta[:-1])),
                'price_dir': int(np.sign(delta[-1])),
                'shape_dir': int(np.sign(np.mean(delta[:-1]))),
            })
        result['break_details'] = details
        return result

    T_stat, reject, tau_hat = cusum_test(W, alpha)
    result = {'T': len(Z), 'q': q, 'd': d, 'T_stat': T_stat,
              'reject': reject, 'tau_hat': tau_hat}
    if reject and tau_hat > 0:
        delta = estimate_full_break(X, Y, tau_hat)
        if delta is not None:
            wm, wt, bt = classify_break_full(delta)
            result.update(omega_mu=wm, omega_theta=wt, break_type=bt,
                          delta_price=float(delta[-1]),
                          delta_shape_mean=float(np.mean(delta[:-1])),
                          price_dir=int(np.sign(delta[-1])),
                          shape_dir=int(np.sign(np.mean(delta[:-1]))))
    return result


# ============================================================================
#  Data generation with an eigengap covariance (fixed p)
# ============================================================================

def toeplitz_corr(p: int, rho: float) -> np.ndarray:
    """AR(1) cross-sectional correlation matrix, R_ij = rho^{|i-j|}."""
    from scipy.linalg import toeplitz
    r = rho ** np.arange(p)
    return toeplitz(r)


def generate_data(T: int, dim: int = 11, q_true: int = 3, break_type: str = 'none',
                  break_location: float = 0.5, snr: float = 2.0,
                  ar_coeff: float = 0.3, error_type: str = 'ar',
                  ma_coeff: float = 0.5, garch_params: tuple = None,
                  rho_cov: float = 0.5, q_pc: int = 5,
                  seed: Optional[int] = None
                  ) -> Tuple[np.ndarray, int, np.ndarray]:
    """Mean-shift DGP with an eigengap shape covariance (fixed p).

    Z_t = eps_t + delta * 1{t > tau*}; the last column is the price
    increment. The shape innovation covariance is sigma^2 (1-phi^2) R with
    R_ij = rho^{|i-j|}, whose eigenvalues satisfy an eigengap. Break
    designs: 'shape-driven'/'joint' (aligned with the leading shape PC),
    'shape-aligned' (uniform over retained PCs 1..q_pc),
    'shape-partial' (confined to discarded PCs q_pc+1..p),
    'price-driven', 'none'.
    """
    if seed is not None:
        np.random.seed(seed)
    tau = int(np.floor(T * break_location))
    sigma = 0.5
    p = dim - 1

    # Whiten innovations per error type, then apply cross-sectional R.
    if error_type == 'iid':
        eta = np.random.randn(T, dim)
        scale = sigma
    elif error_type == 'ma':
        theta = ma_coeff if ma_coeff is not None else 0.5
        eta = np.random.randn(T + 1, dim)
        z_ma = eta[1:] + theta * eta[:-1]
        scale = sigma * np.sqrt(1 / (1 + theta ** 2))
        eta = z_ma
    elif error_type == 'garch':
        omega = 0.05 * sigma ** 2
        a_g, b_g = garch_params if garch_params else (0.15, 0.80)
        h = np.zeros(T)
        e = np.zeros(T)
        for t in range(T):
            if t == 0:
                h[t] = omega / (1 - a_g - b_g)
            else:
                h[t] = omega + a_g * e[t - 1] ** 2 + b_g * h[t - 1]
            e[t] = np.sqrt(h[t]) * np.random.randn()
        eta = e[:, None] * np.random.randn(T, dim)
        scale = 1.0
    else:  # 'ar'
        phi = ar_coeff
        eta = np.random.randn(T, dim)
        scale = sigma * np.sqrt(1 - phi ** 2)

    R = toeplitz_corr(p, rho_cov)
    L = linalg.cholesky(R)
    eta = np.column_stack([eta[:, :p] @ L.T, eta[:, -1]]) * scale

    if error_type == 'ar' and ar_coeff != 0.0:
        Z = np.empty_like(eta)
        Z[0] = eta[0] / np.sqrt(1 - ar_coeff ** 2)
        for t in range(1, T):
            Z[t] = ar_coeff * Z[t - 1] + eta[t]
    else:
        Z = eta

    delta = np.zeros(dim)
    if break_type == 'none':
        pass
    elif break_type == 'price-driven':
        delta[-1] = snr
    elif break_type in ('shape-driven', 'joint'):
        # Uniform per-coordinate magnitude: the per-dimension shape norm
        # equals snr, so omega* = 1 (shape) and 0.5 (joint) by construction.
        delta[:-1] = snr
        if break_type == 'joint':
            delta[-1] = snr
    elif break_type == 'shape-aligned':
        # Confined to the retained PCs 1..q_pc (detectable by construction).
        eigvals, eigvecs = linalg.eigh(R)
        idx = np.argsort(eigvals)[::-1][:q_pc]
        delta[:-1] = snr * eigvecs[:, idx].mean(axis=1)
    elif break_type == 'shape-partial':
        # Confined to the discarded PCs q_pc+1..p (documented limitation).
        eigvals, eigvecs = linalg.eigh(R)
        idx = np.argsort(eigvals)[::-1][q_pc:]
        delta[:-1] = snr * eigvecs[:, idx].mean(axis=1)
    else:
        raise ValueError(f'unknown break_type {break_type}')

    Z[tau:] += delta
    return Z, tau, delta


# ============================================================================
#  Monte Carlo classification (detection + conditional classification)
# ============================================================================

def run_classification_mc(T: int, dim: int, q_true: int, snr: float,
                          n_reps: int = 2000, alpha: float = 0.05,
                          norm_type: str = 'l2_corrected',
                          **kwargs) -> dict:
    """Classification accuracy in the single-break setting (fixed q).

    Returns the detection rate per type, the confusion matrix over detected
    breaks, conditional per-type accuracy, and the unconditional
    classification success rate.
    """
    if norm_type not in ('l2_corrected', 'l2'):
        raise ValueError(f'Unsupported classification norm: {norm_type}')
    true_types = ['shape-driven', 'price-driven', 'joint']
    conf = {t: {'shape-driven': 0, 'price-driven': 0, 'joint': 0} for t in true_types}
    omegas = {t: [] for t in true_types}
    n_detected = {t: 0 for t in true_types}
    n_rep_type = {t: 0 for t in true_types}
    uncond_correct = {t: 0 for t in true_types}

    for rep in range(n_reps):
        for bt in true_types:
            seed_val = rep * 10000 + int(snr * 100) + sum(ord(c) for c in bt)
            ar_coeff = kwargs.get('ar_coeff', 0.0)
            error_type_in = kwargs.get('error_type', 'ar')
            rho_cov = kwargs.get('rho_cov', 0.5)
            Z, _, _ = generate_data(T, dim, q_true, bt, 0.5, snr,
                                    ar_coeff=ar_coeff, error_type=error_type_in,
                                    rho_cov=rho_cov, q_pc=kwargs.get('q_pc', 5),
                                    seed=seed_val)
            fq = kwargs.get('fixed_q', 5)
            result = run_full_pipeline(Z, alpha, q_method='fixed', fixed_q=fq)
            n_rep_type[bt] += 1
            if result['reject']:
                n_detected[bt] += 1
                if norm_type == 'l2':
                    delta = estimate_full_break(Z[:, :-1], Z[:, -1], result['tau_hat'])
                    shape_norm = np.linalg.norm(delta[:-1])
                    omega = shape_norm / (shape_norm + abs(delta[-1]))
                    result['omega_mu'] = float(omega)
                    result['break_type'] = ('shape-driven' if omega > .6 else
                                            'price-driven' if omega < .4 else 'joint')
                classified = result.get('break_type', 'joint')
                conf[bt][classified] += 1
                omegas[bt].append(result.get('omega_mu', 0))
                if classified == bt:
                    uncond_correct[bt] += 1

    d = {'T': T, 'dim': dim, 'snr': snr, 'n_reps': n_reps,
         'confusion_matrix': conf,
         'mean_omega_mu': {t: np.mean(omegas[t]) if omegas[t] else np.nan for t in true_types},
         'detection_rate': {t: n_detected[t] / n_rep_type[t] for t in true_types},
         'uncond_success': {t: uncond_correct[t] / n_rep_type[t] for t in true_types}}
    for t in true_types:
        total = sum(conf[t].values())
        d[f'acc_{t}'] = conf[t][t] / total if total > 0 else np.nan
        d[f'n_{t}'] = total
    return d


def run_monte_carlo(T: int, dim: int, q_true: int, break_type: str,
                    snr: float = 0.5, n_reps: int = 2000, alpha: float = 0.05,
                    error_type: str = 'ar', ar_coeff: float = 0.3,
                    **kwargs):
    """Single-break power/location/classification study (fixed q = 5)."""
    rej = 0
    tau_errs = []
    type_correct = 0
    n_det = 0
    for rep in range(n_reps):
        Z, tau_true, delta_true = generate_data(
            T, dim, q_true, break_type, 0.5, snr,
            error_type=error_type, ar_coeff=ar_coeff,
            rho_cov=kwargs.get('rho_cov', 0.5),
            seed=rep * 1000 + T)
        r = run_full_pipeline(Z, alpha, q_method='fixed', fixed_q=5)
        if r['reject']:
            rej += 1
            n_det += 1
            tau_errs.append(abs(r['tau_hat'] - tau_true) / T)
            if r.get('break_type') == break_type:
                type_correct += 1
    return {'power': rej / n_reps,
            'mean_tau_err': np.mean(tau_errs) if tau_errs else np.nan,
            'median_tau_err': np.median(tau_errs) if tau_errs else np.nan,
            'classif_cond': type_correct / n_det if n_det else np.nan}


def run_full_simulation(n_reps: int = 2000, alpha: float = 0.05):
    raise NotImplementedError(
        'The monolithic simulation driver was removed in the revision; '
        'run comprehensive_simulation.py (updated separately) instead.')

