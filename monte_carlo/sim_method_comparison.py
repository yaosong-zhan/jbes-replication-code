"""Method comparison for the JBES novelty claim.

Four designs on the same DGP (block-preserving engine conventions):
  (i)   Block-preserving CUSUM (proposed): shape-block PCA + price appended,
        W_t = (U_q' X_t, Y_t) in R^{q+1}, d = q+1 critical value.
  (ii)  Joint PCA CUSUM: PCA on the joint (p+1)-vector, reduced dimension q.
  (iii) Full-vector CUSUM: no dimension reduction, CUSUM on Z_t in R^{p+1}.
  (iv)  Separate shape / price tests: two marginal CUSUM tests at level
        alpha/2 each (Bonferroni), with a combined size.

Reported per design: size under H0 (iid / AR / MA / GARCH) and power for
shape-driven, price-driven and joint breaks at representative SNRs, plus the
attribution behaviour of the proposed block-preserving classification.

Output: Script/Results/sim_method_comparison.csv (+ log)
"""
import numpy as np, pandas as pd, os, sys, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulation_joint_change import (generate_data, cusum_test, shape_pca,
                                     build_detection_vec, get_critical_value,
                                     newey_west_lrv, toeplitz_corr)
from numpy import linalg as linalg

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Results')
DIM = 11
P = DIM - 1
Q = 5
ALPHA = 0.05
N_REPS = 2000


def cusum_statistic(W):
    """CUSUM statistic with LRV (used by all designs)."""
    T, d = W.shape
    cumsum = np.cumsum(W, axis=0)
    total = cumsum[-1, :]
    S = cumsum[:-1] - (np.arange(1, T) / T)[:, None] * total
    Om = newey_west_lrv(W)
    try:
        Oi = linalg.inv(Om)
    except linalg.LinAlgError:
        Oi = linalg.inv(Om + 1e-8 * np.eye(d))
    return float(np.max(np.sum((S @ Oi) * S, axis=1) / T)), d


# ---------- designs ----------
def design_block(Z):
    X, Y = Z[:, :-1], Z[:, -1]
    Uq, _, q = shape_pca(X, q_method='fixed', fixed_q=Q)
    W = build_detection_vec(X, Y, Uq)
    stat, d = cusum_statistic(W)
    return stat > get_critical_value(d, ALPHA)


def design_jointpca(Z):
    Zel = Z - Z.mean(0)
    cov = (Zel.T @ Zel) / (len(Z) - 1)
    ev, evec = linalg.eigh(cov)
    Vq = evec[:, np.argsort(ev)[::-1][:Q]]
    W = Z @ Vq
    stat, d = cusum_statistic(W)
    return stat > get_critical_value(d, ALPHA)


def design_full(Z):
    stat, d = cusum_statistic(Z[:, ])
    return stat > get_critical_value(d, ALPHA)


def design_separate(Z, alpha_split=ALPHA / 2):
    X, Y = Z[:, :-1], Z[:, -1]
    Uq, _, q = shape_pca(X, q_method='fixed', fixed_q=Q)
    Ws = X @ Uq                       # shape-only scores
    Wp = Y[:, None]                   # price-only
    s1, d1 = cusum_statistic(Ws)
    s2, d2 = cusum_statistic(Wp)
    return (s1 > get_critical_value(d1, alpha_split)) or (s2 > get_critical_value(d2, alpha_split))


def run():
    rows = []
    t0 = time.time()
    for T in [500, 1000, 2000]:
        for err, param in [('iid', 0), ('ar', 0.3), ('ma', 0.5), ('garch', 0)]:
            for bt, snr in [('none', 0.0), ('shape-driven', 0.1),
                            ('price-driven', 0.5), ('joint', 0.1)]:
                rej = {k: 0 for k in ['block', 'jointpca', 'full', 'separate']}
                for rep in range(N_REPS):
                    Z, _, _ = generate_data(T, DIM, 3, bt, 0.5, snr,
                                            error_type=err,
                                            ar_coeff=param if err == 'ar' else 0.0,
                                            ma_coeff=param if err == 'ma' else 0.5,
                                            garch_params=(0.15, 0.80) if err == 'garch' else None,
                                            rho_cov=0.5, seed=rep * 1000 + T + 7)
                    rej['block'] += design_block(Z)
                    rej['jointpca'] += design_jointpca(Z)
                    rej['full'] += design_full(Z)
                    rej['separate'] += design_separate(Z)
                row = {'T': T, 'error': err, 'break_type': bt, 'snr': snr,
                       **{k: v / N_REPS for k, v in rej.items()}}
                rows.append(row)
                print(f"T={T:4d} {err:5s} {bt:13s} snr={snr:.2f} -> "
                      f"block={row['block']:.3f} jointPCA={row['jointpca']:.3f} "
                      f"full={row['full']:.3f} sep={row['separate']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(R, 'sim_method_comparison.csv'), index=False)
    print(f"\nSaved sim_method_comparison.csv ({len(df)} rows) in {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    run()

