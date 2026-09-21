"""
Simulation with cross-sectional correlation in the DGP (factor structure).
Demonstrates robustness of PCA-based method when shape dimensions are correlated.

DGP: AR(1) over time + AR(1) across shape dimensions (more realistic LOB structure).
"""
import numpy as np, pandas as pd, os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulation_joint_change import generate_data, run_full_pipeline, get_critical_value

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Results')
DIM = 11
N_REPS = 2000
ALPHA = 0.05


def generate_factor_data(T, dim=11, break_type='none', break_location=0.5, snr=0.5,
                          ar_coeff=0.3, rho=0.7, seed=None):
    """
    Generate data with cross-sectional AR(1) correlation across shape dimensions.
    Shape dimensions: correlated (factor structure)
    Price dimension: independent
    """
    if seed is not None:
        np.random.seed(seed)

    p = dim - 1  # shape dimensions
    tau = int(np.floor(T * break_location))
    sigma = 0.5

    # Cross-sectional covariance: AR(1) across shape dimensions
    Sigma_shape = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            Sigma_shape[i, j] = sigma**2 * rho**abs(i - j)

    # Price dimension variance
    sigma_price = sigma**2

    # Full innovation covariance
    Sigma = np.zeros((dim, dim))
    Sigma[:p, :p] = Sigma_shape
    Sigma[p, p] = sigma_price

    # Cholesky for generating innovations
    L = np.linalg.cholesky(Sigma + 1e-8 * np.eye(dim))

    # Generate AR(1) over time
    Z = np.zeros((T, dim))
    eta = np.random.randn(T, dim)
    for t in range(T):
        if t == 0:
            Z[t] = L @ eta[t]
        else:
            Z[t] = ar_coeff * Z[t-1] + np.sqrt(1 - ar_coeff**2) * (L @ eta[t])

    # Add break
    delta = np.zeros(dim)
    if break_type == 'shape-driven':
        delta[:p] = snr
    elif break_type == 'price-driven':
        delta[-1] = snr
    elif break_type == 'joint':
        delta[:] = snr
    if break_type != 'none' and tau < T:
        Z[tau:] += delta

    return Z, tau, delta


def run_simulation(T, rho=0.7, n_reps=2000, alpha=0.05):
    """Run size and power for factor DGP."""
    rows = []

    # Size
    print(f"\n  Size (T={T}, rho={rho}):")
    for error_type in ['iid', 'ar']:
        ac = 0.0 if error_type == 'iid' else 0.3
        rejections = 0
        for rep in range(n_reps):
            Z, _, _ = generate_factor_data(T, DIM, 'none', 0.5, 0.0, ar_coeff=ac, rho=rho, seed=rep*500+T)
            r = run_full_pipeline(Z, alpha, q_method='variance', var_threshold=0.85)
            if r['reject']:
                rejections += 1
        rows.append({'T': T, 'rho': rho, 'DGP': f'{error_type}_factor', 'metric': 'size',
                     'value': rejections/n_reps, 'n_reps': n_reps})
        print(f"    {error_type}: size={rejections/n_reps:.3f}")

    # Power by type
    print(f"  Power (T={T}, rho={rho}):")
    for bt in ['shape-driven', 'price-driven', 'joint']:
        snr = 0.05 if bt in ('shape-driven', 'joint') else 0.25
        rejections = 0
        omegas = []
        for rep in range(n_reps):
            Z, tau_true, _ = generate_factor_data(T, DIM, bt, 0.5, snr, ar_coeff=0.3, rho=rho, seed=rep*1000+T)
            r = run_full_pipeline(Z, alpha, q_method='variance', var_threshold=0.85)
            if r['reject']:
                rejections += 1
                omegas.append(r.get('omega_mu', 0))
        power = rejections / n_reps
        mean_omega = np.mean(omegas) if omegas else np.nan
        rows.append({'T': T, 'rho': rho, 'DGP': f'factor', 'metric': 'power',
                     'break_type': bt, 'snr': snr, 'value': power, 'omega': mean_omega})
        print(f"    {bt:>15s} (snr={snr:.2f}): power={power:.3f} omega={mean_omega:.3f}")

    return pd.DataFrame(rows)


def main():
    print("=" * 65)
    print("  FACTOR DGP SIMULATION (cross-sectional AR(1) correlation)")
    print("=" * 65)

    all_rows = []
    for rho in [0.0, 0.5, 0.7, 0.9]:
        print(f"\n{'='*50}")
        print(f"  rho = {rho}")
        print(f"{'='*50}")
        for T in [500, 1000, 2000]:
            df = run_simulation(T, rho=rho, n_reps=N_REPS)
            all_rows.append(df)

    df_all = pd.concat(all_rows, ignore_index=True)
    out_path = os.path.join(RESULTS_DIR, 'sim_factor_dgp.csv')
    df_all.to_csv(out_path, index=False)

    # Summary
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    for rho in [0.0, 0.5, 0.7, 0.9]:
        sub = df_all[(df_all['rho'] == rho) & (df_all['metric'] == 'size')]
        print(f"\n  rho={rho:.1f} Size: ", end="")
        for _, r in sub.iterrows():
            print(f"{r['DGP']}={r['value']:.3f} ", end="")
        print()
        sub = df_all[(df_all['rho'] == rho) & (df_all['metric'] == 'power') & (df_all['T'] == 2000)]
        for _, r in sub.iterrows():
            print(f"  {r['break_type']:>15s}: power={r['value']:.3f} omega={r.get('omega',0):.3f}")

    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--nreps', type=int, default=None)
    args = ap.parse_args()
    if args.nreps:
        globals()['N_REPS'] = args.nreps
    if args.quick:
        globals()['N_REPS'] = 50
        print(f"[Quick mode: n_reps={N_REPS}]")
    main()

