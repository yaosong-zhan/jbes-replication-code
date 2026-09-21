"""
JBES-level benchmark simulations:
1. Comparison with univariate CUSUM on price
2. Multiple break detection via binary segmentation
3. Break magnitude estimation (omega_mu with CIs)

"""
import numpy as np, pandas as pd, os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulation_joint_change import generate_data, run_full_pipeline

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Results')
DIM = 11
ALPHA = 0.05
N_REPS = 2000


def univariate_cusum_test(x, alpha=0.05):
    """Univariate CUSUM on 1-d array x (price dimension)."""
    T = len(x)
    total = np.sum(x)
    S = np.cumsum(x) - (np.arange(1, T + 1) / T) * total
    acv = np.correlate(x - x.mean(), x - x.mean(), mode='full') / T
    acv = acv[T-1:]
    bw = min(int(np.floor(4 * (T / 100) ** (2/9))), int(np.floor(T ** (1/3))))
    lrv = acv[0] + 2 * sum((1 - h/(bw+1)) * acv[h] for h in range(1, min(bw+1, T)))
    lrv = max(lrv, 1e-10)
    stat = np.max(np.abs(S)) / (np.sqrt(T) * np.sqrt(lrv))
    cv = {0.10: 1.224, 0.05: 1.358, 0.01: 1.628}[alpha]
    return stat > cv, stat


def run_benchmark_simulation(T, alpha=0.05, n_reps=2000):
    """Compare joint test vs univariate CUSUM on price for size and power."""
    results = []
    configs = []

    # Size configurations
    for error_type in ['iid', 'ar', 'ma', 'garch']:
        configs.append({'break_type': 'none', 'snr': 0.0, 'lambda': 0.50,
                        'bt': 'none', 'error_type': error_type,
                        'ar_coeff': 0.3 if error_type == 'ar' else 0.0})

    # Power configurations
    for bt in ['shape-driven', 'price-driven', 'joint']:
        for snr in ([0.05, 0.10, 0.20] if bt in ('shape-driven', 'joint') else [0.25, 0.50, 1.00]):
            configs.append({'break_type': bt, 'snr': snr, 'lambda': 0.50,
                            'bt': bt, 'error_type': 'ar', 'ar_coeff': 0.3})

    for cfg in configs:
        reject_our, reject_uni = 0, 0
        for rep in range(n_reps):
            Z, tau_true, _ = generate_data(
                T, DIM, 3, cfg['break_type'], cfg['lambda'],
                cfg['snr'], ar_coeff=cfg['ar_coeff'],
                error_type=cfg['error_type'], seed=rep*500 + T)

            # Joint test with PCA (our method)
            r = run_full_pipeline(Z, alpha, q_method='variance', var_threshold=0.85)
            if r['reject']:
                reject_our += 1

            # Univariate CUSUM on price (last column)
            rej_u, _ = univariate_cusum_test(Z[:, -1], alpha)
            if rej_u:
                reject_uni += 1

        if cfg['bt'] == 'none':
            row = {'T': T, 'method': 'SIZE', 'break_type': cfg['error_type'],
                   'snr': 0, 'lambda': cfg['lambda'],
                   'Joint_Test': reject_our / n_reps,
                   'Univariate': reject_uni / n_reps,
                   'n_reps': n_reps}
        else:
            row = {'T': T, 'method': 'POWER', 'break_type': cfg['bt'],
                   'snr': cfg['snr'], 'lambda': cfg['lambda'],
                   'Joint_Test': reject_our / n_reps,
                   'Univariate': reject_uni / n_reps,
                   'n_reps': n_reps}
        results.append(row)
        print(f"  [benchmark] T={T:4d} {cfg['bt']:>13s} snr={cfg['snr']:.2f} "
              f"鈫?Joint={row['Joint_Test']:.3f} Uni={row['Univariate']:.3f}")

    return pd.DataFrame(results)


def run_multi_break_simulation(T, alpha=0.05, n_reps=2000):
    """Test binary segmentation with 2 breaks at lambda={0.30, 0.70}."""
    results = []
    configs = [
        ('shape-driven', 0.05), ('shape-driven', 0.10),
        ('price-driven', 0.25), ('price-driven', 0.50),
        ('joint', 0.05), ('joint', 0.10),
    ]

    for bt, snr in configs:
        n_exact = 0
        n_at_least_1 = 0
        n_over = 0
        tau1_errs, tau2_errs = [], []

        for rep in range(n_reps):
            tau1 = int(T * 0.30)
            tau2 = int(T * 0.70)
            Z, _, _ = generate_data(T, DIM, 3, 'none', 0.0, 0.0,
                                     ar_coeff=0.3, seed=rep*500+T)

            delta1 = np.zeros(DIM)
            if bt == 'shape-driven':
                delta1[:-1] = snr
            elif bt == 'price-driven':
                delta1[-1] = snr
            else:
                delta1[:] = snr
            Z[tau1:] += delta1
            Z[tau2:] += delta1 * 0.8

            r = run_full_pipeline(Z, alpha, q_method='variance',
                                  var_threshold=0.85,
                                  binary_seg=True, min_seg_len=30)

            n_breaks = r.get('num_breaks', 0)
            if n_breaks >= 1:
                n_at_least_1 += 1
            if n_breaks == 2:
                n_exact += 1
            elif n_breaks > 2:
                n_over += 1

            if n_breaks >= 1:
                taus = r.get('tau_hats', [])
                if taus:
                    for t_detected in taus:
                        err1 = abs(t_detected - tau1) / T
                        err2 = abs(t_detected - tau2) / T
                        best = min(err1, err2)
                        if err1 < err2:
                            tau1_errs.append(best)
                        else:
                            tau2_errs.append(best)

        results.append({
            'T': T, 'break_type': bt, 'snr': snr,
            'p_exact_2': n_exact / n_reps,
            'p_at_least_1': n_at_least_1 / n_reps,
            'p_over': n_over / n_reps,
            'mean_tau1_err': np.mean(tau1_errs) if tau1_errs else np.nan,
            'mean_tau2_err': np.mean(tau2_errs) if tau2_errs else np.nan,
        })
        print(f"  [multi] T={T:4d} {bt:>13s} snr={snr:.2f} "
              f"鈫?exact2={n_exact/n_reps:.3f} >=1={n_at_least_1/n_reps:.3f}")

    return pd.DataFrame(results)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- 1. Benchmark vs Univariate ----
    print("=" * 65)
    print("  1. BENCHMARK: JOINT TEST vs UNIVARIATE CUSUM")
    print("=" * 65)
    all_bench = []
    for T in [500, 1000, 2000]:
        print(f"\n  --- T = {T} ---")
        df_b = run_benchmark_simulation(T, ALPHA, N_REPS)
        all_bench.append(df_b)
    df_bench = pd.concat(all_bench, ignore_index=True)
    df_bench.to_csv(os.path.join(RESULTS_DIR, 'sim_benchmark.csv'), index=False)
    print(f"\n  Saved to sim_benchmark.csv ({len(df_bench)} rows)")

    # ---- 2. Multiple break detection ----
    print("\n" + "=" * 65)
    print("  2. MULTIPLE BREAK DETECTION (binary segmentation)")
    print("=" * 65)
    all_multi = []
    for T in [500, 1000, 2000]:
        print(f"\n  --- T = {T} ---")
        df_m = run_multi_break_simulation(T, ALPHA, N_REPS)
        all_multi.append(df_m)
    df_multi = pd.concat(all_multi, ignore_index=True)
    df_multi.to_csv(os.path.join(RESULTS_DIR, 'sim_multibreak.csv'), index=False)
    print(f"\n  Saved to sim_multibreak.csv ({len(df_multi)} rows)")

    # ---- 3. Summary ----
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print("\n  Benchmark (size, T=2000):")
    sub = df_bench[(df_bench['T']==2000) & (df_bench['method']=='SIZE')]
    for _, r in sub.iterrows():
        print(f"    {r['break_type']:>8s}: Joint={r['Joint_Test']:.3f} "
              f"Uni={r['Univariate']:.3f}")
    print("\n  Benchmark (power, T=2000, shape-driven):")
    sub = df_bench[(df_bench['T']==2000) & (df_bench['method']=='POWER')]
    for _, r in sub.iterrows():
        print(f"    {r['break_type']:>13s} snr={r['snr']:.2f}: "
              f"Joint={r['Joint_Test']:.3f} Uni={r['Univariate']:.3f}")
    print("\n  Multiple breaks (T=2000):")
    sub = df_multi[df_multi['T']==2000]
    for _, r in sub.iterrows():
        print(f"    {r['break_type']:>13s} snr={r['snr']:.2f}: "
              f"P(exact=2)={r['p_exact_2']:.3f} P(>=1)={r['p_at_least_1']:.3f}")
    print("\nDone.")


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

