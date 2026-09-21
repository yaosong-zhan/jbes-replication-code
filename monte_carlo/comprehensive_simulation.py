"""
Comprehensive Monte Carlo simulation for JBES manuscript.

Follows the structure of Horvath et al. (2023, JBES) with:
  - Size under H0 for various T, 伪, error structures
  - Power under HA for various T, break type, SNR, 位, error
  - Classification accuracy (confusion matrix)
  - 蠅_渭 separation diagnostics

Saves all results to CSV files.
"""
import numpy as np, pandas as pd, os, sys, warnings, itertools, time
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulation_joint_change import generate_data, run_full_pipeline, run_classification_mc

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Results')
os.makedirs(RESULTS_DIR, exist_ok=True)

DIM = 11  # 10 shape + 1 price
N_REPS = 2000
ALPHA = 0.05


def run_size(configs):
    """Size under H0. configs: list of (T, alpha, error_type, param)."""
    rows = []
    total = len(configs)
    for idx, (T, alpha, err, param) in enumerate(configs):
        rej = 0
        for rep in range(N_REPS):
            Z, _, _ = generate_data(T, DIM, 3, 'none', 0.5, 0.0,
                                    error_type=err, ar_coeff=param if err == 'ar' else 0.0,
                                    ma_coeff=param if err == 'ma' else 0.5,
                                    garch_params=(0.15, 0.80) if err == 'garch' else None,
                                    seed=rep * 100 + idx)
            r = run_full_pipeline(Z, alpha, q_method='fixed', fixed_q=5)
            if r['reject']:
                rej += 1
        s = rej / N_REPS
        se = np.sqrt(s * (1 - s) / N_REPS)
        rows.append({'T': T, 'alpha': alpha, 'error': err, 'param': str(param),
                     'size': s, 'se': se})
        print(f"  SIZE T={T:4d} 伪={alpha:.2f} {err:5s} 鈫?{s:.4f} (se={se:.4f})", flush=True)
    return pd.DataFrame(rows)


def run_power(configs):
    """Power under HA. configs: list of (T, bt, snr, lam, err, param)."""
    snr_map = {'shape-driven': [0.05, 0.10, 0.20],
               'price-driven': [0.25, 0.50, 1.00],
               'joint': [0.05, 0.10, 0.20]}
    rows = []
    total = len(configs)
    for idx, (T, bt, snr, lam, err, param) in enumerate(configs):
        rej = 0
        for rep in range(N_REPS):
            Z, _, _ = generate_data(T, DIM, 3, bt, lam, snr,
                                    error_type=err, ar_coeff=param if err == 'ar' else 0.0,
                                    seed=rep * 1000 + idx)
            r = run_full_pipeline(Z, ALPHA, q_method='fixed', fixed_q=5)
            if r['reject']:
                rej += 1
        p = rej / N_REPS
        rows.append({'T': T, 'break_type': bt, 'snr': snr, 'lambda': lam,
                     'error': err, 'param': str(param), 'power': p, 'n_rej': rej})
        print(f"  POWER T={T:4d} {bt:>13s} snr={snr:.2f} 位={lam:.2f} {err:5s} 鈫?{p:.4f}", flush=True)
    return pd.DataFrame(rows)


def run_classification(configs):
    """Classification accuracy. configs: list of (T, snr, err, param)."""
    rows = []
    for T, snr, err, param in configs:
        r = run_classification_mc(T, DIM, 3, snr, N_REPS, ALPHA,
                                  norm_type='l2_corrected',
                                  q_method='fixed', fixed_q=5,
                                  ar_coeff=param if err == 'ar' else 0.0,
                                  error_type=err)
        for bt in ['shape-driven', 'price-driven', 'joint']:
            rows.append({'T': T, 'snr': snr, 'error': err, 'param': str(param),
                         'true_type': bt,
                         'accuracy': r.get(f'acc_{bt}', np.nan),
                         'n_detected': r.get(f'n_{bt}', 0),
                         'detection_rate': r.get('detection_rate', {}).get(bt, np.nan),
                         'uncond_success': r.get('uncond_success', {}).get(bt, np.nan),
                         'omega_mu': r['mean_omega_mu'].get(bt, np.nan)})
        print(f"  CLASSIF T={T} snr={snr:.2f} {err:5s} 鈫?"
              f"shape={rows[-3]['accuracy']:.3f} price={rows[-2]['accuracy']:.3f} joint={rows[-1]['accuracy']:.3f}",
              flush=True)
    return pd.DataFrame(rows)


def run_all(skip_size=False):
    t0 = time.time()

    # =====================================================================
    #  1. SIZE
    # =====================================================================
    print("=" * 70)
    print("  SIZE STUDY")
    print("=" * 70)
    size_configs = []
    for T in [100, 250, 500, 1000, 2000, 5000]:
        for alpha in [0.01, 0.05, 0.10]:
            for err, param in [('iid', 0), ('ar', 0.0), ('ar', 0.3), ('ar', 0.5),
                               ('ma', 0.5), ('garch', 0)]:
                size_configs.append((T, alpha, err, param))
    if not skip_size:
        df_size = run_size(size_configs)
        df_size.to_csv(os.path.join(RESULTS_DIR, 'sim_size.csv'), index=False)
        print(f"  Saved sim_size.csv ({len(df_size)} rows)\n")

    # =====================================================================
    #  2. POWER
    # =====================================================================
    print("=" * 70)
    print("  POWER STUDY")
    print("=" * 70)
    power_configs = []
    for T in [100, 250, 500, 1000, 2000]:
        for bt in ['shape-driven', 'price-driven', 'joint']:
            for snr in (snr_map := {'shape-driven': [0.05, 0.10, 0.20],
                                     'price-driven': [0.25, 0.50, 1.00],
                                     'joint': [0.05, 0.10, 0.20]})[bt]:
                for lam in [0.25, 0.50, 0.75]:
                    for err, param in [('iid', 0), ('ar', 0.3)]:
                        power_configs.append((T, bt, snr, lam, err, param))
    df_power = run_power(power_configs)
    df_power.to_csv(os.path.join(RESULTS_DIR, 'sim_power.csv'), index=False)
    print(f"  Saved sim_power.csv ({len(df_power)} rows)\n")

    # =====================================================================
    #  3. CLASSIFICATION
    # =====================================================================
    print("=" * 70)
    print("  CLASSIFICATION STUDY")
    print("=" * 70)
    classif_configs = []
    for T in [500, 1000, 2000]:
        for snr in [0.10, 0.20, 0.50]:
            for err, param in [('iid', 0), ('ar', 0.3)]:
                classif_configs.append((T, snr, err, param))
    df_class = run_classification(classif_configs)
    df_class.to_csv(os.path.join(RESULTS_DIR, 'sim_classification.csv'), index=False)
    print(f"  Saved sim_classification.csv ({len(df_class)} rows)\n")

    # =====================================================================
    #  4. OMEGA_MU SEPARATION (detailed for manuscript figure/table)
    # =====================================================================
    print("=" * 70)
    print("  OMEGA_MU SEPARATION (T=2000)")
    print("=" * 70)
    omega_rows = []
    for snr in [0.10, 0.20, 0.50]:
        r = run_classification_mc(2000, DIM, 3, snr, N_REPS, ALPHA,
                                  norm_type='l2_corrected',
                                  q_method='fixed', fixed_q=5)
        for bt in ['shape-driven', 'price-driven', 'joint']:
            omega_rows.append({
                'T': 2000, 'snr': snr, 'metric': 'omega', 'true_type': bt,
                'mean_omega': r['mean_omega_mu'].get(bt, np.nan),
                'accuracy': r.get(f'acc_{bt}', np.nan),
            })
        conf = r['confusion_matrix']
        for t1 in ['shape-driven', 'price-driven', 'joint']:
            for t2 in ['shape-driven', 'price-driven', 'joint']:
                omega_rows.append({
                    'T': 2000, 'snr': snr, 'metric': 'confusion',
                    'true_type': t1, 'classified_as': t2,
                    'count': conf[t1][t2],
                })
        shape_om = r['mean_omega_mu']['shape-driven']
        price_om = r['mean_omega_mu']['price-driven']
        joint_om = r['mean_omega_mu']['joint']
        print(f"  SNR={snr:.2f}: shape_蠅={shape_om:.3f} "
              f"price_蠅={price_om:.3f} joint_蠅={joint_om:.3f}", flush=True)

    pd.DataFrame(omega_rows).to_csv(os.path.join(RESULTS_DIR, 'sim_omega.csv'), index=False)
    print(f"  Saved sim_omega.csv ({len(omega_rows)} rows)\n")

    # Summary
    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  DONE in {elapsed/60:.1f} minutes")
    print(f"{'=' * 70}")
    print(f"  Files:")
    for f in ['sim_size.csv', 'sim_power.csv', 'sim_classification.csv', 'sim_omega.csv']:
        path = os.path.join(RESULTS_DIR, f)
        if os.path.exists(path):
            print(f"    {path} ({len(pd.read_csv(path))} rows)")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--nreps', type=int, default=2000)
    ap.add_argument('--skip-size', action='store_true')
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    N_REPS = args.nreps
    if args.quick:
        N_REPS = 50
        print(f"[QUICK MODE: {N_REPS} reps]")
    run_all(skip_size=args.skip_size)

