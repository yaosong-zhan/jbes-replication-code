"""
Robustness simulations for the JBES manuscript (Table 6, 'Robustness and
q sensitivity'):

  Panel A: MA(1) errors (theta = 0.5)
           - empirical size at alpha = 0.05, T in {500, 1000, 2000}
           - power for shape-driven breaks (snr = 0.10), same T
  Panel B: GARCH(1,1) errors (alpha = 0.15, beta = 0.80)
           - same cells as Panel A
  Panel C: sensitivity to the number of principal components q
           - power for shape-driven breaks (T = 1000, snr = 0.10,
             lambda = 0.5) for q in {2, ..., 6}, IID and AR(0.3) errors

Saves Results/sim_robust.csv. All runs use 2000 replications and the fixed-q
block-preserving pipeline without whitening, matching the other simulation scripts.

"""
import numpy as np, pandas as pd, os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulation_joint_change import generate_data, run_full_pipeline

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Results')
os.makedirs(RESULTS_DIR, exist_ok=True)
DIM = 11
N_REPS = 2000
ALPHA = 0.05

rows = []

# ============================================================================
#  Panels A & B: size and power under MA(1) and GARCH(1,1) errors
# ============================================================================
for err, garch_params, label in [('ma', None, 'MA(1)'), ('garch', (0.15, 0.80), 'GARCH')]:
    for T in [500, 1000, 2000]:
        rej_size = 0
        for rep in range(N_REPS):
            Z, _, _ = generate_data(T, DIM, 3, 'none', 0.5, 0.0,
                                    error_type=err, ar_coeff=0.0, ma_coeff=0.5,
                                    garch_params=garch_params, seed=rep * 100 + T)
            r = run_full_pipeline(Z, ALPHA, q_method='fixed', fixed_q=5)
            if r['reject']:
                rej_size += 1
        rows.append({'metric': 'size', 'error': err, 'T': T, 'q': 5,
                     'value': rej_size / N_REPS})
        print(f"  SIZE {label} T={T}: {rej_size/N_REPS:.4f}", flush=True)

        rej_pow = 0
        for rep in range(N_REPS):
            Z, _, _ = generate_data(T, DIM, 3, 'shape-driven', 0.5, 0.10,
                                    error_type=err, ar_coeff=0.0, ma_coeff=0.5,
                                    garch_params=garch_params, seed=rep * 1000 + T)
            r = run_full_pipeline(Z, ALPHA, q_method='fixed', fixed_q=5)
            if r['reject']:
                rej_pow += 1
        rows.append({'metric': 'power', 'error': err, 'T': T, 'q': 5,
                     'value': rej_pow / N_REPS})
        print(f"  POWER {label} T={T}: {rej_pow/N_REPS:.4f}", flush=True)

# ============================================================================
#  Panel C: q sensitivity (shape-driven, T = 1000, snr = 0.10)
# ============================================================================
for q in [2, 3, 4, 5, 6]:
    for err, ar_coeff in [('iid', 0.0), ('ar', 0.3)]:
        rej = 0
        for rep in range(N_REPS):
            Z, _, _ = generate_data(1000, DIM, 3, 'shape-driven', 0.5, 0.10,
                                    error_type=err, ar_coeff=ar_coeff,
                                    seed=rep * 100 + q)
            r = run_full_pipeline(Z, ALPHA, q_method='fixed', fixed_q=q)
            if r['reject']:
                rej += 1
        rows.append({'metric': 'q_power', 'error': err, 'T': 1000, 'q': q,
                     'value': rej / N_REPS})
        print(f"  Q_POWER q={q} {err:3s}: {rej/N_REPS:.4f}", flush=True)

df = pd.DataFrame(rows)
out_path = os.path.join(RESULTS_DIR, 'sim_robust.csv')
df.to_csv(out_path, index=False)
print(f"\nSaved to sim_robust.csv ({len(df)} rows)")

