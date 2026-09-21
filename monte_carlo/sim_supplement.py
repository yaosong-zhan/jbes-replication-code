"""
Supplemental simulations for JBES manuscript:
1. Break location estimation accuracy (|蟿虃-蟿*|/T)
2. 蠅_渭 separation across SNR levels
3. Break magnitude estimation quality
"""
import numpy as np, pandas as pd, os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulation_joint_change import generate_data, run_full_pipeline, get_critical_value

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Results')
DIM = 11; N_REPS = 2000; ALPHA = 0.05

rows = []

# =====================================================================
#  1. Break Location Accuracy
# =====================================================================
print("=" * 60)
print("  BREAK LOCATION ACCURACY")
print("=" * 60)
for T in [100, 250, 500, 1000, 2000]:
    for bt in ['shape-driven', 'price-driven', 'joint']:
        snr = 0.05 if bt in ('shape-driven', 'joint') else 0.25
        for lam in [0.25, 0.50, 0.75]:
            tau_errs = []
            for rep in range(N_REPS):
                Z, tau_true, _ = generate_data(T, DIM, 3, bt, lam, snr, seed=rep*1000)
                r = run_full_pipeline(Z, ALPHA, q_method='fixed', fixed_q=5)
                if r.get('reject') and bt != 'none':
                    tau_errs.append(abs(r['tau_hat'] - tau_true) / T)
            if tau_errs:
                rows.append({'T': T, 'break_type': bt, 'snr': snr, 'lambda': lam,
                             'mean_tau_err': np.mean(tau_errs),
                             'median_tau_err': np.median(tau_errs),
                             'pct_within_005': np.mean(np.array(tau_errs) < 0.05),
                             'pct_within_010': np.mean(np.array(tau_errs) < 0.10)})
                print(f"  T={T:4d} {bt:>13s} 位={lam:.2f} 鈫?mean_err={rows[-1]['mean_tau_err']:.4f} "
                      f"med={rows[-1]['median_tau_err']:.4f} "
                      f"within5%={rows[-1]['pct_within_005']:.2f}")

# =====================================================================
#  2. 蠅_渭 Separation (already have, just extract key stats)
# =====================================================================
print("\n" + "=" * 60)
print("  蠅_渭 SEPARATION (from sim_omega.csv)")
print("=" * 60)
df_om = pd.read_csv(os.path.join(RESULTS_DIR, 'sim_omega.csv'))
omega_data = df_om[df_om['metric'] == 'omega'].copy()
for _, r in omega_data.iterrows():
    print(f"  SNR={r['snr']:.2f}  {r['true_type']:>13s}: 蠅_渭={r['mean_omega']:.3f}  acc={r['accuracy']:.2%}")

# =====================================================================
#  3. Break Magnitude Estimation (未虃_尉)
# =====================================================================
print("\n" + "=" * 60)
print("  BREAK MAGNITUDE ESTIMATION")
print("=" * 60)
for T in [500, 1000, 2000]:
    for bt in ['shape-driven', 'price-driven', 'joint']:
        snr = 0.10 if bt in ('shape-driven', 'joint') else 0.50
        deltas = []
        for rep in range(N_REPS):
            Z, tau_true, delta_true = generate_data(T, DIM, 3, bt, 0.50, snr, seed=rep*500)
            r = run_full_pipeline(Z, ALPHA, q_method='fixed', fixed_q=5)
            if r.get('reject'):
                deltas.append(r.get('omega_mu', 0))
        if deltas:
            rows.append({'T': T, 'break_type': bt, 'snr': snr,
                         'metric': 'omega_mu',
                         'mean_omega': np.mean(deltas),
                         'std_omega': np.std(deltas),
                         'q025': np.percentile(deltas, 2.5),
                         'q975': np.percentile(deltas, 97.5)})
            print(f"  T={T:4d} {bt:>13s} snr={snr:.2f} 鈫?蠅_渭={np.mean(deltas):.3f} "
                  f"[{np.percentile(deltas,2.5):.3f}, {np.percentile(deltas,97.5):.3f}]")

# Save
df = pd.DataFrame(rows)
df.to_csv(os.path.join(RESULTS_DIR, 'sim_supplement.csv'), index=False)
print(f"\nSaved to sim_supplement.csv ({len(df)} rows)")

