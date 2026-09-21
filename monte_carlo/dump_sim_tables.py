"""Dump fresh simulation numbers in the paper's table cell order."""
import pandas as pd, numpy as np
from pathlib import Path
R = str(Path(__file__).resolve().parent / 'Results') + '/' 

print('=' * 72)
print('1. SIZE TABLE  (tab:sim_size)')
print('=' * 72)
sz = pd.read_csv(R + 'sim_size.csv')
def szv(T, a, err, par='0'):
    r = sz[(sz['T'] == T) & (sz['alpha'] == a) & (sz['error'] == err)
           & (sz['param'].astype(str) == str(float(par)))]
    return float(r['size'].iloc[0]) if len(r) else np.nan
for T in [100, 500, 1000, 2000, 5000]:
    for a in [0.01, 0.05, 0.10]:
        print(f'T={T:4d} a={a:.2f} | IID={szv(T,a,"iid","0"):.3f} AR(.3)={szv(T,a,"ar","0.3"):.3f} '
              f'AR(.5)={szv(T,a,"ar","0.5"):.3f} MA(.5)={szv(T,a,"ma","0.5"):.3f} '
              f'GARCH={szv(T,a,"garch","0"):.3f}')

print()
print('=' * 72)
print('2. POWER TABLE  (tab:sim_power)  lambda=0.50')
print('=' * 72)
pw = pd.read_csv(R + 'sim_power.csv')
def pwv(T, bt, snr, err, lam=0.5):
    r = pw[(pw['T'] == T) & (pw['break_type'] == bt) & (pw['snr'] == snr)
           & (pw['error'] == err) & (pw['lambda'] == lam)]
    return float(r['power'].iloc[0]) if len(r) else np.nan
for T in [100, 500, 1000, 2000]:
    for bt, lo, hi in [('shape-driven', 0.05, 0.20), ('price-driven', 0.25, 1.00), ('joint', 0.05, 0.20)]:
        print(f'T={T:4d} {bt:13s} | IID lo={pwv(T,bt,lo,"iid"):.2f} hi={pwv(T,bt,hi,"iid"):.2f} '
              f'| AR lo={pwv(T,bt,lo,"ar"):.2f} hi={pwv(T,bt,hi,"ar"):.2f}')
# lambda 0.25 / 0.75 shape at T=2000
print('  T=2000 shape lambda sweep:',
      [f'lam={l}: {pwv(2000,"shape-driven",0.05,"iid",l):.2f}/{pwv(2000,"shape-driven",0.05,"ar",l):.2f}' for l in [0.25, 0.50, 0.75]])

print()
print('=' * 72)
print('3. CLASSIFICATION  (tab:sim_classif)  T=2000')
print('=' * 72)
cl = pd.read_csv(R + 'sim_classification.csv')
c2000 = cl[cl['T'] == 2000]
for snr in [0.10, 0.20, 0.50]:
    row = c2000[(c2000['snr'] == snr) & (c2000['error'] == 'iid')].drop_duplicates('true_type').set_index('true_type')
    print(f'snr={snr}: shape acc={row.loc["shape-driven","accuracy"]:.3f} om={row.loc["shape-driven","omega_mu"]:.3f} | '
          f'price acc={row.loc["price-driven","accuracy"]:.3f} om={row.loc["price-driven","omega_mu"]:.3f} | '
          f'joint acc={row.loc["joint","accuracy"]:.3f} om={row.loc["joint","omega_mu"]:.3f}')
om = pd.read_csv(R + 'sim_omega.csv')
for snr in [0.10, 0.20, 0.50]:
    conf = om[(om['metric'] == 'confusion') & (om['snr'] == snr)]
    print(f'  confusion snr={snr}:')
    for t1 in ['shape-driven', 'price-driven', 'joint']:
        counts = {t2: int(conf[(conf['true_type'] == t1) & (conf['classified_as'] == t2)]['count'].sum())
                  for t2 in ['shape-driven', 'price-driven', 'joint']}
        print(f'    {t1:13s} -> {counts}')

print()
print('=' * 72)
print('4. MAGNITUDE  (tab:sim_magnitude)')
print('=' * 72)
sp = pd.read_csv(R + 'sim_supplement.csv')
mg = sp[sp['metric'] == 'omega_mu']
for T in [500, 1000, 2000]:
    for bt in ['shape-driven', 'price-driven', 'joint']:
        r = mg[(mg['T'] == T) & (mg['break_type'] == bt)]
        if len(r):
            print(f'T={T} {bt:13s}: om={r["mean_omega"].iloc[0]:.3f} '
                  f'[{r["q025"].iloc[0]:.3f}, {r["q975"].iloc[0]:.3f}]')

print()
print('=' * 72)
print('5. MULTI-BREAK  (tab:sim_multibreak)')
print('=' * 72)
mb = pd.read_csv(R + 'sim_multibreak.csv')
for T in [500, 1000, 2000]:
    for _, r in mb[(mb['T'] == T)].iterrows():
        print(f'T={T} {r["break_type"]:13s} snr={r["snr"]:.2f}: P(K=2)={r["p_exact_2"]:.3f} '
              f'P(K>=1)={r["p_at_least_1"]:.3f} err1={r["mean_tau1_err"]:.3f} err2={r["mean_tau2_err"]:.3f} '
              f'P(over)={r["p_over"]:.3f}')

print()
print('=' * 72)
print('6. ROBUST  (tab:robust): MA/GARCH size from sim_size; power/q from sim_robust.csv')
print('=' * 72)
for err, par in [('ma', '0.5'), ('garch', '0')]:
    sizes = [szv(T, 0.05, err, par) for T in [500, 1000, 2000]]
    print(f'{err}: size T=500/1000/2000 = {[f"{s:.3f}" for s in sizes]}')
import os
if os.path.exists(R + 'sim_robust.csv'):
    rb = pd.read_csv(R + 'sim_robust.csv')
    for err in ['ma', 'garch']:
        pows = rb[(rb['metric'] == 'power') & (rb['error'] == err)].sort_values('T')
        print(f'{err}: power = {list(pows["value"].round(3))}')
    qp = rb[rb['metric'] == 'q_power']
    for err in ['iid', 'ar']:
        qq = qp[qp['error'] == err].sort_values('q')
        print(f'q_power {err}: q=2..6 = {list(qq["value"].round(3))}')
else:
    print('sim_robust.csv not ready yet')

