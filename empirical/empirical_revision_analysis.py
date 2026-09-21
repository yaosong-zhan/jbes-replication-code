"""
JBES Revision Analysis: clustered SEs, placebo checks, cutoff sensitivity.

Rebuilds the empirical tables for the JBES paper with:
  (1) Table 8 verification + two-way clustered standard errors (stock, event date)
  (2) Classification cutoff sensitivity (0.35/0.65) from per-break omega_mu
  (3) Placebo checks: weekday seasonality, within-window permutation, pseudo-event window
  (4) Alt-measures table verification

Inputs : Results/Earnings_Panel.csv, Results/Earnings_BreakDetails.json,
         Results/earnings_alt_measures.csv, ../Data/ReportDate.csv
Output : Results/revision_summary.txt
"""
import numpy as np
import pandas as pd
import json
import os
import sys
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(SCRIPT_DIR, 'Results')
PANEL_PATH = os.path.join(RESULTS, 'Earnings_Panel.csv')
DETAILS_PATH = os.path.join(RESULTS, 'Earnings_BreakDetails.json')
ALT_PATH = os.path.join(RESULTS, 'earnings_alt_measures.csv')
REPORT_PATH = os.path.join(SCRIPT_DIR, '..', 'Data', 'ReportDate.csv')

# Optional CLI overrides: python empirical_revision_analysis.py [panel.csv] [details.json] [tag]
if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
    PANEL_PATH = sys.argv[1]
if len(sys.argv) > 2 and os.path.exists(sys.argv[2]):
    DETAILS_PATH = sys.argv[2]
tag = sys.argv[3] if len(sys.argv) > 3 else ''

out = []
def log(*args):
    s = ' '.join(str(a) for a in args)
    print(s)
    out.append(s)

# ============================================================================
#  Two-way cluster-robust covariance (Cameron, Gelbach & Miller 2011)
# ============================================================================
def twoway_cluster_ols(y, x, g1, g2):
    """OLS of y on x (with constant) with SEs two-way clustered by g1, g2.
    Returns (coef, se_uncl, se_g1, se_twoway) for each coefficient."""
    X = np.column_stack([np.ones(len(y)), x])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    k = X.shape[1]
    meat0 = (X.T * resid) @ X / len(y)  # X'ee'X / n
    bread = np.linalg.inv(X.T @ X / len(y))

    def meat(g):
        g = pd.Series(g).values
        uniq = np.unique(g, return_inverse=True)[1]
        M = np.zeros((k, k))
        for j in range(uniq.max() + 1):
            sel = uniq == j
            if sel.sum() == 0:
                continue
            Xj = X[sel]
            sj = (resid[sel] * Xj.T).T
            M += sj.sum(axis=0)[:, None] @ sj.sum(axis=0)[None, :]
        return M / len(y)

    V_uncl = bread @ meat0 @ bread
    V_g1 = bread @ meat(g1) @ bread
    V_g2 = bread @ meat(g2) @ bread
    # intersection = stock x calendar-date, i.e. observation level -> robust
    inter = np.array([f'{a}|{b}' for a, b in zip(g1, g2)])
    V_int = bread @ meat(inter) @ bread
    V_tw = V_g1 + V_g2 - V_int
    se = lambda V: np.sqrt(np.diag(V) / len(y))
    return beta, se(V_uncl), se(V_g1), se(V_tw)


def stars(p):
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return ''

# ============================================================================
#  Load
# ============================================================================
log('=' * 78)
log('JBES REVISION ANALYSIS')
log('=' * 78)

df = pd.read_csv(PANEL_PATH)
log(f'Panel raw: {len(df)} rows, {df["stock"].nunique()} stocks')
# Some report dates are recorded twice in ReportDate.csv (e.g., the same date
# appears in two report-type columns), so the same event was processed twice.
# Deduplicate on (stock, report_date, event_day) for the analysis panel.
df = df.drop_duplicates(['stock', 'report_date', 'event_day'])
log(f'Panel clean: {len(df)} unique event-day obs, '
    f'{df[["stock", "report_date"]].drop_duplicates().shape[0]} unique events, '
    f'{df["stock"].nunique()} stocks, {df["market"].value_counts().to_dict()}')
log(f'Panel event days: {sorted(df["event_day"].unique())}')

with open(DETAILS_PATH) as f:
    details = json.load(f)
det = pd.DataFrame(details)
log(f'Break details: {len(det)} breaks; panel sum n_breaks = {df["n_breaks"].sum()}')

alt = pd.read_csv(ALT_PATH)
log(f'Alt-measures: {len(alt)} rows, {alt["stock"].nunique()} stocks')

rep = pd.read_csv(REPORT_PATH, encoding='utf-8-sig')

# ============================================================================
#  1. Verify Table 8 Panel A (daily break frequency)
# ============================================================================
log('')
log('--- Table 8 Panel A verification (mean n_breaks by event day) ---')
for mkt in ['STAR', 'ChiNext']:
    sub = df[df['market'] == mkt]
    for ed in [-3, -2, -1, 0, 1, 2, 3]:
        s = sub[sub['event_day'] == ed]
        log(f'  {mkt:>7s} day {ed:+d}: N={len(s):5d}  mean={s["n_breaks"].mean():.2f}')

# ============================================================================
#  2. Table 8 Panel B with clustered SEs
# ============================================================================
log('')
log('--- Table 8 Panel B: Pre vs Post with clustered SEs ---')
log('--- Regression: metric ~ Post, unit = stock-day, days {-3,-2,-1,0,+1} ---')

pre_days = [-3, -2, -1]
post_days = [0, 1]
metrics = [
    ('Break frequency', 'n_breaks'),
    ('Shape-driven (%)', 'shap_pct'),
    ('Price-driven (%)', 'pric_pct'),
    ('Joint (%)', 'join_pct'),
    ('Mean run length', 'mean_run'),
    ('Off-diagonal transition prob.', 'trans_off_diag'),
    ('Direction consistency (shape)', 'shap_dir'),
    ('Direction consistency (price)', 'pric_dir'),
    ('Direction consistency (joint)', 'join_dir'),
]

panelB = []
for mkt in ['STAR', 'ChiNext']:
    sub = df[(df['market'] == mkt) & (df['event_day'].isin(pre_days + post_days))].copy()
    sub['post'] = sub['event_day'].isin(post_days).astype(float)
    g1 = sub['stock'].astype(str).values
    g2 = sub['calendar_date'].astype(str).values
    log(f'  [{mkt}] N={len(sub)}, stocks={sub["stock"].nunique()}, '
        f'dates={sub["calendar_date"].nunique()}')
    for label, col in metrics:
        y = sub[col].values.astype(float)
        beta, se_u, se_1, se_tw = twoway_cluster_ols(y, sub['post'].values, g1, g2)
        diff = beta[1]
        # p-values (two-sided) from t-distribution
        t_tw = beta[1] / se_tw[1]
        dof = sub['stock'].nunique() - 1
        p_tw = 2 * (1 - stats.t.cdf(abs(t_tw), dof))
        t_g1 = beta[1] / se_1[1]
        p_g1 = 2 * (1 - stats.t.cdf(abs(t_g1), dof))
        # Welch (paper's original)
        pre_v = sub.loc[sub['post'] == 0, col].values.astype(float)
        post_v = sub.loc[sub['post'] == 1, col].values.astype(float)
        _, p_welch = stats.ttest_ind(pre_v, post_v, equal_var=False)
        pre_m, post_m = pre_v.mean(), post_v.mean()
        log(f'    {label:<32s} pre={pre_m:8.3f} post={post_m:8.3f} '
            f'diff={diff:+8.3f}  se(2way)={se_tw[1]:.4f}  '
            f'p_welch={p_welch:.3e}  p_stock={p_g1:.3e}  p_2way={p_tw:.3e}  '
            f'st_2way={stars(p_tw)}')
        panelB.append({
            'market': mkt, 'metric': label, 'pre': pre_m, 'post': post_m,
            'diff': diff, 'se_2way': se_tw[1], 'p_welch': p_welch,
            'p_stock': p_g1, 'p_2way': p_tw, 'n': len(sub),
            'n_stocks': sub['stock'].nunique(), 'n_dates': sub['calendar_date'].nunique(),
        })

# ============================================================================
#  3. Classification cutoff sensitivity (0.35/0.65 vs 0.4/0.6)
# ============================================================================
log('')
log('--- Cutoff sensitivity: reclassify breaks from omega_mu ---')

def classify(om, lo=0.4, hi=0.6):
    if om > hi:
        return 'shape-driven'
    if om < lo:
        return 'price-driven'
    return 'joint'

# Replicate compute_metrics from build_earnings_panel.py
def compute_metrics_types(types):
    n = len(types)
    if n == 0:
        return {}
    m = {'n_breaks': n,
         'shap_pct': types.count('shape-driven') / n,
         'pric_pct': types.count('price-driven') / n,
         'join_pct': types.count('joint') / n}
    if n >= 2:
        run_types = []
        cur_t, cur_len = types[0], 1
        for t_i in types[1:]:
            if t_i == cur_t:
                cur_len += 1
            else:
                run_types.append((cur_t, cur_len))
                cur_t, cur_len = t_i, 1
        run_types.append((cur_t, cur_len))
        m['mean_run'] = np.mean([r for _, r in run_types])
        # off-diagonal transition probability
        tm_from = {}
        for i in range(n - 1):
            tm_from.setdefault(types[i], []).append(types[i + 1])
        off = []
        for t1 in ['shape-driven', 'price-driven', 'joint']:
            if t1 in tm_from and len(tm_from[t1]) > 0:
                for t2 in ['shape-driven', 'price-driven', 'joint']:
                    if t1 != t2:
                        off.append(tm_from[t1].count(t2) / len(tm_from[t1]))
        m['trans_off_diag'] = np.mean(off) if off else 0
    else:
        m['mean_run'] = 1.0
        m['trans_off_diag'] = 0
    return m

# Build per-event-day reclassified type sequences (in original file order).
# NOTE: some events were processed twice (duplicate report-date columns), so
# the details file contains duplicate break sequences for the same event-day;
# drop them so each event-day contributes exactly one sequence. The grouping
# unit (stock, report_date, event_day) matches the main analysis panel.
det_sorted = det.drop_duplicates(
    ['stock', 'report_date', 'event_day', 'tau_hat']).sort_values(
    ['stock', 'report_date', 'calendar_date']).reset_index(drop=True)
cutoff_rows = []
for (stock, rd_key, ed), grp in det_sorted.groupby(['stock', 'report_date', 'event_day']):
    types = [classify(om, 0.4, 0.6) for om in grp['omega_mu'].values]
    mm = compute_metrics_types(types)
    cutoff_rows.append({'stock': stock, 'calendar_date': grp['calendar_date'].iloc[0],
                        **{k: mm[k] for k in ['n_breaks', 'shap_pct', 'pric_pct',
                                              'join_pct', 'mean_run', 'trans_off_diag']}})
d41 = pd.DataFrame(cutoff_rows)

cutoff_rows2 = []
for (stock, rd_key, ed), grp in det_sorted.groupby(['stock', 'report_date', 'event_day']):
    types = [classify(om, 0.35, 0.65) for om in grp['omega_mu'].values]
    mm = compute_metrics_types(types)
    cutoff_rows2.append({'stock': stock, 'calendar_date': grp['calendar_date'].iloc[0],
                         **{k: mm[k] for k in ['n_breaks', 'shap_pct', 'pric_pct',
                                               'join_pct', 'mean_run', 'trans_off_diag']}})
d35 = pd.DataFrame(cutoff_rows2)

# Reattach event_day / market (merge on stock x calendar date)
mrg = df[['stock', 'calendar_date', 'market', 'event_day']].drop_duplicates(
    ['stock', 'calendar_date'])
mrg['stock'] = mrg['stock'].astype(str)
for dset in [d41, d35]:
    dset['stock'] = dset['stock'].astype(str)
    dset['market'] = dset.merge(mrg, on=['stock', 'calendar_date'], how='left')['market']
    dset['event_day'] = dset.merge(mrg, on=['stock', 'calendar_date'], how='left')['event_day']

# Also recompute shape/pct from the ORIGINAL panel for comparison on identical rows
orig = df[['stock', 'calendar_date', 'n_breaks', 'shap_pct', 'pric_pct', 'join_pct',
           'mean_run', 'trans_off_diag']].copy()
for name, dset in [('cutoff 0.4/0.6', d41), ('cutoff 0.35/0.65', d35)]:
    log(f'')
    log(f'  {name}: rows={len(dset)}')
    for mkt in ['STAR', 'ChiNext']:
        s = dset[(dset['market'] == mkt) & (dset['event_day'].isin(pre_days + post_days))]
        s = s.copy()
        s['post'] = s['event_day'].isin(post_days).astype(float)
        g1 = s['stock'].astype(str).values
        g2 = s['calendar_date'].astype(str).values
        for label, col in [('Shape-driven (%)', 'shap_pct'), ('Price-driven (%)', 'pric_pct'),
                           ('Joint (%)', 'join_pct'), ('Mean run length', 'mean_run'),
                           ('Off-diagonal trans.', 'trans_off_diag')]:
            y = s[col].values.astype(float)
            beta, _, _, se_tw = twoway_cluster_ols(y, s['post'].values, g1, g2)
            t_tw = beta[1] / se_tw[1]
            dof = s['stock'].nunique() - 1
            p_tw = 2 * (1 - stats.t.cdf(abs(t_tw), dof))
            pre_m = s.loc[s['post'] == 0, col].mean()
            post_m = s.loc[s['post'] == 1, col].mean()
            log(f'    {mkt:>7s} {label:<22s} pre={pre_m:7.3f} post={post_m:7.3f} '
                f'diff={beta[1]:+8.3f} se_2way={se_tw[1]:.4f} p_2way={p_tw:.3e} {stars(p_tw)}')

# Composition shares under each cutoff (all days, pooled)
log('')
for name, dset in [('cutoff 0.4/0.6', d41), ('cutoff 0.35/0.65', d35)]:
    for mkt in ['STAR', 'ChiNext']:
        s = dset[dset['market'] == mkt]
        log(f'  {name}, {mkt}: shape={100*s["shap_pct"].mean():.1f}% '
            f'price={100*s["pric_pct"].mean():.1f}% joint={100*s["join_pct"].mean():.1f}%')

# omega_mu distribution
log('')
om = det['omega_mu'].values
for q in [0.05, 0.25, 0.5, 0.75, 0.95]:
    log(f'  omega_mu quantile {q:.2f}: {np.quantile(om, q):.3f}')
log(f'  omega_mu: share <0.35 = {(om < 0.35).mean():.3f}, '
    f'0.35-0.65 = {((om >= 0.35) & (om <= 0.65)).mean():.3f}, '
    f'>0.65 = {(om > 0.65).mean():.3f}')
log(f'  omega_mu: share <0.4 = {(om < 0.4).mean():.3f}, '
    f'0.4-0.6 = {((om >= 0.4) & (om <= 0.6)).mean():.3f}, '
    f'>0.6 = {(om > 0.6).mean():.3f}')

# ============================================================================
#  4. Placebo checks
# ============================================================================
log('')
log('--- Placebo 1: weekday seasonality ---')

# 4a. Weekday distribution of report dates (STAR + ChiNext)
rep_688 = rep[rep.iloc[:, 0].astype(str).str.match(r'688\d{3}\.SH')]
rep_300 = rep[rep.iloc[:, 0].astype(str).str.match(r'300\d{3}\.SZ')]
rep_all = pd.concat([rep_688, rep_300])
wd_counts = {}
for code in ['688', '300']:
    sub_rep = rep_688 if code == '688' else rep_300
    dts = []
    for col in range(3, 15):
        for v in sub_rep.iloc[:, col]:
            try:
                d = pd.to_datetime(str(v).strip())
                if pd.isna(d):
                    continue
                if pd.Timestamp('2022-01-01') <= d <= pd.Timestamp('2024-11-30'):
                    dts.append(d)
            except Exception:
                pass
    dts = pd.Series(dts)
    wd_counts[code] = dts.dt.dayofweek.value_counts(normalize=True).sort_index()
    log(f'  Report-date weekday distribution ({code}): '
        + ', '.join(f'{"MTWTFSS"[i]}={wd_counts[code].get(i, 0)*100:.1f}%' for i in range(5)))

# 4b. Break frequency by weekday on non-announcement days (event_day -3,-2,+2,+3)
df['dow'] = pd.to_datetime(df['calendar_date']).dt.dayofweek
non_ann = df[df['event_day'].isin([-3, -2, 2, 3])]
for mkt in ['STAR', 'ChiNext']:
    s = non_ann[non_ann['market'] == mkt]
    means = s.groupby('dow')['n_breaks'].mean()
    log(f'  {mkt} non-announcement break freq by weekday: '
        + ', '.join(f'{"MTWTFSS"[d]}={means.get(d, float("nan")):.2f}' for d in range(5)))
    # joint F-test of flatness: regress n_breaks on weekday dummies (cluster by stock)
    Xw = pd.get_dummies(s['dow'], prefix='w', drop_first=True).astype(float).values
    yw = s['n_breaks'].values.astype(float)
    beta, _, se_g1, _ = twoway_cluster_ols(yw, Xw, s['stock'].astype(str).values,
                                           s['calendar_date'].astype(str).values)
    b = beta[1:]
    Vb = None
    # joint Wald test with stock clustering (reuse meat from function via reconstruction)
    X = np.column_stack([np.ones(len(yw)), Xw])
    resid = yw - X @ beta
    k = X.shape[1]
    g = pd.Series(s['stock'].astype(str).values).values
    uniq = np.unique(g, return_inverse=True)[1]
    M = np.zeros((k, k))
    for j in range(uniq.max() + 1):
        sel = uniq == j
        if sel.sum():
            Xj = X[sel]
            sj = (resid[sel] * Xj.T).T
            M += sj.sum(axis=0)[:, None] @ sj.sum(axis=0)[None, :]
    bread = np.linalg.inv(X.T @ X / len(yw))
    V = bread @ (M / len(yw)) @ bread / len(yw)
    R = np.eye(len(b))
    W = b @ np.linalg.inv(R @ V[1:, 1:] @ R.T) @ b
    F = W / len(b)
    p_F = 1 - stats.f.cdf(F, len(b), s['stock'].nunique() - len(b) - 1)
    log(f'    Wald F={F:.2f}, p={p_F:.4f} (flatness of weekday pattern, '
        f'dof={len(b)}, clusters={s["stock"].nunique()})')

# 4c. Pre/post diff controlling for weekday dummies
for mkt in ['STAR', 'ChiNext']:
    s = df[(df['market'] == mkt) & (df['event_day'].isin(pre_days + post_days))].copy()
    s['post'] = s['event_day'].isin(post_days).astype(float)
    Xc = np.column_stack([np.ones(len(s)), s['post'].values,
                          pd.get_dummies(s['dow'], prefix='w', drop_first=True).astype(float).values])
    yc = s['n_breaks'].values.astype(float)
    beta, _, _, se_tw = twoway_cluster_ols(yc, Xc[:, 1:], s['stock'].astype(str).values,
                                           s['calendar_date'].astype(str).values)
    t_tw = beta[1] / se_tw[1]
    dof = s['stock'].nunique() - 1
    p_tw = 2 * (1 - stats.t.cdf(abs(t_tw), dof))
    log(f'  {mkt} pre/post diff of break freq WITH weekday FE: {beta[1]:+.3f}, '
        f'se_2way={se_tw[1]:.4f}, p={p_tw:.3e} {stars(p_tw)}')

# 4d. Permutation placebo (within-window relabeling)
log('')
log('--- Placebo 2: within-window permutation (random pseudo-event days) ---')
rng = np.random.default_rng(20260831)
B = 2000
# event-level diffs: mean(n_breaks) on post days minus mean on pre days (per day basis)
ev = df[df['event_day'].isin([-3, -2, -1, 0, 1, 2, 3])]
actual = {}
pseudo_stats = {}
for mkt in ['STAR', 'ChiNext']:
    sub = ev[ev['market'] == mkt]
    events = sub.groupby(['stock', 'report_date'])
    diffs = []
    for (st, rd), g in events:
        g = g.set_index('event_day')['n_breaks'].reindex([-3, -2, -1, 0, 1])
        pre = g.loc[[-3, -2, -1]].values
        post = g.loc[[0, 1]].values
        if not np.isnan(pre).any() and not np.isnan(post).any():
            diffs.append(post.mean() - pre.mean())
    actual[mkt] = np.mean(diffs)
    log(f'  {mkt}: events with full 5-day windows = {len(diffs)}, '
        f'actual mean diff = {actual[mkt]:+.4f}')
    # permutation: within each event, draw pseudo-post (2 of 7 days), pseudo-pre (3 of remaining 5)
    # Restrict to events with all 7 window days present so the actual and
    # permutation statistics are computed on the same event set.
    all_days = [-3, -2, -1, 0, 1, 2, 3]
    ev_full = []
    for (st, rd), g in ev[ev['market'] == mkt].groupby(['stock', 'report_date']):
        if len(g) == 7:
            g = g.set_index('event_day')['n_breaks'].reindex(all_days)
            ev_full.append(g.values.astype(float))
    E = np.array(ev_full)  # (n_events, 7)
    log(f'  {mkt}: events with all 7 window days = {len(E)}')
    n_ev = len(E)
    if n_ev > 0:
        # actual statistic on this subset (same definition as full sample)
        a_sub = np.mean(E[:, 3:5].mean(axis=1) - E[:, 0:3].mean(axis=1))
        log(f'  {mkt}: actual mean diff on 7-day subset = {a_sub:+.4f}')
        cols = np.arange(7)
        stats_perm = np.empty(B)
        for b in range(B):
            perm = rng.permutation(cols)
            pst_c = perm[:2]
            pre_c = perm[2:5]
            post_v = E[:, pst_c].mean(axis=1)
            pre_v = E[:, pre_c].mean(axis=1)
            stats_perm[b] = np.mean(post_v - pre_v)
    else:
        stats_perm = np.array([np.nan])
        a_sub = np.nan
    pval = (stats_perm >= a_sub).mean()
    log(f'  {mkt}: permutation mean = {stats_perm.mean():+.4f}, '
        f'95% interval = [{np.quantile(stats_perm, 0.025):+.4f}, '
        f'{np.quantile(stats_perm, 0.975):+.4f}], '
        f'pseudo p-value = {pval:.4f}, actual in tail = {a_sub > np.quantile(stats_perm, 0.95)}')
    pseudo_stats[mkt] = (stats_perm, pval)

# 4e. Pseudo-event window: +2/+3 as "post", {-1,0,+1} as "pre"
log('')
log('--- Placebo 3: pseudo-event window (+2,+3 vs -1,0,+1) ---')
for mkt in ['STAR', 'ChiNext']:
    sub = df[(df['market'] == mkt) & (df['event_day'].isin([-1, 0, 1, 2, 3]))].copy()
    sub['post'] = sub['event_day'].isin([2, 3]).astype(float)
    y = sub['n_breaks'].values.astype(float)
    beta, _, _, se_tw = twoway_cluster_ols(y, sub['post'].values,
                                           sub['stock'].astype(str).values,
                                           sub['calendar_date'].astype(str).values)
    t_tw = beta[1] / se_tw[1]
    dof = sub['stock'].nunique() - 1
    p_tw = 2 * (1 - stats.t.cdf(abs(t_tw), dof))
    pre_m = sub.loc[sub['post'] == 0, 'n_breaks'].mean()
    post_m = sub.loc[sub['post'] == 1, 'n_breaks'].mean()
    log(f'  {mkt}: pseudo-pre(-1,0,+1)={pre_m:.2f} vs pseudo-post(+2,+3)={post_m:.2f}, '
        f'diff={beta[1]:+.3f}, se_2way={se_tw[1]:.4f}, p={p_tw:.3e} {stars(p_tw)}')

# ============================================================================
#  5. Alt measures table verification
# ============================================================================
log('')
log('--- Alt measures table ---')
log(f'  Total rows: {len(alt)}; unique stock-days: {alt[["stock","calendar_date"]].drop_duplicates().shape[0]}')
for ed in sorted(alt['event_day'].unique()):
    s = alt[alt['event_day'] == ed]
    log(f'  day {ed:+d}: N={len(s):4d}  rv={s["rv_daily"].mean():.4f}  '
        f'spread_rel={s["avg_spread_rel"].mean():.4f}')

# correlation of break freq and volatility (stock-day level, common sample)
for mkt in ['STAR', 'ChiNext']:
    s = alt[alt['market'] == mkt]
    r = s[['n_breaks', 'rv_daily']].corr().iloc[0, 1]
    log(f'  corr(n_breaks, rv) {mkt}: r={r:.3f}, N={len(s)}')

# ============================================================================
#  Save
# ============================================================================
with open(os.path.join(RESULTS, f"revision_summary{tag}.txt"), "w", encoding="utf-8") as f:
    f.write('\n'.join(out))
log('')
log('Saved to Results/revision_summary.txt')

