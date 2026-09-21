"""
Table 13 (main paper): standard market quality measures around earnings
announcements, computed on the stock-days of the analysis panel for which
intraday price series are available.

The realised-volatility and spread series (earnings_alt_measures.csv) are
computed from raw tick data by compute_alt_measures.py. This script merges
them onto the deduplicated event-day analysis panel and produces the table
reported in the paper (Table 13), together with the correlation between
daily break frequency and realised volatility.

Usage:
    python table_alt_measures.py [panel.csv] [alt_measures.csv]

Inputs:
    panel.csv         event-day panel produced by build_earnings_panel.py
                      (Earnings_Panel.csv; duplicates across report-date
                      records are removed inside this script)
    alt_measures.csv  stock-day file with columns stock, calendar_date,
                      market, n_breaks, rv_daily, avg_spread_rel
Output (stdout):
    daily means of realised volatility and average relative spread by event
    day, per-market correlations corr(n_breaks, rv_daily), and a LaTeX-ready
    version of Table 13.
"""
import sys
import os
import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PANEL = os.path.join(script_dir, '..', '..', '..', 'Script', 'Results',
                             'Earnings_Panel.csv')
DEFAULT_ALT = os.path.join(script_dir, '..', '..', '..', 'Script', 'Results',
                           'earnings_alt_measures.csv')

panel_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PANEL
alt_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_ALT

# --- load and deduplicate the event-day panel ---------------------------
pn = pd.read_csv(panel_path)
pn = pn.drop_duplicates(['stock', 'report_date', 'event_day'])
pn['stock'] = pn['stock'].astype(str)
pn['calendar_date'] = pn['calendar_date'].astype(str)

alt = pd.read_csv(alt_path)
alt['stock'] = alt['stock'].astype(str)
alt['calendar_date'] = alt['calendar_date'].astype(str)

m = pn.merge(alt[['stock', 'calendar_date', 'rv_daily', 'avg_spread_rel']],
             on=['stock', 'calendar_date'], how='inner')
print(f'merged stock-days: {len(m)} '
      f'({m[["stock", "calendar_date"]].drop_duplicates().shape[0]} unique)')

# --- Table 13 -----------------------------------------------------------
g = m.groupby('event_day').agg(
    N=('n_breaks', 'size'),
    rv=('rv_daily', 'mean'),
    spread=('avg_spread_rel', 'mean'))
print('\nEvent day   N   Realised volatility   Avg. rel. spread')
for ed, row in g.iterrows():
    print(f'{ed:+3d}     {int(row["N"]):4d}     {row["rv"]:.4f}        '
          f'{row["spread"]:.4f}')

# --- correlations --------------------------------------------------------
print('\ncorr(n_breaks, rv_daily):')
for mkt in ['STAR', 'ChiNext']:
    s = m[m['market'] == mkt]
    if len(s) > 1:
        r = s[['n_breaks', 'rv_daily']].corr().iloc[0, 1]
        print(f'  {mkt}: r = {r:+.3f}  (N = {len(s)})')

# --- LaTeX-ready table ----------------------------------------------------
print('\nLaTeX (Table 13):')
print(r'\begin{tabular}{lccc}')
print(r'\toprule')
print(r'Event day & Realised volatility & Avg.\ rel.\ spread \\')
print(r'\midrule')
for ed in sorted(g.index):
    row = g.loc[ed]
    print(f'{ed:+d} & {row["rv"]:.4f} & {row["spread"]:.4f} \\\\')
print(r'\bottomrule')

