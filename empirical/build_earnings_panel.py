"""
Build comprehensive earnings event panel: stock-day level break features
for STAR Market (688) and ChiNext (300) around earnings announcements.

Output:
  - Earnings_Panel.csv          (stock-day level, all metrics)
  - Earnings_BreakDetails.json  (individual break details, for deep dives)

"""
import numpy as np, pandas as pd, os, sys, json, warnings, zipfile, io
from datetime import datetime, date
from collections import defaultdict, Counter
warnings.filterwarnings('ignore')

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from simulation_joint_change import run_full_pipeline

RESULTS_DIR = os.path.join(script_dir, 'Results')
REPORT_DATA_PATH = os.path.join(script_dir, '..', 'Data', 'ReportDate.csv')
TRADING_DATES_PATH = os.path.join(script_dir, '..', 'Data', 'trading_dates.csv')
DATA_ROOT = r'E:\Work\DataTick'
ALPHA = 0.05


# ============================================================================
#  Trading Calendar
# ============================================================================

def load_trading_dates():
    df = pd.read_csv(TRADING_DATES_PATH, parse_dates=['trading_date'])
    return sorted(df['trading_date'].dt.date.tolist())


def get_window_dates(trading_dates, event_date, n_before=5, n_after=5):
    """Get event window [-n_before, +n_after] around event_date."""
    if event_date not in trading_dates:
        return []
    idx = trading_dates.index(event_date)
    before = trading_dates[max(0, idx - n_before):idx]
    after = trading_dates[idx + 1:idx + 1 + n_after]
    return before + [event_date] + after


# ============================================================================
#  Data Loading
# ============================================================================

def find_zip_for_date(date, exchange='SH'):
    """Find the monthly zip archive for a given date and exchange."""
    year = date.year
    month = f'{year}{date.month:02d}'
    zip_name = f'{month}{exchange}鑲＄エ浜旀。鍒嗙瑪.zip'
    zip_path = os.path.join(DATA_ROOT, str(year), zip_name)
    if os.path.exists(zip_path):
        return zip_path
    return None


def load_stock_day(stock_code, date_val, exchange='SH'):
    """Load one stock-day CSV from monthly zip. Returns DataFrame or None."""
    zip_path = find_zip_for_date(date_val, exchange)
    if zip_path is None:
        return None
    date_str = date_val.strftime('%Y%m%d')
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            names = z.namelist()
            target = None
            for n in names:
                parts = n.split('/')
                for p in parts:
                    if p == f'{stock_code}_{date_str}.csv':
                        target = n
                        break
                if target:
                    break
            if target is None:
                return None
            raw = z.read(target)
            df = pd.read_csv(io.BytesIO(raw), encoding='gbk', on_bad_lines='skip')
        return df
    except Exception:
        return None


# ============================================================================
#  Construct Joint Process from Raw LOB Data
# ============================================================================

def construct_joint_process_from_raw(df_raw, tick_size=0.01, K=None):
    """Build Z_t = (mu_pct(d), theta_t) from raw LOB CSV columns."""
    vals = df_raw.iloc[:, 6:26].values.astype(float)

    # Mid-price
    bp1, sp1 = vals[:, 0], vals[:, 10]
    bp1[bp1 <= 0] = np.nan; sp1[sp1 <= 0] = np.nan
    theta = (bp1 + sp1) / 2
    mask = np.isnan(theta)
    if mask.any():
        valid = np.flatnonzero(~mask)
        if len(valid) > 1:
            theta[mask] = np.interp(np.flatnonzero(mask), valid, theta[valid])
    T = len(theta)

    # Adaptive K
    if K is None:
        all_d = []
        for level in range(5):
            bp = vals[:, level * 2]; ap = vals[:, 10 + level * 2]
            with np.errstate(invalid='ignore'):
                all_d.extend(np.abs(np.round((bp - theta) / tick_size))[bp > 0])
                all_d.extend(np.abs(np.round((ap - theta) / tick_size))[ap > 0])
        all_d = np.array(all_d)
        all_d = all_d[np.isfinite(all_d)]
        K = max(5, int(np.ceil(np.percentile(all_d, 99)))) if len(all_d) > 0 else 10
        K = min(K, 50)

    d_vals = list(range(-K, 0)) + list(range(1, K + 1))
    mu = np.zeros((T, 2 * K))
    for t in range(T):
        mid = theta[t]
        if np.isnan(mid) or mid <= 0:
            continue
        for level in range(5):
            bp = vals[t, level * 2]; bv = vals[t, level * 2 + 1]
            if bp > 0 and bv > 0 and not np.isnan(bp) and not np.isnan(bv):
                d = int(round((bp - mid) / tick_size))
                d = max(d, -K)
                if d < 0 and d in d_vals:
                    mu[t, d_vals.index(d)] += bv
        for level in range(5):
            ap = vals[t, 10 + level * 2]; av = vals[t, 10 + level * 2 + 1]
            if ap > 0 and av > 0 and not np.isnan(ap) and not np.isnan(av):
                d = int(round((ap - mid) / tick_size))
                d = min(d, K)
                if d > 0 and d in d_vals:
                    mu[t, d_vals.index(d)] -= av

    # Z_t = (mu_pct(d), Delta theta_t): the price block enters as the
    # differenced mid-price so that the joint process is stationary under
    # the null (per the theory in the paper). The first snapshot has no
    # predecessor, so the shape block is aligned to the differences by
    # dropping its first row.
    dtheta = np.diff(theta)
    total_abs = np.abs(mu).sum(axis=1, keepdims=True)
    total_abs[total_abs == 0] = 1
    mu_pct = mu / total_abs
    Z = np.column_stack([mu_pct[1:], dtheta])
    mask = np.any(~np.isfinite(Z), axis=1)
    if mask.any():
        Z = Z[~mask]
    col_std = Z.std(axis=0); col_std[col_std == 0] = 1.0
    Z = Z / col_std
    return Z


# ============================================================================
#  Process One Stock-Day
# ============================================================================

def process_one_day(stock_code, date_val, exchange='SH'):
    """Run pipeline on one stock-day. Returns list of break dicts."""
    df = load_stock_day(stock_code, date_val, exchange)
    if df is None or df.shape[1] < 26:
        return []

    # Filter continuous session
    tc = df.iloc[:, 1].astype(str).str.strip()
    times = pd.to_datetime(tc, format='%H:%M:%S', errors='coerce')
    mins = times.dt.hour * 60 + times.dt.minute
    mask = (((mins >= 9*60+35) & (mins <= 11*60+30)) |
            ((mins >= 13*60) & (mins <= 14*60+55)))
    df = df[mask].reset_index(drop=True)
    if len(df) < 100:
        return []

    Z = construct_joint_process_from_raw(df)
    if Z.shape[0] < 60 or Z.shape[1] < 2:
        return []

    try:
        res = run_full_pipeline(Z, alpha=ALPHA, q_method='variance',
                                var_threshold=0.85, max_q=5,
                                binary_seg=True, min_seg_len=30)
    except Exception:
        return []

    if not res.get('reject') or res.get('num_breaks', 0) == 0:
        return []

    details = res.get('break_details', [])
    result = []
    q_used = res.get('q', None)
    for b in details:
        result.append({
            'tau_hat': b.get('tau_hat', 0),
            'q': q_used,
            'break_type': b.get('break_type', 'unknown'),
            'omega_mu': b.get('omega_mu', 0),
            'omega_theta': b.get('omega_theta', 0),
            'delta_price': b.get('delta_price', 0),
            'delta_shape_mean': b.get('delta_shape_mean', 0),
            'price_dir': b.get('price_dir', 0),
            'shape_dir': b.get('shape_dir', 0),
        })
    return result


# ============================================================================
#  Day-Level Metrics
# ============================================================================

def compute_metrics(breaks):
    """All day-level metrics from a list of break dicts."""
    n = len(breaks)
    if n == 0:
        return {}

    types = [b['break_type'] for b in breaks]
    m = {
        'n_breaks': n,
        'n_shape': types.count('shape-driven'),
        'n_price': types.count('price-driven'),
        'n_joint': types.count('joint'),
        'shap_pct': types.count('shape-driven') / n,
        'pric_pct': types.count('price-driven') / n,
        'join_pct': types.count('joint') / n,
    }

    # Run lengths
    if n >= 2:
        cur, length = types[0], 1
        runs = []
        for t in types[1:]:
            if t == cur:
                length += 1
            else:
                runs.append(length)
                cur, length = t, 1
        runs.append(length)
        m['n_runs'] = len(runs)
        m['mean_run'] = np.mean(runs)
        m['max_run'] = max(runs)
        # Per-type runs
        for t in ['shape-driven', 'price-driven', 'joint']:
            tl = [r for r, rt in zip(runs, [types[0]] + [types[i] for i in range(1, len(types)) if types[i] != types[i-1]]) if rt == t]
            # Actually, redo: runs and their types
            run_types = []
            cur_t, cur_len = types[0], 1
            for t_i in types[1:]:
                if t_i == cur_t:
                    cur_len += 1
                else:
                    run_types.append((cur_t, cur_len))
                    cur_t, cur_len = t_i, 1
            run_types.append((cur_t, cur_len))
            for t2 in ['shape-driven', 'price-driven', 'joint']:
                lens = [rl for rt, rl in run_types if rt == t2]
                m[f'{t2[:4]}_mean_run'] = np.mean(lens) if lens else 0
                m[f'{t2[:4]}_max_run'] = max(lens) if lens else 0
    else:
        m['n_runs'] = 1
        m['mean_run'] = 1.0
        m['max_run'] = 1

    # Self-persistence (diagonal of transition matrix)
    if n >= 2:
        trans = {t: 0 for t in ['shape-driven', 'price-driven', 'joint']}
        total_trans = {t: 0 for t in ['shape-driven', 'price-driven', 'joint']}
        for i in range(n - 1):
            total_trans[types[i]] = total_trans.get(types[i], 0) + 1
            if types[i] == types[i+1]:
                trans[types[i]] = trans.get(types[i], 0) + 1
        for t in ['shape-driven', 'price-driven', 'joint']:
            m[f'{t[:4]}_self'] = trans[t] / total_trans[t] if total_trans[t] > 0 else 0
    else:
        for t in ['shape-driven', 'price-driven', 'joint']:
            m[f'{t[:4]}_self'] = 0

    # Transition probabilities (off-diagonal average)
    if n >= 2:
        tm_from = defaultdict(list)
        for i in range(n - 1):
            tm_from[types[i]].append(types[i+1])
        off_diag = []
        for t1 in ['shape-driven', 'price-driven', 'joint']:
            if t1 in tm_from and len(tm_from[t1]) > 0:
                for t2 in ['shape-driven', 'price-driven', 'joint']:
                    if t1 != t2:
                        off_diag.append(tm_from[t1].count(t2) / len(tm_from[t1]))
        m['trans_off_diag'] = np.mean(off_diag) if off_diag else 0
    else:
        m['trans_off_diag'] = 0

    # Direction persistence
    dirs = [(b.get('price_dir', 0), b.get('shape_dir', 0)) for b in breaks]
    for t in ['shape-driven', 'price-driven', 'joint']:
        td = [dirs[i] for i in range(n) if types[i] == t]
        if len(td) >= 2:
            m[f'{t[:4]}_dir'] = sum(1 for i in range(1, len(td)) if td[i] == td[i-1]) / (len(td) - 1)
        else:
            m[f'{t[:4]}_dir'] = 0

    return m


# ============================================================================
#  Main Panel Builder
# ============================================================================

def build_panel(max_events_per_stock=None, max_stocks=None, resume=False):
    print("=" * 70)
    print("  EARNINGS EVENT PANEL: STAR Market (688) + ChiNext (300)")
    print("=" * 70)

    # Load report dates
    df_rep = pd.read_csv(REPORT_DATA_PATH, encoding='utf-8-sig')
    print(f"\nReport dates file: {len(df_rep)} stocks")

    # Filter: 688 (STAR/绉戝垱鏉? and 300 (ChiNext/鍒涗笟鏉?
    codes_raw = df_rep.iloc[:, 0].astype(str).str.strip()
    mask_688 = codes_raw.str.match(r'688\d{3}\.SH')
    mask_300 = codes_raw.str.match(r'300\d{3}\.SZ')
    df_sub = df_rep[mask_688 | mask_300].copy()
    print(f"  688 (绉戝垱鏉?: {mask_688.sum()}, 300 (鍒涗笟鏉?: {mask_300.sum()}")
    print(f"  Total in scope: {len(df_sub)}")

    # Parse exchange
    def get_exchange(code_str):
        raw = code_str.strip()
        if raw.startswith('688') and '.SH' in raw:
            return ('SH', raw.replace('.SH', '').replace('.SZ', '').strip().zfill(6))
        if raw.startswith('300') and '.SZ' in raw:
            return ('SZ', raw.replace('.SH', '').replace('.SZ', '').strip().zfill(6))
        # Fallback: check suffix
        if '.SH' in raw:
            return ('SH', raw.replace('.SH', '').replace('.SZ', '').strip().zfill(6))
        return ('SZ', raw.replace('.SH', '').replace('.SZ', '').strip().zfill(6))

    exchanges = []
    codes = []
    for _, row in df_sub.iterrows():
        ex, cd = get_exchange(str(row.iloc[0]))
        exchanges.append(ex)
        codes.append(cd)
    df_sub['exchange'] = exchanges
    df_sub['code'] = codes

    # Limit stocks if specified (after exchange/code parsed)
    if max_stocks:
        unique_codes = df_sub[['code']].drop_duplicates().head(max_stocks)
        keep_set = set(unique_codes['code'])
        df_sub = df_sub[df_sub['code'].isin(keep_set)]
        print(f"  Limited to {max_stocks} stocks: {len(df_sub)} rows in scope")
    print(f"  SH (绉戝垱鏉?: {(df_sub['exchange']=='SH').sum()}, SZ (鍒涗笟鏉?: {(df_sub['exchange']=='SZ').sum()}")

    # Collect report dates for 2022-2024 (columns 3 to 14)
    report_cols = list(range(3, 15))
    all_events = []
    for _, row in df_sub.iterrows():
        code = row['code']; exchange = row['exchange']
        for col in report_cols:
            d_str = str(row.iloc[col]).strip()
            if d_str and d_str not in ('nan', 'NaT', ''):
                try:
                    dt = datetime.strptime(d_str, '%Y-%m-%d').date()
                    if datetime(2022, 1, 1).date() <= dt <= datetime(2024, 11, 30).date():
                        all_events.append({'code': code, 'exchange': exchange, 'report_date': dt})
                except:
                    pass
    print(f"\nTotal report events (2022-2024): {len(all_events)}")
    codes_in_events = set(e['code'] for e in all_events)
    print(f"Unique stocks: {len(codes_in_events)}")

    # The same report date can appear in more than one report-type column of
    # the RESSET file (e.g., annual and Q1 reports disclosed on the same day),
    # so the same event would otherwise be processed twice. Deduplicate on
    # (stock, report date) before expanding event windows.
    n_before = len(all_events)
    all_events = list({(e['code'], e['report_date']): e for e in all_events}.values())
    print(f"Deduplicated report records: {n_before} -> {len(all_events)} unique events")

    # Limit per stock if specified
    if max_events_per_stock:
        from collections import Counter
        cnt = Counter(e['code'] for e in all_events)
        limited = []
        code_counts = defaultdict(int)
        for e in all_events:
            if code_counts[e['code']] < max_events_per_stock:
                limited.append(e)
                code_counts[e['code']] += 1
        all_events = limited
        print(f"Limited to {max_events_per_stock} events/stock: {len(all_events)} events")

    # Load trading dates
    trading_dates = load_trading_dates()
    print(f"Trading dates: {len(trading_dates)}")

    # =====================================================================
    #  Process each event
    # =====================================================================
    WIN_BEFORE = 3  # matches the n_before in get_window_dates
    panel_path = os.path.join(RESULTS_DIR, 'Earnings_Panel.csv')
    details_path = os.path.join(RESULTS_DIR, 'Earnings_BreakDetails.json')

    panel_rows = []
    all_break_details = []
    processed_set = set()
    if resume and os.path.exists(panel_path):
        try:
            old = pd.read_csv(panel_path)
            panel_rows = old.to_dict('records')
            processed_set = set(zip(old['stock'].astype(str),
                                   old['report_date'].astype(str)))
            print(f"Resume: loaded {len(panel_rows)} existing panel rows, "
                  f"{len(processed_set)} events to skip")
        except Exception as e:
            print(f"Resume: could not load existing panel ({e}); starting fresh")
        try:
            with open(details_path, 'r') as f:
                all_break_details = json.load(f)
            print(f"Resume: loaded {len(all_break_details)} existing break details")
        except Exception:
            print("Resume: no usable break details file; details start empty")

    total = len(all_events)
    processed = 0
    skipped_no_data = 0
    skipped_few_breaks = 0
    SAVE_EVERY = 500

    for idx, ev in enumerate(all_events):
        code = ev['code']; exchange = ev['exchange']; rep_date = ev['report_date']

        # Skip events already processed in a previous (possibly crashed) run
        if resume and (str(code), str(rep_date)) in processed_set:
            continue

        # Get event window
        window_dates = get_window_dates(trading_dates, rep_date, WIN_BEFORE, WIN_BEFORE)

        for day_offset, cal_date in enumerate(window_dates):
            event_day = day_offset - WIN_BEFORE

            # Skip if event_day is not a report date or prior (keep day+0)
            breaks = process_one_day(code, cal_date, exchange)

            if len(breaks) == 0:
                skipped_few_breaks += 1
                continue

            # Compute day metrics
            metrics = compute_metrics(breaks)
            if not metrics:
                continue

            row = {
                'stock': code,
                'exchange': exchange,
                'report_date': str(rep_date),
                'calendar_date': str(cal_date),
                'event_day': event_day,
                'market': 'STAR' if exchange == 'SH' else 'ChiNext',
            }
            row.update(metrics)
            panel_rows.append(row)

            # Store break details
            for b in breaks:
                bd = {
                    'stock': code,
                    'exchange': exchange,
                    'report_date': str(rep_date),
                    'calendar_date': str(cal_date),
                    'event_day': event_day,
                }
                bd.update(b)
                all_break_details.append(bd)

        processed += 1
        if processed % 100 == 0 or processed == 1:
            print(f"[{idx+1}/{total}] {code} ({exchange}) {rep_date}: "
                  f"{len(window_dates)} days in window, {len(panel_rows)} panel rows so far",
                  flush=True)
        if processed % SAVE_EVERY == 0:
            # Incremental checkpoint. Compact JSON (no indent) keeps the peak
            # memory of the dump low; a failed checkpoint must not kill the run
            # because the data stays in memory and is saved again at the end.
            try:
                pd.DataFrame(panel_rows).to_csv(panel_path, index=False)
                with open(details_path, 'w') as f:
                    json.dump(all_break_details, f, separators=(',', ':'))
                print(f"  [checkpoint at {idx+1}/{total}]: saved "
                      f"{len(panel_rows)} rows / {len(all_break_details)} breaks", flush=True)
            except Exception as e:
                print(f"  WARNING: checkpoint save failed ({e}); will retry at "
                      f"the next checkpoint", flush=True)

    # =====================================================================
    #  Save
    # =====================================================================
    df_panel = pd.DataFrame(panel_rows)

    # Reorder columns: identifiers first, then metrics
    id_cols = ['stock', 'exchange', 'market', 'report_date', 'calendar_date', 'event_day']
    metric_cols = [c for c in df_panel.columns if c not in id_cols]
    df_panel = df_panel[id_cols + metric_cols]
    df_panel = df_panel.sort_values(['stock', 'report_date', 'event_day']).reset_index(drop=True)

    panel_path = os.path.join(RESULTS_DIR, 'Earnings_Panel.csv')
    df_panel.to_csv(panel_path, index=False)
    print(f"\nPanel saved: {len(df_panel)} rows to {panel_path}")

    # Break details as JSON (compact separators to bound peak memory)
    details_path = os.path.join(RESULTS_DIR, 'Earnings_BreakDetails.json')
    with open(details_path, 'w') as f:
        json.dump(all_break_details, f, separators=(',', ':'))
    print(f"Break details saved: {len(all_break_details)} breaks to {details_path}")

    # =====================================================================
    #  Summary
    # =====================================================================
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Events processed: {processed}/{total}")
    print(f"  Panel rows (stock-days): {len(df_panel)}")
    print(f"  Total breaks: {len(all_break_details)}")
    print(f"  Unique stocks: {df_panel['stock'].nunique()}")
    print(f"\n  By market:")
    for mkt in ['STAR', 'ChiNext']:
        sub = df_panel[df_panel['market'] == mkt]
        print(f"    {mkt}: {sub['stock'].nunique()} stocks, {len(sub)} stock-days, "
              f"{sub['n_breaks'].sum()} breaks")
    print(f"\n  By event day:")
    for ed in range(-5, 6):
        sub = df_panel[df_panel['event_day'] == ed]
        if len(sub) > 0:
            print(f"    Day {ed:+d}: {len(sub)} stock-days, "
                  f"mean breaks = {sub['n_breaks'].mean():.2f}")
    print(f"\nDone.")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_events', type=int, default=None,
                    help='Max events per stock (for testing)')
    ap.add_argument('--max_stocks', type=int, default=None,
                    help='Max stocks to process (for testing)')
    ap.add_argument('--resume', action='store_true',
                    help='Skip events already present in Earnings_Panel.csv')
    args = ap.parse_args()
    build_panel(max_events_per_stock=args.max_events, max_stocks=args.max_stocks,
                resume=args.resume)

