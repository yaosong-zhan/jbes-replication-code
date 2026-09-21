"""
Compute alternative LOB measures (volatility, spread, volume) for earnings panel
stock-days and compare with break frequency around earnings announcements.
"""
import numpy as np, pandas as pd, os, sys, warnings, zipfile, io
from datetime import datetime, date
from collections import defaultdict

warnings.filterwarnings('ignore')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

DATA_ROOT = r'E:\Work\DataTick'
PANEL_PATH = os.path.join(SCRIPT_DIR, 'Results', 'Earnings_Panel.csv')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'Results')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_stock_day(stock_code, date_val, exchange='SH'):
    year = date_val.year
    month = f'{date_val.year}{date_val.month:02d}'
    zip_name = f'{month}{exchange}鑲＄エ浜旀。鍒嗙瑪.zip'
    zip_path = os.path.join(DATA_ROOT, str(year), zip_name)
    if not os.path.exists(zip_path):
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
                        target = n; break
                if target: break
            if target is None:
                return None
            raw = z.read(target)
            df = pd.read_csv(io.BytesIO(raw), encoding='gbk', on_bad_lines='skip')
        return df
    except Exception:
        return None


def filter_session(df_raw):
    """Filter to continuous auction session. Returns filtered df and time index."""
    tc = df_raw.iloc[:, 1].astype(str).str.strip()
    times = pd.to_datetime(tc, format='%H:%M:%S', errors='coerce')
    mins = times.dt.hour*60 + times.dt.minute
    mask = (((mins >= 9*60+35) & (mins <= 11*60+30)) |
            ((mins >= 13*60) & (mins <= 14*60+55)))
    df_filt = df_raw[mask].copy().reset_index(drop=True)
    time_filt = times[mask].reset_index(drop=True)
    return df_filt, time_filt


def compute_measures(df, times):
    """Compute volatility, spread, volume from LOB data."""
    if df is None or len(df) < 60:
        return None

    vals = df.iloc[:, 6:26].values.astype(float)
    bp1, sp1 = vals[:, 0], vals[:, 10]
    bp1[bp1 <= 0] = np.nan; sp1[sp1 <= 0] = np.nan
    theta = (bp1 + sp1) / 2
    valid = np.isfinite(theta)
    if valid.sum() < 60:
        return None

    # Filter valid observations
    theta = theta[valid]
    vals = vals[valid]
    times = times.iloc[valid].reset_index(drop=True) if hasattr(times, 'iloc') else times[valid]

    # 1. Realized volatility: 1-min log returns
    # Group by minute
    minutes_since = times.dt.hour * 60 + times.dt.minute
    unique_minutes = np.unique(minutes_since)
    if len(unique_minutes) < 2:
        return None

    min_prices = []
    for m in unique_minutes:
        mask = minutes_since == m
        min_prices.append(theta[mask][-1])  # last price in each minute

    log_ret = np.diff(np.log(min_prices))
    rv = np.sum(log_ret**2) * (6.5 * 60 / len(log_ret))  # annualize
    # Also compute realized volatility (standard deviation)
    rv_std = np.std(log_ret) * np.sqrt(6.5 * 60)  # daily vol from minute returns

    # 2. Spread computation
    bid1 = vals[:, 0]; ask1 = vals[:, 10]
    bid1[bid1 <= 0] = np.nan; ask1[ask1 <= 0] = np.nan
    spread_abs = ask1 - bid1
    spread_rel = spread_abs / theta
    avg_spread_abs = np.nanmean(spread_abs)
    avg_spread_rel = np.nanmean(spread_rel)

    # 3. Volume (turnover)
    bid_vol = vals[:, 1::2]  # columns 1,3,5,7,9
    ask_vol = vals[:, 11::2]  # columns 11,13,15,17,19
    # Replace NaN with 0
    bid_vol = np.nan_to_num(bid_vol, 0)
    ask_vol = np.nan_to_num(ask_vol, 0)
    total_volume = bid_vol.sum() + ask_vol.sum()

    # 4. Price range
    price_range = theta.max() - theta.min()

    # 5. Number of quote changes (mid-price moves)
    price_moves = (np.diff(theta) != 0).sum()

    return {
        'rv_daily': float(rv_std),
        'rv_raw': float(rv),
        'avg_spread_abs': float(avg_spread_abs),
        'avg_spread_rel': float(avg_spread_rel),
        'total_volume': float(total_volume),
        'price_range': float(price_range),
        'n_price_moves': int(price_moves),
        'n_obs': len(theta),
    }


def process_panel_subset(df_panel, max_days=2000, seed=42):
    """Process a subset of stock-days from the panel for alternative measures."""
    np.random.seed(seed)
    # Stratify by market and event day
    unique_days = df_panel.groupby(['stock', 'calendar_date', 'exchange']).first().reset_index()

    # Sample if too many
    if len(unique_days) > max_days:
        # Ensure both markets represented
        star_days = unique_days[unique_days['market'] == 'STAR']
        cnx_days = unique_days[unique_days['market'] == 'ChiNext']
        n_star = min(len(star_days), max_days // 3)
        n_cnx = min(len(cnx_days), max_days - n_star)
        star_sampled = star_days.sample(n=n_star, random_state=seed)
        cnx_sampled = cnx_days.sample(n=n_cnx, random_state=seed+1)
        sampled = pd.concat([star_sampled, cnx_sampled])
    else:
        sampled = unique_days

    print(f"Processing {len(sampled)} stock-days...")
    results = []
    total = len(sampled)
    for idx, (_, row) in enumerate(sampled.iterrows()):
        stock = row['stock']
        cal_date = datetime.strptime(row['calendar_date'], '%Y-%m-%d').date() if isinstance(row['calendar_date'], str) else row['calendar_date']
        ex = row['exchange']

        df_raw = load_stock_day(stock, cal_date, ex)
        if df_raw is None:
            continue

        df_filt, times = filter_session(df_raw)
        measures = compute_measures(df_filt, times)
        if measures is None:
            continue

        measures['stock'] = stock
        measures['calendar_date'] = str(cal_date)
        measures['exchange'] = ex
        results.append(measures)

        if (idx + 1) % 100 == 0:
            print(f"  [{idx+1}/{total}] processed, {len(results)} with data")

    return pd.DataFrame(results)


def main(max_days=3000):
    # Load panel
    df_panel = pd.read_csv(PANEL_PATH)
    print(f"Panel loaded: {len(df_panel)} rows, {df_panel['stock'].nunique()} stocks")

    # Get unique stock-days and process
    df_alt = process_panel_subset(df_panel, max_days=max_days)
    print(f"Alternative measures computed for {len(df_alt)} stock-days")

    # Merge with panel
    df_alt['calendar_date'] = df_alt['calendar_date'].astype(str)
    df_panel['calendar_date'] = df_panel['calendar_date'].astype(str)
    df_merged = df_panel.merge(df_alt, on=['stock', 'calendar_date', 'exchange'], how='inner')
    print(f"Merged: {len(df_merged)} rows")

    if len(df_merged) > 0:
        # Save
        out_path = os.path.join(OUTPUT_DIR, 'earnings_alt_measures.csv')
        df_merged.to_csv(out_path, index=False)
        print(f"Saved to {out_path}")

        # Summary by event day
        print(f"\n{'='*65}")
        print("  ALTERNATIVE MEASURES BY EVENT DAY")
        print(f"{'='*65}")
        for ed in sorted(df_merged['event_day'].unique()):
            sub = df_merged[df_merged['event_day'] == ed]
            print(f"  Day {ed:+d}: N={len(sub):,} breaks={sub['n_breaks'].mean():.2f} "
                  f"vol={sub['rv_daily'].mean():.4f} spread={sub['avg_spread_rel'].mean():.4f} "
                  f"volume={sub['total_volume'].mean():.0f}")

        # Pre vs Post comparison
        df_merged['period'] = 'other'
        df_merged.loc[df_merged['event_day'].isin([-3,-2,-1]), 'period'] = 'Pre'
        df_merged.loc[df_merged['event_day'].isin([0,1]), 'period'] = 'Post'

        print(f"\n  Pre vs Post comparison:")
        for mkt in ['STAR', 'ChiNext']:
            sub = df_merged[df_merged['market'] == mkt]
            for period in ['Pre', 'Post']:
                sub2 = sub[sub['period'] == period]
                if len(sub2) > 0:
                    print(f"  {mkt:>8s} {period:>4s}: breaks={sub2['n_breaks'].mean():.2f} "
                          f"vol={sub2['rv_daily'].mean():.4f} spread={sub2['avg_spread_rel'].mean():.4f}")

        # Correlation between breaks and traditional measures
        print(f"\n  Correlations with break frequency:")
        for mkt in ['STAR', 'ChiNext']:
            sub = df_merged[df_merged['market'] == mkt]
            print(f"  {mkt}:")
            for col in ['rv_daily', 'avg_spread_rel', 'total_volume', 'price_range']:
                corr = sub['n_breaks'].corr(sub[col])
                print(f"    breaks vs {col:>20s}: {corr:.3f}")

    else:
        print("No merged data available!")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_days', type=int, default=3000)
    args = ap.parse_args()
    main(max_days=args.max_days)

