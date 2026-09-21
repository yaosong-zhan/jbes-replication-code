"""
Plot LOB shape before and after detected breaks for three example stocks.
Shows shape-driven, price-driven, and joint break examples.
"""
import numpy as np, pandas as pd, os, sys, zipfile, io, warnings, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import date

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_ROOT = r'E:\Work\DataTick'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, 'Figures')
os.makedirs(FIG_DIR, exist_ok=True)

TICK_SIZE = 0.01


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


def compute_volume_profile(df_raw, tick_size=0.01):
    """Compute volume distribution by tick offset from mid."""
    vals = df_raw.iloc[:, 6:26].values.astype(float)
    bp1, sp1 = vals[:, 0], vals[:, 10]
    bp1[bp1 <= 0] = np.nan; sp1[sp1 <= 0] = np.nan
    theta = (bp1 + sp1) / 2

    # Determine K adaptively
    all_d = []
    for level in range(5):
        bp = vals[:, level*2]; ap = vals[:, 10+level*2]
        with np.errstate(invalid='ignore'):
            all_d.extend(np.abs(np.round((bp - theta)/tick_size))[bp>0])
            all_d.extend(np.abs(np.round((ap - theta)/tick_size))[ap>0])
    all_d = np.array([d for d in all_d if np.isfinite(d)])
    K = max(5, int(np.ceil(np.percentile(all_d, 99)))) if len(all_d)>0 else 10
    K = min(K, 50)

    n_obs = len(vals)
    bid_vol = np.zeros((n_obs, K))
    ask_vol = np.zeros((n_obs, K))
    bid_prices = np.zeros((n_obs, K))
    ask_prices = np.zeros((n_obs, K))

    for t in range(n_obs):
        mid = theta[t]
        if np.isnan(mid) or mid <= 0:
            continue
        # Bid side: prices below mid
        for level in range(5):
            bp = vals[t, level*2]; bv = vals[t, level*2+1]
            if bp > 0 and bv > 0 and not np.isnan(bp) and not np.isnan(bv):
                d = int(round((mid - bp) / tick_size))
                if 1 <= d <= K:
                    bid_vol[t, d-1] += bv
                    bid_prices[t, d-1] = float(bp)
        # Ask side: prices above mid
        for level in range(5):
            ap = vals[t, 10+level*2]; av = vals[t, 10+level*2+1]
            if ap > 0 and av > 0 and not np.isnan(ap) and not np.isnan(av):
                d = int(round((ap - mid) / tick_size))
                if 1 <= d <= K:
                    ask_vol[t, d-1] += av
                    ask_prices[t, d-1] = float(ap)

    return theta, bid_vol, ask_vol, bid_prices, ask_prices, K


def filter_session(df_raw):
    """Filter to continuous auction session."""
    tc = df_raw.iloc[:, 1].astype(str).str.strip()
    times = pd.to_datetime(tc, format='%H:%M:%S', errors='coerce')
    mins = times.dt.hour*60 + times.dt.minute
    mask = (((mins >= 9*60+35) & (mins <= 11*60+30)) |
            ((mins >= 13*60) & (mins <= 14*60+55)))
    return df_raw[mask].reset_index(drop=True), times[mask].reset_index(drop=True)


# ============================================================
#  Three examples (selected from the differenced-midprice break
#  details, Earnings_BreakDetails.json). tau_hat indexes the
#  joint process Z (first snapshot dropped), so the index into
#  the filtered session is tau_hat + 1.
# ============================================================
examples = [
    ('300001', date(2022,8,25), 804, 'SZ', 'Shape-driven', 'omega_mu = 0.99'),
    ('300260', date(2023,4,27), 1706, 'SZ', 'Price-driven', 'omega_mu = 0.22'),
    ('688575', date(2022,7,21), 2911, 'SH', 'Joint', 'omega_mu = 0.54'),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (stock, dt, tau, ex, title, subtitle) in zip(axes, examples):
    print(f"Loading {stock} {dt}...")
    df_raw = load_stock_day(stock, dt, ex)
    if df_raw is None:
        print(f"  No data for {stock}")
        continue

    df_filt, times = filter_session(df_raw)
    tau = tau + 1  # tau_hat in Z -> index in filtered session
    print(f"  Filtered: {len(df_filt)} obs, tau={tau}")

    theta, bid_vol, ask_vol, bid_prices, ask_prices, K = compute_volume_profile(df_filt)

    # Average LOB before and after tau (window of 30 obs each side)
    win = 5  # narrow window to capture break sharply
    before_slice = slice(max(0, tau-win), tau)
    after_slice = slice(tau, min(len(theta), tau+win))

    if before_slice.stop <= before_slice.start or after_slice.stop <= after_slice.start:
        print(f"  Insufficient data around tau={tau}")
        continue

    mid_before = theta[before_slice].mean()
    mid_after = theta[after_slice].mean()

    bid_before = bid_vol[before_slice].sum(axis=0)
    ask_before = ask_vol[before_slice].sum(axis=0)
    bid_after = bid_vol[after_slice].sum(axis=0)
    ask_after = ask_vol[after_slice].sum(axis=0)

    # Normalize to percentage of total volume
    total_before = bid_before.sum() + ask_before.sum()
    total_after = bid_after.sum() + ask_after.sum()
    total_before = max(total_before, 1)
    total_after = max(total_after, 1)

    bid_before_pct = bid_before / total_before * 100
    ask_before_pct = ask_before / total_before * 100
    bid_after_pct = bid_after / total_after * 100
    ask_after_pct = ask_after / total_after * 100

    ticks = np.arange(1, K+1)

    ax.set_title(f'{title}\n{stock}', fontsize=12, fontweight='bold')

    ax.plot(ticks, bid_before_pct, 'b-', linewidth=1.5, alpha=0.7, label='Bid before')
    ax.plot(ticks, ask_before_pct, 'r-', linewidth=1.5, alpha=0.7, label='Ask before')
    ax.plot(ticks, bid_after_pct, 'b--', linewidth=2.0, alpha=0.9, label='Bid after')
    ax.plot(ticks, ask_after_pct, 'r--', linewidth=2.0, alpha=0.9, label='Ask after')

    # Mid-price lines
    ax.axvline(x=0, color='gray', linewidth=0.5, linestyle=':')

    # Info box
    price_info = f'Mid: {mid_before:.2f} -> {mid_after:.2f}'
    info = f'{subtitle}\n{price_info}'
    ax.text(0.95, 0.95, info, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Ticks from mid-price', fontsize=10)
    ax.set_ylabel('Volume share (%)', fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim(0.5, min(K+0.5, 20))
    ax.grid(True, alpha=0.3)

plt.tight_layout()
png_path = os.path.join(FIG_DIR, 'break_examples.png')
eps_path = os.path.join(FIG_DIR, 'break_examples.eps')
plt.savefig(png_path, dpi=150, bbox_inches='tight')
plt.savefig(eps_path, format='eps', bbox_inches='tight')
print(f"\nSaved:\n  {png_path}\n  {eps_path}")

