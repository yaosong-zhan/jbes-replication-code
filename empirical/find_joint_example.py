"""Find a joint-break example with a visible raw mid-price move over the
卤5-snapshot window used in the break-examples figure (螖胃 pipeline)."""
import json, numpy as np, pandas as pd, zipfile, io, sys
from datetime import date

with open('Script/Results/Earnings_BreakDetails.json', 'r') as f:
    det = json.load(f)
cands = [b for b in det if b['break_type'] == 'joint'
         and 0.44 <= b['omega_mu'] <= 0.55 and abs(b['delta_price']) >= 0.15]
# prefer STAR stocks (more volatile) and 2022-2023 dates
cands.sort(key=lambda b: (0 if b['stock'].startswith('688') else 1,
                          b['calendar_date']))

def load(stock_code, date_val, exchange):
    year = date_val.year
    zip_path = f'E:/Work/DataTick/{year}/{year}{date_val.month:02d}{exchange}鑲＄エ浜旀。鍒嗙瑪.zip'
    ds = date_val.strftime('%Y%m%d')
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            target = next((n for n in z.namelist() if f'{stock_code}_{ds}.csv' in n), None)
            if target is None:
                return None
            return pd.read_csv(io.BytesIO(z.read(target)), encoding='gbk', on_bad_lines='skip')
    except Exception:
        return None

def filt(df):
    tc = df.iloc[:, 1].astype(str).str.strip()
    tm = pd.to_datetime(tc, format='%H:%M:%S', errors='coerce')
    mins = tm.dt.hour * 60 + tm.dt.minute
    mask = (((mins >= 9 * 60 + 35) & (mins <= 11 * 60 + 30)) |
            ((mins >= 13 * 60) & (mins <= 14 * 60 + 55)))
    return df[mask].reset_index(drop=True)

best = []
tested = 0
for b in cands:
    if tested >= 400:
        break
    ex = 'SH' if b['stock'].startswith('688') else 'SZ'
    y, m, d = map(int, b['calendar_date'].split('-'))
    df = load(b['stock'], date(y, m, d), ex)
    if df is None:
        continue
    df = filt(df)
    if len(df) < 2000:
        continue
    tested += 1
    tau = b['tau_hat'] + 1
    if tau < 10 or tau > len(df) - 10:
        continue
    vals = df.iloc[:, 6:26].values.astype(float)
    bp1, sp1 = vals[:, 0], vals[:, 10]
    bp1[bp1 <= 0] = np.nan; sp1[sp1 <= 0] = np.nan
    th = (bp1 + sp1) / 2
    mb = np.nanmean(th[tau - 5:tau]); ma = np.nanmean(th[tau:tau + 5])
    chg = ma - mb
    best.append((abs(chg), b, chg, len(df)))
    if len(best) >= 5:
        break

best.sort(key=lambda t: t[0], reverse=True)
for ab, b, chg, n in best[:5]:
    print(f'{b["stock"]} {b["calendar_date"]} tau={b["tau_hat"]} '
          f'om={b["omega_mu"]:.3f} dp_std={b["delta_price"]:+.3f} | raw5={chg:+.3f} n={n}')
print('tested:', tested)

