"""Rerun Tables 2 and 10 with 2,000 replications and original seed rules.

Usage: python rerun_revision_mc.py size|comparison
Parallel workers evaluate independent configurations; results are checkpointed.
"""
import os
for name in ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS']:
    os.environ[name] = '1'
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from simulation_joint_change import generate_data, run_full_pipeline

REPS = 2000
OUT = Path(__file__).parent / 'Results'


def size_task(task):
    idx, T, alpha, err, param = task
    count = 0
    old_count = 0
    for rep in range(REPS):
        Z, _, _ = generate_data(T, 11, 3, 'none', 0.5, 0.0,
            error_type=err, ar_coeff=param if err == 'ar' else 0.,
            ma_coeff=param if err == 'ma' else .5,
            garch_params=(.15, .80) if err == 'garch' else None,
            seed=rep * 100 + idx)
        reject = run_full_pipeline(Z, alpha, q_method='fixed', fixed_q=5)['reject']
        count += int(reject)
        if rep < 500:
            old_count += int(reject)
    size = count / REPS
    return dict(T=T, alpha=alpha, error=err, param=param, size=size,
                se=np.sqrt(size*(1-size)/REPS), n_reps=REPS,
                n_rej=count, first_500_size=old_count/500, config_index=idx)


def comparison_task(task):
    import sim_method_comparison as mc
    T, err, param, bt, snr = task
    counts = dict(block=0, jointpca=0, full=0, separate=0)
    for rep in range(REPS):
        Z, _, _ = generate_data(T, 11, 3, bt, .5, snr,
            error_type=err, ar_coeff=param if err == 'ar' else 0.,
            ma_coeff=param if err == 'ma' else .5,
            garch_params=(.15,.80) if err == 'garch' else None,
            rho_cov=.5, seed=rep*1000+T+7)
        for key, fn in [('block', mc.design_block), ('jointpca', mc.design_jointpca),
                        ('full', mc.design_full), ('separate', mc.design_separate)]:
            counts[key] += int(fn(Z))
    return dict(T=T, error=err, break_type=bt, snr=snr,
                **{k:v/REPS for k,v in counts.items()}, n_reps=REPS)


def main(mode):
    if mode == 'size':
        configs = [(T,a,e,p) for T in [100,250,500,1000,2000,5000]
                   for a in [.01,.05,.10]
                   for e,p in [('iid',0),('ar',0.),('ar',.3),('ar',.5),('ma',.5),('garch',0)]]
        tasks = [(i,*c) for i,c in enumerate(configs)]
        fn, name = size_task, 'sim_size'
    elif mode == 'comparison':
        tasks = [(T,e,p,b,s) for T in [500,1000,2000]
                 for e,p in [('iid',0),('ar',.3),('ma',.5),('garch',0)]
                 for b,s in [('none',0.),('shape-driven',.1),('price-driven',.5),('joint',.1)]]
        fn, name = comparison_task, 'sim_method_comparison'
    else:
        raise ValueError(mode)
    rows, start = [], time.time()
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fn,t):i for i,t in enumerate(tasks)}
        for f in as_completed(futures):
            row=f.result()
            row['_order']=futures[f]
            rows.append(row)
            pd.DataFrame(rows).sort_values('_order').drop(columns='_order').to_csv(
                OUT / (name+'_2000_checkpoint.csv'),index=False)
            print(f'{mode} {len(rows)}/{len(tasks)} ({time.time()-start:.0f}s): {row}',flush=True)
    pd.DataFrame(rows).sort_values('_order').drop(columns='_order').to_csv(OUT/(name+'.csv'),index=False)


if __name__ == '__main__':
    main(sys.argv[1])

