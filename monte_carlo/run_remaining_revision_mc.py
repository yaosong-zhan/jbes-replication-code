"""Run all remaining Monte Carlo experiments at the requested replication count."""
import os
for name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:
    os.environ[name]='1'
os.environ['PYTHONIOENCODING']='utf-8'
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).parent

def run(name, *args):
    with (ROOT/'Results'/f'{name}_2000.log').open('w',encoding='utf-8') as log:
        subprocess.run([sys.executable,'-u',str(ROOT/(name+'.py')),*args],
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    print(f'Completed {name}',flush=True)

if __name__=='__main__':
    jobs=[('comprehensive_simulation','--skip-size'),('sim_supplement',),
          ('sim_robust_table',),('sim_factor_dgp',),('sim_jbes_benchmark',),
          ('sim_revision_supplement',)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        fs=[pool.submit(run,*job) for job in jobs]
        for f in as_completed(fs):
            f.result()

