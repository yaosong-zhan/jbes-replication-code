"""Reproduce every SA-8 table and the raw-L2 comparison with 2,000 draws."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from simulation_joint_change import (generate_data, shape_pca, build_detection_vec,
    cusum_test, run_full_pipeline, run_classification_mc)

N_REPS=2000
OUT=Path(__file__).parent/'Results'

def main():
    rows=[]
    for phi in [0.,.3,.5]:
        counts={m:0 for m in [.5,1.,2.]}
        base=int(np.floor(2000**(1/3)))
        for rep in range(N_REPS):
            Z,_,_=generate_data(2000,11,3,'none',.5,0.,error_type='ar',ar_coeff=phi,seed=rep*1000+2000)
            U,_,_=shape_pca(Z[:,:-1],q_method='fixed',fixed_q=5)
            W=build_detection_vec(Z[:,:-1],Z[:,-1],U)
            for m in counts:
                counts[m]+=int(cusum_test(W,.05,bandwidth=max(1,int(m*base)))[1])
        for m,c in counts.items():
            rows.append(dict(study='bandwidth',phi=phi,multiplier=m,bandwidth=int(m*base),value=c/N_REPS,n_reps=N_REPS))
        print('Bandwidth',phi,counts,flush=True)
    for psi in [-.5,0.,.5]:
        count=0
        for rep in range(N_REPS):
            Z,_,_=generate_data(2000,11,3,'none',.5,0.,error_type='ma',ma_coeff=psi,seed=rep*1000+2000)
            count+=int(run_full_pipeline(Z,.05,q_method='fixed',fixed_q=5)['reject'])
        rows.append(dict(study='ma',psi=psi,value=count/N_REPS,n_reps=N_REPS))
        print('MA',psi,count,flush=True)
    for q in [2,3,4,5,6]:
        count=0
        for rep in range(N_REPS):
            Z,_,_=generate_data(1000,11,3,'shape-driven',.5,1.,ar_coeff=.3,seed=rep*1000+1000)
            count+=int(run_full_pipeline(Z,.05,q_method='fixed',fixed_q=q)['reject'])
        rows.append(dict(study='q_power',q=q,value=count/N_REPS,n_reps=N_REPS))
    pd.DataFrame(rows).to_csv(OUT/'sim_sa8.csv',index=False)
    results={}
    for name,T,snr,norm,err,phi in [('classif_ar',2000,2.,'l2_corrected','ar',.3),
                                 ('raw_l2_500',500,.1,'l2','iid',0.),
                                 ('raw_l2_1000',1000,.1,'l2','iid',0.)]:
        results[name]=run_classification_mc(T,11,3,snr,N_REPS,.05,norm_type=norm,
                         q_method='fixed',fixed_q=5,error_type=err,ar_coeff=phi)
        print(name,results[name],flush=True)
    (OUT/'sim_sa8_classification.json').write_text(json.dumps(results,indent=2),encoding='utf-8')

if __name__=='__main__':
    main()

