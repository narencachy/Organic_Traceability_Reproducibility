import importlib.util, math, os
import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

spec=importlib.util.spec_from_file_location('bench','/mnt/data/organic_traceability_reproducible_benchmark.py')
bench=importlib.util.module_from_spec(spec); spec.loader.exec_module(bench)

def ci95(vals):
    arr=np.asarray(vals,float); mean=float(arr.mean()); se=float(arr.std(ddof=1)/math.sqrt(len(arr))); h=float(t.ppf(.975,len(arr)-1)*se); return mean,mean-h,mean+h

rows=[]
for seed in bench.TRIAL_SEEDS:
    df=bench.generate_events(seed, bench.N_EVENTS); y=df.noncompliant.to_numpy(); split=int(len(df)*bench.TRAIN_FRAC)
    weighted=[]; temp=[]; anyv=[]
    for r in df.itertuples(index=False):
        e=bench.row_to_event(r); rule=bench.RULES[e['stage']]
        weighted.append(bench.compliance_score(e))
        tx=bench.excess(e['temperature_c'],*rule['temp']); temp.append(tx)
        v=tx>0 or bench.excess(e['humidity_pct'],*rule['humidity'])>0 or (('soil' in rule) and e['soil_moisture_pct'] is not None and bench.excess(e['soil_moisture_pct'],*rule['soil'])>0) or e['certificate_valid']==0 or e['custody_gap']==1 or e['seal_breach']==1
        anyv.append(int(v))
    weighted=np.asarray(weighted); temp=np.asarray(temp); anyv=np.asarray(anyv)
    thr_w=bench.choose_threshold(weighted[:split],y[:split]); thr_t=bench.choose_threshold(temp[:split],y[:split])
    methods=[('temperature-only',(temp[split:]>=thr_t).astype(int),temp[split:]),('any-violation',anyv[split:],anyv[split:]),('weighted-score',(weighted[split:]>=thr_w).astype(int),weighted[split:])]
    for name,pred,score in methods:
        rows.append({'seed':seed,'method':name,'accuracy':accuracy_score(y[split:],pred),'precision':precision_score(y[split:],pred,zero_division=0),'recall':recall_score(y[split:],pred,zero_division=0),'f1':f1_score(y[split:],pred,zero_division=0),'auc':roc_auc_score(y[split:],score)})

df=pd.DataFrame(rows)
out=[]
for method in ['temperature-only','any-violation','weighted-score']:
    row={'method':method}; sub=df[df.method==method]
    for m in ['accuracy','precision','recall','f1','auc']:
        mean,lo,hi=ci95(sub[m]); row[m]=mean; row[m+'_lo']=lo; row[m+'_hi']=hi
    out.append(row)
out=pd.DataFrame(out)
os.makedirs('/mnt/data/organic_traceability_results',exist_ok=True)
df.to_csv('/mnt/data/organic_traceability_results/classifier_baseline_trials.csv',index=False)
out.to_csv('/mnt/data/organic_traceability_results/classifier_baseline_summary.csv',index=False)
print(out.to_string(index=False))
