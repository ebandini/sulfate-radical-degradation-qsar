# -*- coding: utf-8 -*-
"""
RDKit descriptor-selection analyses for the ln k QSAR.
Answers "how many descriptors are needed?" empirically:
  1. STABILITY SELECTION over 200 resamples -> which descriptors are robust;
  2. NESTED-CV SCAN of descriptor count k -> how many are needed (performance plateau);
  3. LOO performance for a fold-wise f-test/Ridge procedure and a full-data
     Ridge fit on the stability-selected set (training fit is descriptive only).
Outputs: Fig_R4_DescriptorCount.png, Fig_R5_StabilitySelection.png, Fig_R6_FinalParity.png,
         feature_selection_results.json
"""
import json, warnings, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

DATA='Final SMILES list 33-Zhongda.csv'
df=pd.read_csv(DATA,sep=';')
required={'SMILES','lnk'}
missing=required.difference(df.columns)
if missing:
    raise ValueError(f'{DATA} is missing required columns: {sorted(missing)}')
if df[['SMILES','lnk']].isna().any().any():
    raise ValueError(f'{DATA} contains missing SMILES or lnk values')
df['lnk']=pd.to_numeric(df['lnk'],errors='coerce')
if not np.isfinite(df['lnk']).all():
    raise ValueError(f'{DATA} contains non-numeric or non-finite lnk values')
names=[x[0] for x in Descriptors._descList]
calc=MoleculeDescriptors.MolecularDescriptorCalculator(names)
rows,idx,invalid=[],[],[]
for i,smi in enumerate(df['SMILES']):
    m=Chem.MolFromSmiles(smi)
    if m is not None: rows.append(calc.CalcDescriptors(m)); idx.append(i)
    else: invalid.append(i+1)
if invalid: warnings.warn(f'Skipping invalid SMILES on CSV data rows: {invalid}',stacklevel=1)
if not rows: raise ValueError('No valid molecules were found in the input CSV')
X=pd.DataFrame(rows,columns=names,index=idx).replace([np.inf,-np.inf],np.nan)
feat=np.array(X.columns); Xv=X.values; y=df.loc[idx,'lnk'].astype(float).values
loo=LeaveOneOut()
def cv(est):
    yp=np.asarray(cross_val_predict(est,Xv,y,cv=loo,n_jobs=-1)).ravel()
    return r2_score(y,yp),float(np.sqrt(mean_squared_error(y,yp))),float(mean_absolute_error(y,yp))

# ---- 1. stability selection ----
rng=np.random.default_rng(42); n,p=Xv.shape; ITERS=200; m=int(round(n*0.75))
fc=np.zeros(p); lc=np.zeros(p); eligible=np.zeros(p)
for _ in range(ITERS):
    s=rng.choice(n,m,replace=False); ys=y[s]
    Xi=SimpleImputer(strategy='median',keep_empty_features=True).fit_transform(Xv[s])
    vt=VarianceThreshold(0.0); Xi=vt.fit_transform(Xi); support=np.flatnonzero(vt.get_support())
    eligible[support]+=1
    Xz=StandardScaler().fit_transform(Xi)
    fsel=SelectKBest(f_regression,k=min(10,Xz.shape[1])).fit(Xz,ys)
    fc[support[fsel.get_support()]]+=1
    lasso=LassoCV(cv=3,n_alphas=30,max_iter=5000,random_state=0).fit(Xz,ys)
    lc[support[np.abs(lasso.coef_)>1e-8]]+=1
ff=np.divide(fc,eligible,out=np.zeros_like(fc),where=eligible>0)
lf=np.divide(lc,eligible,out=np.zeros_like(lc),where=eligible>0)
avg=(ff+lf)/2; topidx=np.argsort(-avg)[:12]

# ---- 2. how many descriptors? nested-CV scan ----
ks=range(1,16); ols_q,rid_q=[],[]
for k in ks:
    prep=[('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('vt',VarianceThreshold(0.0)),('sc',StandardScaler())]
    ols_q.append(cv(Pipeline(prep+[('sel',SelectKBest(f_regression,k=k)),('m',LinearRegression())]))[0])
    rid_q.append(cv(Pipeline(prep+[('sel',SelectKBest(f_regression,k=k)),('m',RidgeCV(alphas=np.logspace(-2,3,20)))]))[0])

# ---- 3. CV ridge with fold-wise f-test selection; full-data stable set is descriptive ----
K=10; sel=list(topidx[:K]); Xs=Xv[:,sel]
prep=[('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('vt',VarianceThreshold(0.0)),('sc',StandardScaler())]
infold=Pipeline(prep+[('sel',SelectKBest(f_regression,k=K)),('m',RidgeCV(alphas=np.logspace(-2,3,30)))])
yp_final=np.asarray(cross_val_predict(infold,Xv,y,cv=loo,n_jobs=-1)).ravel()
q2,rmse,mae=r2_score(y,yp_final),np.sqrt(mean_squared_error(y,yp_final)),mean_absolute_error(y,yp_final)
imputer=SimpleImputer(strategy='median',keep_empty_features=True).fit(Xs); Xsi=imputer.transform(Xs)
scaler=StandardScaler().fit(Xsi); Xsz=scaler.transform(Xsi)
ridge=RidgeCV(alphas=np.logspace(-2,3,30)).fit(Xsz,y)
print('Top stability-selected descriptors (full dataset; descriptive):',list(feat[sel]))
print('Ridge-10 with fold-wise f-test selection, honest LOO: Q2=%.3f RMSE=%.3f MAE=%.3f'%(q2,rmse,mae))
print('Full-data ridge fit on the stability-selected set: training R2=%.3f (not a predictive estimate)'%r2_score(y,ridge.predict(Xsz)))
json.dump({'stability_frequency':{feat[i]:{'f_test':float(ff[i]),'lasso':float(lf[i]),'mean':float(avg[i])} for i in topidx},
           'stability_selected_top10':list(feat[sel]),
           'scan_ols_q2_loo':list(ols_q),'scan_ridge_q2_loo':list(rid_q),
           'foldwise_ftest_ridge10_loo':[float(q2),float(rmse),float(mae)],
           'full_data_stable_ridge':{'training_r2':float(r2_score(y,ridge.predict(Xsz))),
                                     'alpha':float(ridge.alpha_),
                                     'intercept':float(ridge.intercept_),
                                     'standardized_coefficients':{feat[i]:float(c) for i,c in zip(sel,ridge.coef_)}}},
          open('feature_selection_results.json','w'),indent=2)

# ---- figures ----
plt.rcParams.update({'font.size':11,'axes.grid':True,'grid.alpha':.3,'figure.dpi':200,'axes.axisbelow':True})
fig,ax=plt.subplots(figsize=(7.2,5))
ax.plot(list(ks),ols_q,'o-',color='#4C72B0',label='OLS (selection in-fold)')
ax.plot(list(ks),rid_q,'s-',color='#DD8452',label='Ridge (selection in-fold)')
ax.axhline(0,color='#888',ls='--',lw=1,label='Q2 = 0 reference'); ax.axvspan(9,13,color='#CDE0CD',alpha=.5,zorder=0)
ax.axvline(10,color='#2E7D32',lw=1.2,ls='-.',alpha=.8); ax.set_xticks(list(ks))
ax.set_ylim(min(0,min(ols_q+rid_q))-.1,max(0,max(ols_q+rid_q))+.1)
ax.set_xlabel('Number of descriptors (k)'); ax.set_ylabel('Q² (leave-one-out)')
ax.set_title('How many descriptors are needed?\nNested-CV performance vs descriptor count'); ax.legend(loc='lower right',fontsize=9)
plt.tight_layout(); plt.savefig('Fig_R4_DescriptorCount.png',bbox_inches='tight'); plt.close()

nm=[feat[i] for i in topidx][::-1]; ffv=[ff[i] for i in topidx][::-1]; lfv=[lf[i] for i in topidx][::-1]
yp=np.arange(len(nm)); h=0.4; fig,ax=plt.subplots(figsize=(7.6,5.2))
ax.barh(yp+h/2,ffv,h,color='#4C72B0',label='f-test top-10 (freq.)'); ax.barh(yp-h/2,lfv,h,color='#55A868',label='Lasso non-zero (freq.)')
ax.set_yticks(yp); ax.set_yticklabels(nm,fontsize=9); ax.set_xlim(0,1); ax.set_xlabel('Selection frequency across 200 resamples')
ax.set_title('Stability of RDKit descriptor selection'); ax.legend(loc='lower right',fontsize=9)
plt.tight_layout(); plt.savefig('Fig_R5_StabilitySelection.png',bbox_inches='tight'); plt.close()

fig,ax=plt.subplots(figsize=(5.6,5.4)); lo,hi=min(y.min(),yp_final.min())-.3,max(y.max(),yp_final.max())+.3
ax.plot([lo,hi],[lo,hi],'k--',lw=1.3,alpha=.7); ax.scatter(y,yp_final,c='#DD8452',edgecolors='k',s=60,alpha=.85)
ax.set_xlim(lo,hi); ax.set_ylim(lo,hi); ax.set_aspect('equal'); ax.set_xlabel('Experimental ln k'); ax.set_ylabel('LOO-predicted ln k')
ax.set_title('Ridge with fold-wise f-test selection (10 RDKit descriptors)')
ax.text(.04,.96,'Q²(LOO) = %.2f\nRMSE = %.2f\nMAE = %.2f'%(q2,rmse,mae),transform=ax.transAxes,va='top',bbox=dict(boxstyle='round',fc='white',ec='gray',alpha=.9),fontsize=10)
plt.tight_layout(); plt.savefig('Fig_R6_FinalParity.png',bbox_inches='tight'); plt.close()
print('wrote figures + feature_selection_results.json')

