# -*- coding: utf-8 -*-
"""
Reproducible QSAR/QSPR benchmark for ln k (sulfate-radical degradation rate).
Manuscript: "Sulfate radical-based degradation of micropollutants in water..."

Addresses reviewer comments on the modelling:
  * proper hyperparameter tuning by NESTED cross-validation (tuning kept
    separate from performance estimation);
  * every model reported only in its single, tuned form;
  * R2/Q2, RMSE and MAE reported for all models (+ a mean-predictor baseline);
  * an expanded, like-for-like comparison of 10 regressors incl. PLS & Elastic Net;
  * all scaling/feature-selection performed INSIDE the CV folds (no leakage).

Run:  python qsar_benchmark.py
Outputs: final_results.csv, loo_predictions.csv, best_params.json, baseline_results.json,
         Fig_R1_ModelComparison.png, Fig_R2_Parity.png, Fig_R3_Leakage.png
Requires: rdkit, scikit-learn>=1.3, pandas, numpy, matplotlib
"""
import json, warnings, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.model_selection import LeaveOneOut, RepeatedKFold, GridSearchCV, cross_val_predict, cross_validate
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

SEED = 42
DATA = 'dataset.csv'   # columns: No;Name;SMILES;lnk  (';'-separated)

# ---------------------------------------------------------------- 1. data
df = pd.read_csv(DATA, sep=';')
required = {'Name', 'SMILES', 'lnk'}
missing = required.difference(df.columns)
if missing:
    raise ValueError(f'{DATA} is missing required columns: {sorted(missing)}')
if df[['SMILES', 'lnk']].isna().any().any():
    raise ValueError(f'{DATA} contains missing SMILES or lnk values')
df['lnk'] = pd.to_numeric(df['lnk'], errors='coerce')
if not np.isfinite(df['lnk']).all():
    raise ValueError(f'{DATA} contains non-numeric or non-finite lnk values')
names = [x[0] for x in Descriptors._descList]
calc = MoleculeDescriptors.MolecularDescriptorCalculator(names)
rows, idx, invalid = [], [], []
for i, smi in enumerate(df['SMILES']):
    m = Chem.MolFromSmiles(smi)
    if m is not None:
        rows.append(calc.CalcDescriptors(m)); idx.append(i)
    else:
        invalid.append(i + 1)
if invalid:
    warnings.warn(f'Skipping invalid SMILES on CSV data rows: {invalid}', stacklevel=1)
if not rows:
    raise ValueError('No valid molecules were found in the input CSV')
X = pd.DataFrame(rows, columns=names, index=idx)
y = df.loc[idx, 'lnk'].astype(float).values
names_cmpd = df.loc[idx, 'Name'].tolist()
X = X.replace([np.inf, -np.inf], np.nan)
feat = list(X.columns); Xv = X.values
X_full_for_count = SimpleImputer(strategy='median', keep_empty_features=True).fit_transform(Xv)
p_full = int(VarianceThreshold(0.0).fit(X_full_for_count).get_support().sum())
print(f'N = {len(y)} compounds | {Xv.shape[1]} raw RDKit descriptors | '
      f'{p_full} non-constant across the full dataset (descriptive count; filtering is fitted per CV fold)')

# ---------------------------------------------------------------- 2. models
def PREP(scale=True):
    # Fit imputation and variance filtering on each training fold only.
    steps = [('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
             ('vt', VarianceThreshold(0.0))]
    if scale:
        steps.append(('sc', StandardScaler()))
    return steps

k3 = [5, 10, 20]
MODELS = {
 'OLS (all descriptors)': (Pipeline(PREP()+[('m',LinearRegression())]), {}),
 'OLS (k-best, tuned)':   (Pipeline(PREP()+[('sel',SelectKBest(f_regression)),('m',LinearRegression())]),
                           {'sel__k':[3,5,8]}),
 'Ridge':      (Pipeline(PREP()+[('m',Ridge())]), {'m__alpha':np.logspace(-2,4,13)}),
 'Lasso':      (Pipeline(PREP()+[('m',Lasso(max_iter=50000))]), {'m__alpha':np.logspace(-3,1,13)}),
 'ElasticNet': (Pipeline(PREP()+[('m',ElasticNet(max_iter=50000))]),
                {'m__alpha':np.logspace(-3,1,7),'m__l1_ratio':[.2,.5,.8]}),
 'PLS':        (Pipeline(PREP()+[('m',PLSRegression())]), {'m__n_components':list(range(1,11))}),
 'kNN':        (Pipeline(PREP()+[('sel',SelectKBest(f_regression)),('m',KNeighborsRegressor())]),
                {'sel__k':k3,'m__n_neighbors':[3,5,7],'m__weights':['uniform','distance']}),
 'SVR (RBF)':  (Pipeline(PREP()+[('sel',SelectKBest(f_regression)),('m',SVR(kernel='rbf'))]),
                {'sel__k':k3,'m__C':[1,10,100],'m__gamma':['scale',0.01,0.1],'m__epsilon':[0.1,0.3]}),
 'Random Forest': (Pipeline(PREP(scale=False)+[('m',RandomForestRegressor(n_estimators=120,random_state=SEED))]),
                {'m__max_depth':[3,None],'m__min_samples_leaf':[2]}),
 'Gradient Boosting': (Pipeline(PREP(scale=False)+[('m',GradientBoostingRegressor(random_state=SEED,max_depth=2))]),
                {'m__n_estimators':[100],'m__learning_rate':[0.03,0.05,0.1]}),
}

loo   = LeaveOneOut()
inner = RepeatedKFold(n_splits=5, n_repeats=1,  random_state=SEED)  # hyperparameter tuning
outer = RepeatedKFold(n_splits=5, n_repeats=5,  random_state=SEED)  # robustness estimate

# ---------------------------------------------------------------- 3. nested CV
primary, robust, best_params, loo_pred = [], [], {}, {'compound':names_cmpd, 'y_true':y.tolist()}
for name, (pipe, grid) in MODELS.items():
    # Keep inner searches serial; the outer CV call is already parallelized.
    est = pipe if not grid else GridSearchCV(pipe, grid, scoring='neg_mean_squared_error', cv=inner, n_jobs=1)
    # (a) nested LOO -> out-of-sample predictions
    yp = np.asarray(cross_val_predict(est, Xv, y, cv=loo, n_jobs=-1)).ravel()
    loo_pred[name] = yp.tolist()
    primary.append({'Model':name, 'Q2_LOO':r2_score(y,yp),
                    'RMSE_LOO':float(np.sqrt(mean_squared_error(y,yp))),
                    'MAE_LOO':float(mean_absolute_error(y,yp))})
    # (b) nested repeated 5x5-fold -> robustness
    cv = cross_validate(est, Xv, y, cv=outer, n_jobs=-1,
                        scoring={'r2':'r2','rmse':'neg_root_mean_squared_error','mae':'neg_mean_absolute_error'})
    robust.append({'Model':name, 'R2_mean':cv['test_r2'].mean(), 'R2_sd':cv['test_r2'].std(),
                   'RMSE_mean':float((-cv['test_rmse']).mean()), 'RMSE_sd':float((-cv['test_rmse']).std()),
                   'MAE_mean':float((-cv['test_mae']).mean()),  'MAE_sd':float((-cv['test_mae']).std())})
    # (c) tuned hyperparameters (fit on full data, inner CV only)
    if grid:
        gs = GridSearchCV(pipe, grid, scoring='neg_mean_squared_error', cv=inner, n_jobs=1).fit(Xv, y)
        best_params[name] = {k:(v.item() if hasattr(v,'item') else v) for k,v in gs.best_params_.items()}
    else:
        best_params[name] = {}
    print(f'  {name:22s} Q2={primary[-1]["Q2_LOO"]:+.3f}  RMSE={primary[-1]["RMSE_LOO"]:.3f}  '
          f'MAE={primary[-1]["MAE_LOO"]:.3f}  | rep-CV R2={robust[-1]["R2_mean"]:+.3f}')

res = pd.DataFrame(primary).merge(pd.DataFrame(robust), on='Model')
res['params'] = res['Model'].map(lambda k: json.dumps(best_params[k]))
res.to_csv('final_results.csv', index=False, float_format='%.8g')
json.dump(best_params, open('best_params.json','w'), indent=2)

# baseline (predict the mean)
yb = cross_val_predict(DummyRegressor(strategy='mean'), Xv, y, cv=loo).ravel()
base_rmse = float(np.sqrt(mean_squared_error(y, yb)))
base_q2 = float(r2_score(y, yb)); base_mae = float(mean_absolute_error(y,yb))
print(f'\nMean-predictor baseline: Q2={base_q2:.3f}  RMSE={base_rmse:.3f}  MAE={base_mae:.3f}')
aromaticity = np.array([int(any(a.GetIsAromatic() for a in Chem.MolFromSmiles(s).GetAtoms()))
                           for s in df.loc[idx,'SMILES']], dtype=float)
aromaticity_tpsa = np.column_stack([aromaticity, X['TPSA'].to_numpy(dtype=float)])
simple_model = Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),
                         ('sc',StandardScaler()),('m',LinearRegression())])
yp_simple = np.asarray(cross_val_predict(simple_model,aromaticity_tpsa,y,cv=loo,n_jobs=-1)).ravel()
simple_metrics = {'Q2_LOO':float(r2_score(y,yp_simple)),
                  'RMSE_LOO':float(np.sqrt(mean_squared_error(y,yp_simple))),
                  'MAE_LOO':float(mean_absolute_error(y,yp_simple))}
loo_pred['Aromaticity + TPSA'] = yp_simple.tolist()
pd.DataFrame(loo_pred).to_csv('loo_predictions.csv', index=False)
print('Aromaticity + TPSA baseline: Q2={Q2_LOO:.3f}  RMSE={RMSE_LOO:.3f}  MAE={MAE_LOO:.3f}'.format(**simple_metrics))
json.dump({'mean_predictor':{'Q2_LOO':base_q2,'RMSE_LOO':base_rmse,'MAE_LOO':base_mae},
           'aromaticity_tpsa':simple_metrics},open('baseline_results.json','w'),indent=2)

# ---------------------------------------------------------------- 4. leakage demonstration
X_full = SimpleImputer(strategy='median', keep_empty_features=True).fit_transform(Xv)
X_full = VarianceThreshold(0.0).fit_transform(X_full)
sel = SelectKBest(f_regression, k=3).fit(X_full, y)                 # selection on FULL data
yp_leak = cross_val_predict(Pipeline(PREP()+[('m',LinearRegression())]), sel.transform(X_full), y, cv=loo).ravel()
right   = Pipeline(PREP()+[('sel',SelectKBest(f_regression,k=3)),('m',LinearRegression())])
yp_ok   = cross_val_predict(right, Xv, y, cv=loo).ravel()          # selection INSIDE each fold
q2_leak, q2_ok = r2_score(y, yp_leak), r2_score(y, yp_ok)
print(f'Leakage demo (3-descriptor OLS): select-on-full-data Q2={q2_leak:.3f}  vs  select-in-fold Q2={q2_ok:.3f}')

# ---------------------------------------------------------------- 5. figures
plt.rcParams.update({'font.size':11,'axes.grid':True,'grid.alpha':.3,'figure.dpi':200,'axes.axisbelow':True})
order = list(MODELS.keys())
res['o']=res['Model'].map({m:i for i,m in enumerate(order)}); res=res.sort_values('o')
short={'OLS (all descriptors)':'OLS\n(all)','OLS (k-best, tuned)':'OLS\n(k-best)','Ridge':'Ridge','Lasso':'Lasso',
       'ElasticNet':'Elastic\nNet','PLS':'PLS','kNN':'kNN','SVR (RBF)':'SVR','Random Forest':'Random\nForest','Gradient Boosting':'Gradient\nBoosting'}
lab=[short[m] for m in res['Model']]
lin=np.array([m in ['OLS (all descriptors)','OLS (k-best, tuned)','Ridge','Lasso','ElasticNet','PLS'] for m in res['Model']])
col=['#4C72B0' if l else '#C44E52' for l in lin]; xr=np.arange(len(res))

fig,ax=plt.subplots(1,2,figsize=(13,5.2))
ax[0].bar(xr,res['Q2_LOO'],color=col,edgecolor='k',lw=.6)
ax[0].errorbar(xr,res['R2_mean'],yerr=res['R2_sd'],fmt='D',color='#333',ms=5,lw=1,capsize=3,label='Repeated 5×5-fold R² (mean±SD)')
ax[0].axhline(0,color='k',lw=1); ax[0].set_ylabel('Predictive R²'); ax[0].set_ylim(-1.3,1)
ax[0].set_xticks(xr); ax[0].set_xticklabels(lab,fontsize=8.5); ax[0].legend(loc='lower right',fontsize=9)
ax[0].set_title('(a) Predictive R²: Q² by LOO (bars) vs repeated CV (points)')
ax[1].bar(xr,res['RMSE_LOO'],color=col,edgecolor='k',lw=.6)
ax[1].errorbar(xr,res['RMSE_mean'],yerr=res['RMSE_sd'],fmt='D',color='#333',ms=5,lw=1,capsize=3,label='Repeated 5×5-fold RMSE')
ax[1].axhline(base_rmse,color='#666',ls='--',lw=1.5,label=f'Mean-predictor baseline (RMSE={base_rmse:.2f})')
ax[1].set_ylabel('RMSE (ln k units)'); ax[1].set_xticks(xr); ax[1].set_xticklabels(lab,fontsize=8.5)
ax[1].legend(loc='upper right',fontsize=9); ax[1].set_title('(b) RMSE by LOO (bars) vs repeated CV (points)')
from matplotlib.patches import Patch
fig.legend(handles=[Patch(fc='#4C72B0',ec='k',label='Linear / regularized'),Patch(fc='#C44E52',ec='k',label='Non-linear (kernel/tree)')],
           loc='lower center',ncol=2,fontsize=9,bbox_to_anchor=(0.5,-0.02))
plt.tight_layout(rect=[0,0.05,1,1]); plt.savefig('Fig_R1_ModelComparison.png',bbox_inches='tight'); plt.close()

def parity(ax,name,title,c):
    yp=np.array(loo_pred[name]); q2=r2_score(y,yp); rmse=np.sqrt(mean_squared_error(y,yp)); mae=mean_absolute_error(y,yp)
    lo,hi=min(y.min(),yp.min())-.3,max(y.max(),yp.max())+.3
    ax.plot([lo,hi],[lo,hi],'k--',lw=1.3,alpha=.7); ax.scatter(y,yp,c=c,edgecolors='k',s=60,alpha=.85)
    ax.set_xlim(lo,hi); ax.set_ylim(lo,hi); ax.set_aspect('equal')
    ax.set_xlabel('Experimental ln k'); ax.set_ylabel('LOO-predicted ln k'); ax.set_title(title)
    ax.text(.04,.96,f'Q²(LOO) = {q2:.2f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}',transform=ax.transAxes,va='top',
            bbox=dict(boxstyle='round',fc='white',ec='gray',alpha=.9),fontsize=10)
fig,ax=plt.subplots(1,2,figsize=(11,5.3))
parity(ax[0],'Gradient Boosting','(a) Gradient Boosting (best overall)','#C44E52')
parity(ax[1],'Lasso','(b) Lasso (interpretable, regularized)','#4C72B0')
plt.tight_layout(); plt.savefig('Fig_R2_Parity.png',bbox_inches='tight'); plt.close()

fig,ax=plt.subplots(figsize=(6.2,5))
b=ax.bar(['Preprocessing + selection\non FULL data,\nthen LOO\n(leaked)','Preprocessing + selection\nINSIDE each\nLOO fold\n(honest)'],
         [q2_leak,q2_ok],color=['#C44E52','#4C72B0'],edgecolor='k',width=.6)
for r,v in zip(b,[q2_leak,q2_ok]): ax.text(r.get_x()+r.get_width()/2,v+.01,f'Q²={v:.2f}',ha='center',fontweight='bold')
ax.axhline(0,color='k',lw=1); ax.set_ylabel('Q² (LOO)')
ax.set_ylim(min(0,q2_leak,q2_ok)-.05,max(0,q2_leak,q2_ok)+.05)
ax.set_title('Feature selection outside vs inside the CV loop\n(3-descriptor OLS, same data)')
plt.tight_layout(); plt.savefig('Fig_R3_Leakage.png',bbox_inches='tight'); plt.close()
print('\nWrote final_results.csv, loo_predictions.csv, best_params.json and Fig_R1/R2/R3.')

