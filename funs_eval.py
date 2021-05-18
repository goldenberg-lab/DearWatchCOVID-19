"""
SMALL HELPER FUNCTIONS
"""

import os
from time import time
from scipy import stats
import numpy as np
import pandas as pd
from colorspace.colorlib import HCL
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import warnings
import itertools

from sklearn.metrics import roc_auc_score as auc

# y, score, groups = df_woy.query('has_woy==False').ili.values, df_woy.query('has_woy==False').ili_score.values, df_woy.query('has_woy==False').woy.values
# Function to decompose the within/between/aggregate AUC
def auc_decomp(y, score, groups):
    assert y.shape[0] == score.shape[0] == groups.shape[0]
    idx1, idx0 = np.where(y == 1)[0], np.where(y == 0)[0]
    # Calculate number of pairs
    npairs_agg = len(idx1) * len(idx0)
    auc_agg = auc(y, score)
    ugroups = np.unique(groups)
    # --- Calculate within AUC --- #
    df = pd.DataFrame({'y':y,'score':score,'groups':groups})
    dat_within = df.groupby(['groups','y']).size().reset_index().rename(columns={0:'n'})
    ugroups_w = dat_within.groups.value_counts().reset_index().query('groups==2').iloc[:,0].values
    dat_within = dat_within.query('groups.isin(@ugroups_w)',engine='python').reset_index(None,True)
    dat_within = dat_within.groupby('groups').apply(lambda x: x['n'].iloc[0]*x['n'].iloc[1]).reset_index().rename(columns={0:'npair'})
    npairs_within = dat_within.npair.sum()
    within_auc = df.query('groups.isin(@ugroups_w)',engine='python').groupby('groups').apply(lambda x: auc(x.y, x.score))
    within_auc = within_auc.reset_index().rename(columns={0:'auc'})
    within_auc = within_auc.merge(dat_within)
    auc_within = np.sum(within_auc.auc * within_auc.npair) / npairs_within
    # --- Calculate the between AUC --- #
    mat_between = np.zeros(dat_within.shape)
    for ii, group in enumerate(dat_within.groups):
        if (ii + 1) % 10 == 0:
            print(ii+1)
        score1 = df.query('groups==@group & y==1').score.values
        score0 = df.query('groups!=@group & y==0').score.values
        npair = len(score1) * len(score0)
        aucb = auc(np.append(np.repeat(1,len(score1)), np.repeat(0,len(score0))), np.append(score1, score0))
        mat_between[ii] = [npair, aucb]
    dat_between = pd.DataFrame(mat_between,columns=['npair','auc']).assign(groups=dat_within.groups)
    dat_between.npair = dat_between.npair.astype(int)
    npairs_between = dat_between.npair.sum()
    assert npairs_between + npairs_within == npairs_agg
    auc_between = np.sum(dat_between.auc * dat_between.npair) / npairs_between
    res = pd.DataFrame({'tt':['agg','within','between'],
                       'auc':[auc_agg, auc_within, auc_between],
                       'npair':[npairs_agg, npairs_within, npairs_between]})
    return res, within_auc, dat_between

# Wrapper for column/row vectors
def rvec(x):
    return np.atleast_2d(x)

def cvec(x):
    return np.atleast_2d(x).T

# Join lists
def ljoin(x):
    return(list(itertools.chain.from_iterable(x)))

# Function to return the predicted/actual from a chi-squared test
def chi_process(long, cc):
    assert long.columns.isin([cc,'metric','n']).sum()==3
    pt = long.pivot_table(index='metric',columns=cc,values='n',aggfunc=np.sum)
    res = stats.chi2_contingency(pt)
    dat_exp = pd.DataFrame(res[3],columns=pt.columns,index=pt.index).T.reset_index().melt(cc)
    dat_exp.rename(columns={'value':'expected'},inplace=True)
    dat_act = pt.T.reset_index().melt(cc).rename(columns={'value':'actual'})
    dat_res = dat_act.merge(dat_exp).assign(stat=res[0],pval=res[1],dof=res[2])
    return dat_res

# Function to calculate AUROC
def multi_auroc(inp):
    pid, horizon, tt = inp[0]
    df = inp[1]
    auc = auroc(df.ili, df.score)
    vec = pd.DataFrame({'id': pid, 'horizon': horizon, 'tt': tt, 'auroc': auc}, index=[0])
    return vec


# Function for calculating odds ratio and inference
def fun_oddsr(tab, to_df=False):
    if isinstance(tab, pd.DataFrame):
        tab = tab.values
    if np.any(np.isnan(tab)) | tab.shape[1]==1:
        oddsr, pval, se = np.NaN, np.NaN, np.NaN
    else:
        tab = np.where(tab == 0, 0.5, tab)  # Remove zeros (continuity correction)
        oddsr = np.log((tab[0, 0] / tab[1, 0]) / (tab[0, 1] / tab[1, 1]))
        se = np.sqrt(np.sum(1 / tab.flatten()))
        pval = 2 * (1 - stats.norm.cdf(np.abs(oddsr) / se))
    if to_df:
        vec = pd.Series({'oddsr':oddsr, 'pval':pval, 'se':se})
        return vec
    else:
        return oddsr, pval, se


def auroc(ytrue, ypred):
    idx = ypred.notnull()
    try:
        res = roc_auc_score(ytrue[idx], ypred[idx])
    except:
        res = np.NaN
    return res


def auprc(ytrue, ypred):
    idx = ypred.notnull()
    try:
        res = average_precision_score(ytrue[idx], ypred[idx])
    except:
        res = np.NaN
    return res


def gg_color_hue(n):
    hues = np.linspace(15, 375, num=n + 1)[:n]
    hcl = []
    for h in hues:
        hcl.append(HCL(H=h, L=65, C=100).colors()[0])
    return hcl


# ---- FUNCTION TO LOAD IN THE RESULTS ---- #
def res_loader(dat_fn, dir_folder, fn_test, target='ili'):
    assert target in ['ili', 'covid']
    
    warnings.filterwarnings('ignore')
    holder_rs, holder_ind, holder_all = [], [], []
    lst_attr = dat_fn.columns.drop('fn').to_list()
    cn_idx1 = ['id', 'date', 'horizon']
    cn_idx2 = ['id','horizon','tt']
    for ii, rr in dat_fn.iterrows():
        #print('File %i of %i' % (ii+1, dat_fn.shape[0]))
        fn = rr['fn']
        fold = os.path.join(dir_folder, fn)
        path = os.path.join(fold, fn_test)
        assert os.path.exists(path)
        print('Getting full data')
        dat_ii = pd.read_csv(path).rename(columns={'participant_id':'id'})
        # Make the "horizon" clear in the label
        dat_ii.rename(columns={target:target+'_0',target+'_score':target+"_0_score"}, inplace=True)
        if 'tt' in dat_ii.columns:
            dat_ii.drop(columns=['tt'], inplace=True)
        #print(dat_ii.head()); print('\n')
        cidx_score = dat_ii.columns.str.contains('score')
        tmp = pd.Series(dat_ii.columns.str.replace(target+'_', '').to_list())
        tmp = tmp.str.split('_', 1, True).fillna('na').iloc[:, [1, 0]].apply(lambda x: '_'.join(x), 1)
        dat_ii.columns = np.where(cidx_score, tmp, dat_ii.columns)
        #print(dat_ii.head()); print('\n')
        tmp_idx = pd.MultiIndex.from_frame(dat_ii[['id','date']])
        dat_ii.drop(columns=['id','date'],inplace=True)
        dat_ii.index = tmp_idx
        tmp = pd.Series(dat_ii.columns.to_list()).str.split('_',1,True)
        dat_ii.columns = pd.MultiIndex.from_frame(tmp)
        dat_ii = dat_ii.reset_index().melt(['id','date']).rename(columns={0:'tmp',1:'horizon'})
        dat_ii = dat_ii.pivot_table('value',cn_idx1,'tmp').reset_index()
        dat_ii[[target,'horizon']] = dat_ii[[target,'horizon']].astype(int)
        dat_ii = dat_ii.sort_values(cn_idx1).reset_index(None, True)
        for a in lst_attr:
            dat_ii.insert(dat_ii.shape[1], a, rr[a])
        if rr['task'] == 'timetoevent':
            assert np.all( dat_ii.groupby('id')[target].sum() == 1 )
        if rr['task'] == 'missing':
            print('----- NO TASK! CREATING TIMETOEVENT VERSION -----')
            forecast_ii = dat_ii.groupby(['id','horizon']).apply(lambda x: x[:x.reset_index(None,True)[target].idxmax()+1])
            forecast_ii.columns = forecast_ii.columns.to_list()
            forecast_ii = forecast_ii.reset_index(None,True).assign(task='timetoevent')
            assert np.all( forecast_ii.groupby('id')[target].sum() == 1 )
            dat_ii = pd.concat([dat_ii.assign(task='allevent'), forecast_ii],0).reset_index(None,True)
        print('Getting aggregate results')
        res_ii = dat_ii.groupby(['horizon']+lst_attr).apply(lambda x:
                             pd.Series({'auroc':auroc(x[target], x.score), 'auprc':auprc(x[target], x.score)})).reset_index()
        holder_rs.append(res_ii), holder_all.append(dat_ii)
    # Merge and return
    holder_rs = pd.concat(holder_rs).reset_index(None,True)
    #holder_ind = pd.concat(holder_ind).reset_index(None,True)
    holder_all = pd.concat(holder_all).reset_index(None,True)
    warnings.filterwarnings('default')
    return holder_rs, holder_all

#         if 'tanh_daily_z_rhr' in dat_ii.columns:
#             print('Transforming lancet output')
#             dat_ii = dat_ii.drop(columns=['score']).rename(columns={'label':'ili_0', 'tanh_daily_z_rhr':'score_0'})
#         if ('score' in dat_ii.columns) & ('ili' in dat_ii.columns):
#             print('Transforming lstm/grud')
#             dat_ii.rename(columns={'score':'score_0', 'ili':'ili_0'}, inplace=True)
#         if ('score' in dat_ii.columns) & ('label' in dat_ii.columns):
#             print('Transforming XGboost')
#             dat_ii.rename(columns={'score':'score_0', 'label':'ili_0'}, inplace=True)

#         # Get the performance for forecasting first event

#         all_ii = pd.concat([forecast_ii.assign(tt='forecast'),dat_ii.assign(tt='full')]).reset_index(None, True)
#         print(all_ii.shape)
#         idx_drop = (all_ii.tt=='full') & (all_ii.horizon > 0)
#         print('Removing forecasting from all event\nDropping %i rows' % idx_drop.sum())
#         all_ii = all_ii[~idx_drop].reset_index(None,True)
#         print(all_ii.shape)

        #print(res_ii.head())
        # print('Getting individual results with at least 30 rows')
        # size_ii = all_ii.assign(null=lambda x: x.score.isnull()).groupby(cn_idx2+lst_attr+['null']).size().reset_index().rename(columns={0:'n'})
        # size_ii = size_ii[size_ii.null == False].drop(columns=['null']).reset_index(None,True)
        # size_ii = size_ii[size_ii.n >= 30].drop(columns=['n']).merge(all_ii,'left',cn_idx2 + lst_attr).reset_index(None,True)
        # # Multiprocessing
        # tnow = time()
        # ind_ii = pd.concat(p.map(multi_auroc, size_ii.groupby(cn_idx2))).reset_index(None, True)
        # ind_ii = size_ii.groupby(cn_idx2+lst_attr).size().reset_index().drop(columns=[0]).merge(ind_ii,'left',cn_idx2)
        # print('Multiprocessing took %i seconds' % (time() - tnow))
        # ind_ii.head()
        # Save results