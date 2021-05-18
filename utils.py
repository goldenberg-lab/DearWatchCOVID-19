import pandas as pd
import numpy as np
import os
import glob
import sklearn.metrics


def x_not_missing(x, by='SCORE'):
        x=x[~x[by.replace('_score', '')].isna()]
#         if by =='covid_score':
#             print(len(x), len(x[~x[by].isna()]))
        return x[~x[by].isna()]

def sens(target, pred):
    pred_orig=pred.copy()
    pred=pred[~np.isnan(target)]
    target=target[~np.isnan(target)]
    target=target[~np.isnan(pred)].astype(np.int32)
    pred=pred[~np.isnan(pred)].astype(np.int32)

    if all(np.asarray(target)==0): return np.nan
    if all(target==pred)&all(target==0):return np.nan # everything is tn
    if all(target==pred)&all(target==1):return 1 # everything is tp
    if all(target!=pred)&all(target==0):return np.nan # everything is fp
    if all(target!=pred)&all(target==1):return 0 # everything is fn
    try:
        tn, fp, fn, tp =  sklearn.metrics.confusion_matrix(target, pred).ravel()
    except:
            print(np.unique(target, return_counts=True))
            print(np.unique(pred, return_counts=True))
            print(np.unique(pred_orig, return_counts=True))
            print(target)
            print(pred)
            print(len(target), len(pred))
            print('cm')
            print(sklearn.metrics.confusion_matrix(target, pred))
            print(sklearn.metrics.confusion_matrix(target, pred).ravel())
            raise
    return tp/(tp+fn)

def spec(target, pred):
    pred=pred[~np.isnan(target)]
    target=target[~np.isnan(target)]
    target=target[~np.isnan(pred)].astype(np.int32)
    pred=pred[~np.isnan(pred)].astype(np.int32)

    if all(target==pred)&all(target==0):return 1 # everything is tn
    if all(target==pred)&all(target==1):return np.nan # everything is tp
    if all(target!=pred)&all(target==0):return 0 # everything is fp
    if all(target!=pred)&all(target==1):return np.nan # everything is fn
    tn, fp, fn, tp =  sklearn.metrics.confusion_matrix(target, pred).ravel()
    return 1-fp/(fp+tn)

def ppv(target, pred):
    pred=pred[~np.isnan(target)]
    target=target[~np.isnan(target)]
    target=target[~np.isnan(pred)].astype(np.int32)
    pred=pred[~np.isnan(pred)].astype(np.int32)

    if all(target==pred)&all(target==0):return np.nan # everything is tn
    if all(target==pred)&all(target==1):return 1 # everything is tp
    if all(target!=pred)&all(target==0):return 0 # everything is fp
    if all(target!=pred)&all(target==1):return np.nan # everything is fn
    tn, fp, fn, tp =  sklearn.metrics.confusion_matrix(target, pred).ravel()

    return tp/(tp+fp)

def safe_AUC(target, score):
    score=score[~np.isnan(target)]
    target=target[~np.isnan(target)].astype(np.int32)
    assert not(any(np.isnan(target)))
    if len(target)==0:
        return np.nan
    if len(set(target))==1:
        return np.nan
    try:
        return sklearn.metrics.roc_auc_score(target, score)
    except:
        print(set(target))
        print()
        raise

def safe_AUPR(target, score):
    score=score[~np.isnan(target)]
    target=target[~np.isnan(target)].astype(np.int32)
    if len(target)==0:
        return np.nan
    if len(set(target))==1:
        return np.nan
    return sklearn.metrics.average_precision_score(target, score)

def outputs(x, target='ili'):
    survey_target='covid_survey' if 'covid_survey' in x.columns else 'covid'
    survey_suffix = '_survey' if 'covid_survey' in x.columns else ''
    d={'AUC': safe_AUC(x_not_missing(x, by=target+'_score')[target].values, x_not_missing(x, by=target+'_score')[target+'_score'].values),
       'AUPR':safe_AUPR(x_not_missing(x, by=target+'_score')[target].values, x_not_missing(x, by=target+'_score')[target+'_score'].values),
       'sens. @ 0.98 spec': sens(x[target].values, x[target+'_pred_98_spec'].values),
       'spec. @ 0.98 spec': spec(x[target].values, x[target+'_pred_98_spec'].values),
       'ppv. @ 0.98 spec': ppv(x[target].values, x[target+'_pred_98_spec'].values),
       'sens. @ 0.98 sens': sens(x[target].values, x[target+'_pred_98_sens'].values),
       'spec. @ 0.98 sens': spec(x[target].values, x[target+'_pred_98_sens'].values),
       'ppv. @ 0.98 sens': ppv(x[target].values, x[target+'_pred_98_sens'].values),
       'sens. @ 0.7 sens': sens(x[target].values, x[target+'_pred_70_sens'].values),
       'spec. @ 0.7 sens': spec(x[target].values, x[target+'_pred_70_sens'].values),
       'ppv. @ 0.7 sens':ppv(x[target].values, x[target+'_pred_70_sens'].values),
       'Survey AUC': safe_AUC(x_not_missing(x, by='covid_score'+survey_suffix)['covid'+survey_suffix].values, x_not_missing(x, by='covid_score'+survey_suffix)['covid_score'+survey_suffix].values),
       'Survey AUPR':safe_AUPR(x_not_missing(x, by='covid_score'+survey_suffix)['covid'+survey_suffix].values, x_not_missing(x, by='covid_score'+survey_suffix)['covid_score'+survey_suffix].values),
       'Survey sens. @ 0.98 spec': sens(x['covid'+survey_suffix].values, x['covid_pred_98_spec'+survey_suffix].values),
       'Survey spec. @ 0.98 spec': spec(x['covid'+survey_suffix].values, x['covid_pred_98_spec'+survey_suffix].values),
       'Survey ppv. @ 0.98 spec': ppv(x['covid'+survey_suffix].values, x['covid_pred_98_spec'+survey_suffix].values),
       'Survey sens. @ 0.98 sens': sens(x['covid'+survey_suffix].values, x['covid_pred_98_sens'+survey_suffix].values),
       'Survey spec. @ 0.98 sens': spec(x['covid'+survey_suffix].values, x['covid_pred_98_sens'+survey_suffix].values),
       'Survey ppv. @ 0.98 sens': ppv(x['covid'+survey_suffix].values, x['covid_pred_98_sens'+survey_suffix].values),
       'Survey sens. @ 0.7 sens': sens(x['covid'+survey_suffix].values, x['covid_pred_70_sens'+survey_suffix].values),
       'Survey spec. @ 0.7 sens': spec(x['covid'+survey_suffix].values, x['covid_pred_70_sens'+survey_suffix].values),
       'Survey ppv. @ 0.7 sens':ppv(x['covid'+survey_suffix].values, x['covid_pred_70_sens'+survey_suffix].values),       
       'Medically diagnosed influenza as covid': sum(x.loc[x['flu_covid'].isin([1]), 'covid_pred_70_sens'].values)/len(x.loc[x['flu_covid'].isin([1]), 'covid_pred_70_sens'].values),
       'Unspecified ILI as covid':1-spec(x.loc[~x['flu_covid'].isin([1,2]), target].values, x.loc[~x['flu_covid'].isin([1,2]), 'covid_pred_70_sens'].values),
       'Combined sens. @ 0.98 spec': sens(x['covid'+survey_suffix].values, np.logical_and(x['covid_pred_98_spec'+survey_suffix].values, x[target+'_pred_70_sens'].values)),
       'Combined spec. @ 0.98 spec': spec(x['covid'+survey_suffix].values, np.logical_and(x['covid_pred_98_spec'+survey_suffix].values, x[target+'_pred_70_sens'].values)),
       'Combined ppv. @ 0.98 spec': ppv(x['covid'+survey_suffix].values, np.logical_and(x['covid_pred_98_spec'+survey_suffix].values, x[target+'_pred_70_sens'].values)),
       'Combined sens. @ 0.98 sens': sens(x['covid'+survey_suffix].values, np.logical_and(x['covid_pred_98_sens'+survey_suffix].values, x[target+'_pred_70_sens'].values)),
       'Combined spec. @ 0.98 sens': spec(x['covid'+survey_suffix].values, np.logical_and(x['covid_pred_98_sens'+survey_suffix].values, x[target+'_pred_70_sens'].values)),
       'Combined ppv. @ 0.98 sens': ppv(x['covid'+survey_suffix].values, np.logical_and(x['covid_pred_98_sens'+survey_suffix].values, x[target+'_pred_70_sens'].values)),
       'Combined sens. @ 0.7 sens': sens(x['covid'+survey_suffix].values, np.logical_and(x['covid_pred_70_sens'+survey_suffix].values, x[target+'_pred_70_sens'].values)),
       'Combined spec. @ 0.7 sens': spec(x['covid'+survey_suffix].values, np.logical_and(x['covid_pred_70_sens'+survey_suffix].values, x[target+'_pred_70_sens'].values)),
       'Combined ppv. @ 0.7 sens':ppv(x['covid'+survey_suffix].values, np.logical_and(x['covid_pred_70_sens'+survey_suffix].values, x[target+'_pred_70_sens'].values)),
       'Number of Surveys': np.sum(x['ili_pred_70_sens'].values) if target=='ili' else 0,
       'Survey Rates': np.sum(x['ili_pred_70_sens'].values)/len(x) if target=='ili' else 0,
       'Number of Surveys COVID': np.sum(x['covid_score'].values>0) if target=='ili' else 0,
       'Survey Rates COVID': np.sum(x['covid_score'].values>0)/len(x) if target=='ili' else 0,
      }
    return pd.Series(d)


def participant_date_fill(df_in):
    """
    Make sure irregular indices are made to be regular
    """
#     print('reindexing')
    assert 'participant_id' in df_in.index.names
    assert 'date' in df_in.index.names
    min_inds= df_in.reset_index('date').groupby('participant_id').min().set_index(['date'], append=True).index
    max_inds= df_in.reset_index('date').groupby('participant_id').max().set_index(['date'], append=True).index

    person, min_i = zip(*sorted(min_inds.tolist()))
    person, max_i = zip(*sorted(max_inds.tolist()))
    all_inds=zip(person, min_i, max_i)

    result_inds=[]
    for ind in all_inds:
        dates = pd.date_range(start=ind[1], end=ind[2])
        result_inds+=list(zip([ind[0]]* len(dates), list(dates)))
    result_inds = pd.MultiIndex.from_tuples(result_inds, names = ['participant_id', 'date'])        
    
    df_in = df_in.reindex(result_inds)
    return df_in

    
def read_results(base_path, wearable_path, survey_path, save_name=None, target='ili', agg_days=0):
    """
    """
    if save_name is None:
        save_name='*estset_results.csv'
    files=sorted(glob.glob(os.path.join(wearable_path,save_name), recursive=True))
    assert len(files)>0, os.path.join(wearable_path, save_name)
    dfs=[]
    for f in files:
        try:
#             assert any([name in f for name in ['bret/', 'bret1/','bret2/','bret3/','bret4/','bret5/']])
            assert any([name in f for name in ['bret1_prosp_val/','bret2_prosp_val/','bret3_prosp_val/','bret4_prosp_val/','bret5_prosp_val/', 'dec1_prosp_val/','prosp_val2/','prosp_val3/','prosp_val4/','prosp_val5/']]), print("could not find ", f)
            for name in ['bret1_prosp_val/','bret2_prosp_val/','bret3_prosp_val/','bret4_prosp_val/','bret5_prosp_val/', 'dec1_prosp_val/','prosp_val2/','prosp_val3/','prosp_val4/','prosp_val5/']:
                if name in f:
                    date=f.split(name)[1].split('/')[0].replace('_', '-')
            date
        except:
            date=f.split('dec1/')[1].split('/')[0].replace('_', '-')
            
        if pd.to_datetime(date) < pd.to_datetime('2020-02-07'):
            # do this for speed
            continue
        if pd.to_datetime(date) >pd.to_datetime('2020-06-04'):
            # do this for speed
            break
#         if ('2020-05' in date):
#             # for debug
#             pass
#         else:
#             continue
        temp_df = pd.read_csv(f)
        temp_df=temp_df[[col for col in temp_df.columns if 'unnamed' not in col.lower()]]
        temp_df.columns = [col if col!='predicted' else target+'_score'  for col in temp_df.columns]
#         print(temp_df.head())
        try:
#             print(date, temp_df['date'].min(), temp_df.date.max())
            temp_df['date'].min()
        except:
#             print(temp_df.head())
            raise
#         print(date, temp_df['date'].min(), temp_df.date.max())
        if any(['unnamed' in col.lower() for col in temp_df.columns]):
            temp_df.columns=['participant_id', 'date', temp_df.iloc[1]['label'], 'predicted']
            temp_df=temp_df.iloc[2:]
        
#                     Unnamed: 0 	Unnamed: 1 	label 	label.1
#                 0 	NaN 	NaN 	ili 	predicted
#                 1 	participant_id 	date 	NaN 	NaN
        # remove base_path, then split'/' and take first one
        
        temp_df['date']=pd.to_datetime(temp_df['date'])

        temp_df.set_index(['participant_id', 'date'], inplace=True)
        
#         print('max date', temp_df.index.get_level_values('date').max())
        
        
        # fill the index for the dates of this week
        additional_index = pd.MultiIndex.from_product([list(set(temp_df.index.get_level_values('participant_id'))), pd.date_range(start=date, periods=7)], names=['participant_id', 'date'])
        
        new_index = temp_df.index.union(additional_index)
#         print(len(temp_df.index), len(additional_index), len(new_index))
        temp_df = temp_df.reindex(new_index)
        temp_df=temp_df.sort_index()
        
        # forward fill the prediction values and labels
        ffill_cols=[col for col in temp_df.columns if ('score' not in col.lower())]
        temp_df[ffill_cols]=temp_df[ffill_cols].ffill()
        temp_df[ffill_cols]=temp_df[ffill_cols].fillna(0)
        
        # if agg_days!=0 then carry scores forward
        if agg_days!=0:
            print(temp_df.head())
            temp_df[target] = temp_df[target].replace( {0:np.nan}).groupby( 'participant_id').ffill(limit=agg_days).fillna(0)
                
        
#         temp_df=temp_df.loc[(temp_df.index.get_level_values('date')>=date)&(temp_df.index.get_level_values('date')<=(pd.to_datetime(date)+pd.Timedelta('6D')))]
#         print('num indices', np.sum((temp_df.index.get_level_values('date')>=date)))
#         display(temp_df.head())
#         display(temp_df.loc[(temp_df.index.get_level_values('date')>=pd.to_datetime(date))].head())
#         temp_df2=temp_df.loc[(temp_df.index.get_level_values('date')>=pd.to_datetime(date))]
#         print('first half', temp_df2['ili_score'].isna().mean())
#         temp_df2=temp_df.loc[(temp_df.index.get_level_values('date')>=date)]
#         print('no_datetime', temp_df2['ili_score'].isna().mean())

        temp_df = temp_df.loc[((temp_df.index.get_level_values('date')>=pd.to_datetime(date)) & (temp_df.index.get_level_values('date')<=(pd.to_datetime(date)+pd.Timedelta(days=6))))]
#         print('1', temp_df['ili_score'].isna().mean())
        temp_df['model_date']=date
#         print('2', temp_df['ili_score'].isna().mean())
        dfs.append(temp_df)
        
    df=pd.concat(dfs)
#     print(df['ili_score'].isna().mean())
    
        
    files=sorted(glob.glob(os.path.join(survey_path,'covid_testset_results.csv'), recursive=True))
    assert len(files)>0, os.path.join(survey_path,'covid_testset_results.csv')
    survey_dfs=[]
    for f in files:
        try:
            assert any([name in f for name in ['bret1_prosp_val/','bret2_prosp_val/','bret3_prosp_val/','bret4_prosp_val/','bret5_prosp_val/','dec1_prosp_val/','prosp_val2/','prosp_val3/','prosp_val4/','prosp_val5/']])
            for name in ['bret1_prosp_val/','bret2_prosp_val/','bret3_prosp_val/','bret4_prosp_val/','bret5_prosp_val/','dec1_prosp_val/','prosp_val2/','prosp_val3/','prosp_val4/','prosp_val5/']:
                if name in f:
                    date=f.split(name)[1].split('/')[0].replace('_', '-')
            date
        except:
            date=f.split('dec1/')[1].split('/')[0].replace('_', '-')
        if pd.to_datetime(date) < pd.to_datetime('2020-02-07'):
            # do this for speed
            continue
        if pd.to_datetime(date) >pd.to_datetime('2020-06-04'):
            # do this for speed
            break
#         if ('2020-05' in date):
#             # for debug
#             pass
#         else:
#             continue
        temp_df = pd.read_csv(f)
        # remove base_path, then split'/' and take first one
        
        temp_df['date']=pd.to_datetime(temp_df['date'])
#         print(date, temp_df['date'].min(), temp_df.date.max())


        
        temp_df.set_index(['participant_id', 'date'], inplace=True)
        
        # fill the index for the dates of this week
        additional_index = pd.MultiIndex.from_product([list(set(temp_df.index.get_level_values('participant_id'))), pd.date_range(start=date, periods=7)], names=['participant_id', 'date'])
        new_index = temp_df.index.union(additional_index)
        temp_df = temp_df.reindex(new_index)
        temp_df=temp_df.sort_index()

        
        # forward fill the prediction values and labels
        ffill_cols=[col for col in temp_df.columns if ('score' not in col.lower())]
#         print(ffill_cols)
        temp_df[ffill_cols]=temp_df[ffill_cols].ffill()
        temp_df[ffill_cols]=temp_df[ffill_cols].fillna(0)
                
        
        temp_df=temp_df.loc[(temp_df.index.get_level_values('date')>=date)&(temp_df.index.get_level_values('date')<=(pd.to_datetime(date)+pd.Timedelta('6D')))]

        survey_dfs.append(temp_df)
        
    survey_df=pd.concat(survey_dfs)
    
#     print('before')
#     print(df.head())
#     print(df['ili'].isna().mean(), survey_df['covid'].isna().mean())
    
    df=df.join(survey_df, on=['participant_id', 'date'], how='outer', rsuffix='_survey')
    df=df.sort_index()
    cols = [col for col in ['ili', 'covid', 'covid_survey'] if col in df.columns]
    df[cols] = df.groupby('participant_id')[cols].apply(lambda x:x.ffill())
    df[cols] = df.groupby('participant_id')[cols].apply(lambda x: x.fillna(0))
    
#     print(df.head())
#     print(df['ili'].isna().mean(), df['covid'].isna().mean())
    
#     print(df.loc[df['ili'].isna()])
#     print(df.loc[df['covid'].isna()])
    

#     print('after')
    
    return df