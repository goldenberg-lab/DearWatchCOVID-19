import pandas as pd
import numpy as np

import pickle
import os
from multiprocessing import Pool
from functools import partial

from dataclasses import dataclass
from run_model import load_data
from constants import *
from utils import *

@dataclass
class Args():
    def __init__(self, all_survey=False, regularly_sampled=False, only_healthy = False ):
        self.all_survey = all_survey
        self.regularly_sampled = regularly_sampled
        self.only_healthy = only_healthy
        
        

        
def read_fold(item, flu_covid_col=None, time_to_onset_col=None, save_name='', target='ili', tts=False, agg_days=0, participants=None):
    """
    global object for parallelisatoin
    """
    k,v = item
    print('Fold ', k)
    if participants is not None: print(len(participants))
    wearable_path=os.path.join(v, '*/test')
    survey_path=os.path.join(v, '*/out_gru_survey_WOY')

    temp_df = read_results(v, wearable_path, survey_path, save_name=save_name, target=target, agg_days=agg_days)
    if participants is not None:
        temp_df = temp_df.loc[ temp_df.index.get_level_values('participant_id').isin(participants)]
    temp_df = temp_df.join(flu_covid_col, how='left', on=['participant_id', 'date'])
    #merge time to onset
    temp_df = temp_df.join(time_to_onset_col, how='left', on=['participant_id', 'date'])
    if participants is not None:
        assert all([t in participants for t in temp_df.index.get_level_values('participant_id')])
    # for weekly results
    if tts:
        result_temp = temp_df.fillna(0)
        # looking only at time from -2 to 9 days, accumulate positive cases, and save those for
        survey_target='covid_survey' if 'covid_survey' in result_temp.columns else 'covid'
        survey_suffix = '_survey' if 'covid_survey' in result_temp.columns else ''
        result_temp['covid_pred'] = np.logical_and(result_temp['covid_pred_70_sens'+survey_suffix].values, result_temp[target+'_pred_70_sens'].values)




        onset_region = (result_temp['time_to_onset']>=-2)&(result_temp['time_to_onset']<=9)
        assert 'participant_id' in result_temp.index.names
        assert 'date' in result_temp.index.names
        result_temp.sort_index(inplace=True)
        result_temp.loc[onset_region, 'covid_pred'] = result_temp.loc[onset_region, 'covid_pred'].groupby('participant_id').cumsum()
        result_temp.loc[onset_region, target+'_pred_70_sens'] = result_temp.loc[onset_region, target+'_pred_70_sens'].groupby('participant_id').cumsum()

        result_temp['covid_pred']=result_temp['covid_pred'].apply(lambda x: max(x,1))
        result_temp[target+'_pred_70_sens']=result_temp[target+'_pred_70_sens'].apply(lambda x: max(x,1))

        # add fold to index
        result_temp['fold']=k
        result_temp.set_index('fold', append=True)

    else:
        result_temp = temp_df.fillna(0).groupby('model_date').apply(lambda x: outputs(x, target=target))
#         result_temp = temp_df.loc[~temp_df[target+'_score'].isna()].groupby('model_date').apply(lambda x: outputs(x))
        result_temp.index=pd.MultiIndex.from_product([result_temp.index.tolist(), [k]], names=['model_date', 'fold'])
    return result_temp
            
            
            
            
def read_fluvey(dfs, agg_days=0, tts=False, participants=None, n_cpu=1, covid=False):

    target='ili'
    base_path = {1:'/datasets/evidationdata/ili_split_nov16_bret1_prosp_val',
                2:'/datasets/evidationdata/ili_split_nov16_bret2_prosp_val',
                3:'/datasets/evidationdata/ili_split_nov16_bret3_prosp_val',
                4:'/datasets/evidationdata/ili_split_nov16_bret4_prosp_val',
                5:'/datasets/evidationdata/ili_split_nov16_bret5_prosp_val',
                }


#     save_name='ili_xgb_prosp_testset_results_WOY_mm1_bounded.csv'

    save_name='ili_xgb_prosp_testset_results_WOY_reg_mm1_bounded.csv'
    save_name='ili_xgb_prosp_testset_results_WOY_reg_mm1.csv'
#     save_name='iligrud_prosp_testset_results_WOY_reg_mm1.csv'
    
    if covid:
        target='covid'
        base_path = {1:'/datasets/evidationdata/covid_split_dec1_prosp_val',
                    2:'/datasets/evidationdata/covid_split_dec1_prosp_val2',
                    3:'/datasets/evidationdata/covid_split_dec1_prosp_val3',
                    4:'/datasets/evidationdata/covid_split_dec1_prosp_val4',
                    5:'/datasets/evidationdata/covid_split_dec1_prosp_val5',
                    }
        save_name='covid_xgb_prosp_testset_results_WOY_reg_mm1.csv'

    # survey_path=os.path.join(base_path, '*/out_linear01_survey_WOY')
    # survey_path=os.path.join(base_path, '*/out_linear_last7_survey_WOY')
    if isinstance(base_path, dict):
        if n_cpu==1:
            result_df=None
            for k, v in base_path.items():
                print('Fold ', k)
                if participants is not None:print(len(participants))
                wearable_path=os.path.join(v, '*/test')
                survey_path=os.path.join(v, '*/out_gru_survey_WOY')

                temp_df = read_results(v, wearable_path, survey_path, save_name=save_name, target=target, agg_days=agg_days)
                if participants is not None:
                    print(1, len(set(temp_df.index.get_level_values('participant_id'))))
                    temp_df = temp_df.loc[ temp_df.index.get_level_values('participant_id').isin(participants)]
                print(2, len(set(temp_df.index.get_level_values('participant_id'))))
                temp_df = temp_df.join(dfs['survey']['flu_covid'], how='left', on=['participant_id', 'date'])
                print(3, len(set(temp_df.index.get_level_values('participant_id'))))
                #merge time to onset
                try:
                    temp_df = temp_df.join(dfs['survey']['time_to_onset'], how='left', on=['participant_id', 'date'])
                except:
                    print(dfs['survey'].columns.tolist())
                    raise
                if participants is not None:
                    assert all([t in participants for t in temp_df.index.get_level_values('participant_id')])
                print(4, len(set(temp_df.index.get_level_values('participant_id'))))
                # for weekly results
                if tts:
                    result_temp = temp_df.fillna(0)
                    # looking only at time from -2 to 9 days, accumulate positive cases, and save those for
                    #target='ili'
                    survey_target='covid_survey' if 'covid_survey' in result_temp.columns else 'covid'
                    survey_suffix = '_survey' if 'covid_survey' in result_temp.columns else ''
                    result_temp['covid_pred'] = np.logical_and(result_temp['covid_pred_70_sens'+survey_suffix].values, result_temp[target+'_pred_70_sens'].values)




                    onset_region = (result_temp['time_to_onset']>=-2)&(result_temp['time_to_onset']<=9)
                    assert 'participant_id' in result_temp.index.names
                    assert 'date' in result_temp.index.names
                    result_temp.sort_index(inplace=True)
                    result_temp.loc[onset_region, 'covid_pred'] = result_temp.loc[onset_region, 'covid_pred'].groupby('participant_id').cumsum()
                    result_temp.loc[onset_region, target+'_pred_70_sens'] = result_temp.loc[onset_region, target+'_pred_70_sens'].groupby('participant_id').cumsum()

                    result_temp['covid_pred']=result_temp['covid_pred'].apply(lambda x: max(x,1))
                    result_temp[target+'_pred_70_sens']=result_temp[target+'_pred_70_sens'].apply(lambda x: max(x,1))

                    # add fold to index
                    result_temp['fold']=k
                    result_temp.set_index('fold', append=True)

                else:
                    result_temp = temp_df.fillna(0).groupby('model_date').apply(lambda x: outputs(x, target=target))
            #         result_temp = temp_df.loc[~temp_df[target+'_score'].isna()].groupby('model_date').apply(lambda x: outputs(x))
                    result_temp.index=pd.MultiIndex.from_product([result_temp.index.tolist(), [k]], names=['model_date', 'fold'])
                if result_df is None:
                    result_df = result_temp
                else:
                    result_df = pd.concat((result_df, result_temp), axis=0)
        else:
            #n_cpu>1
            
            
            # make partial, with target, save_name, agg_days, participants, flu_covid_col, and tts            
            with Pool(n_cpu) as p:
                temp_dfs = p.map(partial(read_fold, target=target, save_name=save_name, agg_days=agg_days, participants=participants, tts=tts, flu_covid_col=dfs['survey']['flu_covid'], time_to_onset_col=dfs['survey']['time_to_onset']), [(k,v) for k,v in base_path.items()])
                
            result_df = pd.concat(temp_dfs, axis=0)
            
                
                
#         result_df = result_df.groupby('model_date').agg({c:['mean', 'std', 'count'] for c in result_df.columns.tolist()})
    else:
        wearable_path=os.path.join(base_path, '*/test')
        survey_path=os.path.join(base_path, '*/out_gru_survey_WOY')

        temp_df = read_results(base_path, wearable_path, survey_path, save_name=save_name, target=target, agg_days=agg_days)
        # participant_fill

        # # print(dfs['survey'].columns.tolist())
        temp_df = temp_df.join(dfs['survey']['flu_covid'], how='left', on=['participant_id', 'date'])
        # for weekly results
        result_df = temp_df.loc[~temp_df[target+'_score'].isna()].groupby('model_date').apply(lambda x: outputs(x))
        # result_df = temp_df.fillna(0).groupby('model_date').apply(lambda x: outputs(x))

    try:
        result_df = result_df.drop(0, axis=0)
    except:
        pass

    return result_df
        

    
    
    
    
    
    
def main():
    args = Args(all_survey=False, regularly_sampled=True)

    with open(DATA_PATH_DICTIONARY_FILE, 'rb') as f:
        tmp = pickle.load(f)
    path = tmp[GET_PATH_DICT_KEY(args)]

    print('loading from: ',path)
    d = os.path.dirname(path)
    f = os.path.basename(path)
    dfs=load_data(d, regular=args.regularly_sampled, only_healthy=args.only_healthy, fname=f, load_activity=False)
    
    if 'all_survey' not in path:
        print("There are this many participants in FLUCOVID: ", len(set(dfs['survey'].loc[dfs['survey'].index.get_level_values('date')<'2020-06-01'].index.get_level_values('participant_id'))))
        
        covid_participants = set(dfs['survey'].loc[(dfs['survey'].index.get_level_values('date')<'2020-06-01')&(dfs['survey']['covid']==1)].index.get_level_values('participant_id'))
        print("there are this many COVID participants: ", len(covid_participants))
              
        flu_participants = set(dfs['survey'].loc[(dfs['survey'].index.get_level_values('date')<'2020-06-01')&(dfs['survey']['flu_covid']==1)].index.get_level_values('participant_id')) - covid_participants
        print("there are this many FLU (non-covid) participants: ", len(flu_participants))
        
        unspecified_ili_participants = set(dfs['survey'].loc[(dfs['survey'].index.get_level_values('date')<'2020-06-01') & (dfs['survey']['flu_covid'].isin([0]))&(dfs['survey']['ili']==1)].index.get_level_values('participant_id')) - covid_participants - flu_participants
        print("there are this many unspecified ILI participants: ", len(unspecified_ili_participants))
        
        #dfs['survey']['event_start']= (dfs['survey']['ili'].diff()).fillna(dfs['survey']['ili']).astype(int).clip(lower=0, upper=1)
        dfs['survey']['event_start']= (dfs['survey']['ili'].groupby('participant_id').diff()).fillna(dfs['survey']['ili']).astype(int).clip(lower=0, upper=1)
        
        
        num_events = dfs['survey'].loc[(dfs['survey'].index.get_level_values('date')<'2020-06-01'), 'event_start'].groupby('participant_id').apply(lambda x:x.sum()>1).sum()
        
        
        print("This many people had multiple events: ", num_events)
                                                 

    
    
#     print(dfs['survey'].head())
    
#     print(sorted(dfs['survey'].columns.tolist()))
#     # subset to april1-june1
#     tmp = dfs['survey'].loc[(dfs['survey'].index.get_level_values('date')>='2020-04-01')&(dfs['survey'].index.get_level_values('date')<='2020-06-01')&(dfs['survey']['ili']==1)]

#     print("This many people reported symptoms:", len(set(tmp.index.get_level_values('participant_id'))))
    
# #     print('covid__diagnosed')
# #     print(np.unique(tmp.groupby('participant_id').max()['covid__diagnosed'].values ,return_counts=True))
    
# #     print('covid__diagnosis_method__nasal_swab')
# #     print(np.unique(tmp.groupby('participant_id').max()['covid__diagnosis_method__nasal_swab'].values ,return_counts=True))
    
# #     print('covid__diagnosis_method__throat_swab')
# #     print(np.unique(tmp.groupby('participant_id').max()['covid__diagnosis_method__throat_swab'].values ,return_counts=True))
    
# #     print('flu_covid')
# #     print(np.unique(tmp.groupby('participant_id').max()['flu_covid'].values ,return_counts=True))
    
# #     print('medical__diagnosis_method__throat_swab')
# #     print(np.unique(tmp.groupby('participant_id').max()['medical__diagnosis_method__throat_swab'].values ,return_counts=True))
    
# #     print('medical__diagnosis_method__nasal_swab')
# #     print(np.unique(tmp.groupby('participant_id').max()['medical__diagnosis_method__nasal_swab'].values ,return_counts=True))
    
# #     print('medical__diagnosed')
# #     print(np.unique(tmp.groupby('participant_id').max()['medical__diagnosed'].values ,return_counts=True))
# #     #covid__diagnosis_method__nasal_swab, covid__diagnosis_method__throat_swab, flu_covid, medical__diagnosis_method__throat_swab, medical__diagnosis_method__nasal_swab, medical__diagnosed
    
    
    
    
    
#     tmp.loc[:,'influenza'] = np.logical_and(tmp['medical__diagnosed'].values.astype(int) ,tmp['ili'].values.astype(int))
    
# #     print('influenza')
# #     print(np.unique(tmp.loc[tmp['ili']==1].groupby('participant_id').max()['influenza'].values ,return_counts=True))
    
#     tmp.loc[:,'influenza_strict'] = np.logical_and(np.logical_and(np.logical_or(tmp['medical__diagnosis_method__nasal_swab'].values.astype(int),tmp['medical__diagnosis_method__throat_swab'].values.astype(int)), tmp['ili'].values.astype(int)), tmp['medical__diagnosed'].values.astype(int))
    
# #     print('influenza_strict')
# #     print(np.unique(tmp.loc[tmp['ili']==1].groupby('participant_id').max()['influenza_strict'].values ,return_counts=True))
    
#     tmp.loc[:,'covid_tested'] = np.logical_or(tmp['covid__diagnosis_method__throat_swab'].values.astype(int), tmp['covid__diagnosis_method__nasal_swab'].values.astype(int))
    
# #     print('covid_tested')
# #     print(np.unique(tmp.loc[tmp['ili']==1].groupby('participant_id').max()['covid_tested'].values.astype(int) ,return_counts=True))
    
#     tmp.loc[:,'covid_positive'] = np.logical_and(np.logical_or(tmp['covid__diagnosis_method__throat_swab'].values.astype(int),tmp['covid__diagnosis_method__nasal_swab'].values.astype(int)),tmp['covid'].values)
    
# #     print('covid_positive')
# #     print(np.unique(tmp.loc[tmp['ili']==1].groupby('participant_id').max()['covid_positive'].values ,return_counts=True))
    
# #     print("This many people got tested for covid:",  )
# #     print("This many people had tested positive:", )
# #     print("This many people had medically diagnosed influenza:", )
    
    
    
    
    
#     ###
#     # find where ILI ==1 and covid_test happens within -2 to +21 days from reported symptom onset date.
#     # 21 days is PCR test +reporting delay
#     tmp.loc[:, 'event_range_ili']=(tmp['ili'].groupby('participant_id').diff()).fillna(tmp['ili']).astype(int).clip(lower=0, upper=1)
#     tmp.loc[:, 'event_range_ili'] = tmp['event_range_ili'].replace({0: np.nan}).groupby('participant_id').bfill(limit=2).groupby('participant_id').ffill(limit=21).fillna(0)
#     tmp.loc[:, 'event_range_covid']=(tmp['covid'].groupby('participant_id').diff()).fillna(tmp['covid']).astype(int).clip(lower=0, upper=1)
#     tmp.loc[:, 'event_range_covid'] = tmp['event_range_covid'].replace({0: np.nan}).groupby('participant_id').bfill(limit=2).groupby('participant_id').ffill(limit=21).fillna(0)
    
    
#     # get participants where they had a covid test, and report NO covid (event_covid_range==0, but have ILI (event_range_ili==1)
#     tmp.loc[:, 'covid_negative'] = np.logical_and(tmp['covid_tested'].values, tmp['event_range_ili'].values-tmp['event_range_covid'].values)
#     # now for each event, we must propogate the covid negative label within the event only.
#     # first, create an event_number for each new ILI event, then groupby participant, event_number, and propogate the max covid negative. If they didn't get tested in an event number range, then the test will not be shown as covid_negative.
#     tmp.loc[:, 'ili_event_number']= (tmp['ili'].groupby('participant_id').diff()).fillna(tmp['ili']).astype(int).clip(lower=0, upper=1).groupby('participant_id').cumsum()
#     # in this instance the event_number should start in the two days preceding symptom onset.   
#     tmp.loc[tmp['ili']==0, 'ili_event_number'] = 0
#     tmp.loc[:, 'ili_event_number'] = tmp.loc[:, 'ili_event_number'].replace({0:np.nan}).groupby('participant_id').bfill(limit=2).groupby('participant_id').ffill().bfill()
#     tmp.set_index(['ili_event_number'], append=True, inplace=True)
    
#     # now that we have the event_number, we can transform groups to be the max of this event_number
#     tmp.loc[:, 'covid_negative'] = tmp['covid_negative'].replace({0, np.nan}).groupby(['participant_id', 'ili_event_number']).bfill().groupby(['participant_id', 'ili_event_number']).ffill().fillna(0)
    
#     # we also need a post covid indicator for immunity
#     dfs['survey'].loc[:, 'after_covid'] = dfs['survey']['covid'].replace({0: np.nan}).groupby('participant_id').ffill().fillna(0)
#     print(tmp.head())
#     print(dfs['survey'].head())
#     tmp.loc[:, 'after_covid']=dfs['survey'].loc[tmp.reset_index('ili_event_number').index, 'after_covid'].values
    
#     # tested_this_event
#     tmp.loc[:, 'tested_this_event'] = tmp['covid_tested'].replace({0:np.nan}).groupby(['participant_id', 'ili_event_number']).bfill().groupby(['participant_id', 'ili_event_number']).ffill().fillna(0)
    
    
#     ###### Everything we need is constructed ########
#     print('*'*40)
#     num_events = set(tmp.loc[(tmp.index.get_level_values('ili_event_number')>0)&(tmp['ili']==1)].reset_index('date').index.tolist())
#     print(sorted(list(num_events))[:10])
#     num_participants = set(tmp.loc[tmp['ili']==1].index.get_level_values('participant_id'))
#     print(f"Taken from {len(num_events)} symptomatic events from {len(num_participants)} distinct participants")
    
#     #### covid tested positive
#     # event_range_covid = 1
#     covid_positive_events = tmp['covid'].groupby(['participant_id', 'ili_event_number']).max().sum()
#     print(f"There are {covid_positive_events} events in which the participant tested positive for covid")
    
#     #### covid tested negative
#     # covid_negative
    
#     covid_negative_events = tmp['covid_negative'].groupby(['participant_id', 'ili_event_number']).max().sum()
#     print(f"There are {covid_negative_events} events in which the participant reported testing negative between -2 and 21 days")
    
#     #### influenza tested positive
#     # covid_negative
    
#     influenza_positive_events = tmp.loc[tmp['flu_covid']==1, 'covid_negative'].groupby(['participant_id', 'ili_event_number']).max().sum()
#     print(f"There are {influenza_positive_events} events in which the participant reported testing positive for influenza between -2 and 21 days")
#     influenza_positive_events = tmp.loc[tmp['flu_covid']==1, 'ili'].groupby(['participant_id', 'ili_event_number']).max().sum()
#     print(f"There are {influenza_positive_events} events in which the participant reported testing positive for influenza between -2 and 21 days")
    
    
#     #### untested
#     # not( after_covid)
#     # and event_range_ili
#     # and not(event_range_covid)
#     # and not(tested_this_event)
    
#     tmp.loc[:, 'untested_events'] = np.logical_and(np.logical_and(1-tmp['after_covid'],
#                                      tmp['event_range_ili']-tmp['event_range_covid']),
#                                      1-tmp['tested_this_event'])
#     untested_events = tmp.loc[tmp['event_range_ili']==1, 'untested_events'].groupby(['participant_id', 'ili_event_number']).max().sum()
#     print(f"There are {untested_events} events in which the participant reported ili symptoms, but did not get a covid test between -2 and 21 days")
    
#     #sanity check
#     untested_events_prime = tmp.loc[tmp['event_range_ili']==1,'untested_events'].groupby(['participant_id', 'ili_event_number']).min().sum()
#     assert untested_events == untested_events_prime, print(untested_events, untested_events_prime)
    
#     # if trouble, het a participant where the min \neq max, then show their trajectory.
    
#     #untested, but recent contact with covid within the 14 days before symptom onset.
#     tmp.loc[:, 'covid__contact_covid_rolling'] = tmp['covid__contact_covid'].replace({0: np.nan}).groupby('participant_id').ffill(limit=14).fillna(0)
#     untested_events = tmp.loc[tmp['covid__contact_covid_rolling']==1, 'untested_events'].groupby(['participant_id', 'ili_event_number']).max().sum()
#     print(f"There are {untested_events} events in which the participant reported ili symptoms, but did not get a covid test between -2 and 21 days DESPITE having contact with a covid case in the 14 days leading up to symptom onset")
                                     
#     raise
    
#      #Exclusively amongst those who did not receive a COVID-19 test, XX\% report close contact with COVID-19, XX\% report , and XX\% report.
        
#      # ili==1, covid==0, covid_negative==0
#     tmp.loc[:,'no_diagnosis']=1 - np.logical_or( tmp['covid__diagnosis_method__throat_swab'].values.astype(int), tmp['covid__diagnosis_method__nasal_swab'].values.astype(int))
#     print("ILI not tested for covid, but contact with covid", tmp.loc[(tmp['ili']==1)&(tmp['covid']==0)&(tmp['no_diagnosis']==1)&(tmp['medical__diagnosed']==0)].groupby('participant_id').max()['covid__contact_covid'].sum())
    
#     print("ILI not tested for covid, but covid__behavior__air_travel", tmp.loc[(tmp['ili']==1)&(tmp['covid']==0)&(tmp['no_diagnosis']==1)&(tmp['medical__diagnosed']==0)].groupby('participant_id').max()['covid__behavior__air_travel'].sum())
    
#     print("ILI not tested for covid, but covid__behavior__large_gatherings", tmp.loc[(tmp['ili']==1)&(tmp['covid']==0)&(tmp['no_diagnosis']==1)&(tmp['medical__diagnosed']==0)].groupby('participant_id').max()['covid__behavior__large_gatherings'].sum())
    
#     print("ILI not tested for covid, but covid__behavior__large_gatherings", tmp.loc[(tmp['ili']==1)&(tmp['covid']==0)&(tmp['no_diagnosis']==1)&(tmp['medical__diagnosed']==0)].groupby('participant_id').max()['covid__behavior__large_gatherings'].sum())
    
    
#     print("ILI not tested for covid, but medical__vaccinated_this_year", tmp.loc[(tmp['ili']==1)&(tmp['covid']==0)&(tmp['no_diagnosis']==1)&(tmp['medical__diagnosed']==0)].groupby('participant_id').max()['medical__vaccinated_this_year'].sum())
    
    
    print('*'*40)
    print('*'*40)
    print('*'*40)
    print(' '*10, 'Results', ' '*30)
    print('*'*40)
    print('*'*40)
    print('*'*40)
    
    # read all results
    
    # direct covid prediction results
    flucovid_results = read_fluvey(dfs, n_cpu=5, covid=True)
    #calculate mean across all results
    sens = flucovid_results['sens. @ 0.7 sens'].mean()
    sens_lower = flucovid_results['sens. @ 0.7 sens'].quantile(0.025)
    sens_upper = flucovid_results['sens. @ 0.7 sens'].quantile(0.975)
    sens_std = flucovid_results['Combined sens. @ 0.7 sens'].std()
    spec = flucovid_results['spec. @ 0.7 sens'].mean()
    spec_lower = flucovid_results['spec. @ 0.7 sens'].quantile(0.025)
    spec_upper = flucovid_results['spec. @ 0.7 sens'].quantile(0.975)
    spec_std = flucovid_results['spec. @ 0.7 sens'].std()
    
    print(f'COVID sensitivity is {sens} ({sens_lower}-{sens_upper}, 95\% CI) (std {sens_std}) and specificity is {spec} ({spec_lower}-{spec_upper}, 95\% CI (std {spec_std})')
    
    
    
    
    flucovid_results = read_fluvey(dfs, n_cpu=5)
    #calculate mean across all results
    sens = flucovid_results['Combined sens. @ 0.7 sens'].mean()
    sens_lower = flucovid_results['Combined sens. @ 0.7 sens'].quantile(0.025)
    sens_upper = flucovid_results['Combined sens. @ 0.7 sens'].quantile(0.975)
    spec = flucovid_results['Combined spec. @ 0.7 sens'].mean()
    spec_lower = flucovid_results['Combined spec. @ 0.7 sens'].quantile(0.025)
    spec_upper = flucovid_results['Combined spec. @ 0.7 sens'].quantile(0.975)
    
    print(f'flucovid sensitivity is {sens} ({sens_lower}-{sens_upper}, 95\% CI) and specificity is {spec} ({spec_lower}-{spec_upper}, 95\% CI')
    
    
    flucovid_results = read_fluvey(dfs, agg_days=7)
    #calculate mean across all results
    sens = flucovid_results['Combined sens. @ 0.7 sens'].mean()
    sens_lower = flucovid_results['Combined sens. @ 0.7 sens'].quantile(0.025)
    sens_upper = flucovid_results['Combined sens. @ 0.7 sens'].quantile(0.975)
    spec = flucovid_results['Combined spec. @ 0.7 sens'].mean()
    spec_lower = flucovid_results['Combined spec. @ 0.7 sens'].quantile(0.025)
    spec_upper = flucovid_results['Combined spec. @ 0.7 sens'].quantile(0.975)
    
    print(f'flucovid cumulative sensitivity is {sens} ({sens_lower}-{sens_upper}, 95\% CI) and specificity is {spec} ({spec_lower}-{spec_upper}, 95\% CI')
    

    
    flucovid_results = read_fluvey(dfs, tts=True, agg_days=7)
    flucovid_results['covid_region'] = flucovid_results['covid'].replace({0, np.nan}).groupby('participant_id').bfill(limit=2).fillna(1)
    flucovid_results['model_date']=pd.to_numeric(pd.to_datetime(flucovid_results['model_date']).dt.second, errors='coerce')#.astype(float)
    print(flucovid_results['fold'].dtype)
    print(flucovid_results['model_date'].dtype)
    print(flucovid_results['covid_pred'].dtype)
    flucovid_results['covid_pred'] = flucovid_results['covid_pred'].apply(pd.to_numeric)
    

              
    for day in range(-2, 7):
        bin_cols = (flucovid_results['covid_region']==1)&(flucovid_results['time_to_onset']==day)
        
#         print(flucovid_results.index.names)
#         print(flucovid_results.columns.tolist())
#         flucovid_results.loc[bin_cols]
#         print(flucovid_results.head())
#         print(flucovid_results.loc[bin_cols, ['fold', 'model_date', 'covid_pred']].groupby(['fold', 'model_date']).mean())#.mean()
#         flucovid_results.loc[bin_cols].groupby(['fold', 'model_date'])[['covid_pred']].mean()#.mean()
        print(flucovid_results.loc[bin_cols, ['fold', 'model_date', 'covid_pred']].groupby(['fold', 'model_date']).mean())

        
        
        percent_detected = flucovid_results.loc[bin_cols, ['fold', 'model_date', 'covid_pred']].groupby(['fold', 'model_date'])['covid_pred'].mean().mean()
        percent_detected_std = flucovid_results.loc[bin_cols, ['fold', 'model_date', 'covid_pred']].groupby(['fold', 'model_date'])['covid_pred'].mean().std()
        
        print(f'by day {day} there are {percent_detected*100} $\pm$ {percent_detected_std*100} \% COVID cases detected')
              
    flucovid_results['non_covid_ili'] = flucovid_results['ili'].values - flucovid_results['covid'].values
    flucovid_results['ili_region'] = flucovid_results['non_covid_ili'].replace({0, np.nan}).groupby('participant_id').bfill(limit=2).fillna(1)
    for day in range(-2, 7):
        bin_cols = (flucovid_results['ili_region']==1)&(flucovid_results['time_to_onset']==day)
        
        percent_detected = flucovid_results.loc[bin_cols, ['fold', 'model_date', 'covid_pred']].groupby(['fold', 'model_date'])['covid_pred'].mean().mean()
        percent_detected_std = flucovid_results.loc[bin_cols, ['fold', 'model_date', 'covid_pred']].groupby(['fold', 'model_date'])['covid_pred'].mean().std()
        print(f'by day {day} there are {percent_detected*100} $\pm$ {percent_detected_std*100} \% of non-COVID ILI cases detected')
        
        
    print('*'*40)
    # now we want to dissect via race and gender
    
    
    for race in ['race_american indian or alaskan native',
     'race_asian or pacific islander',
     'race_black or african american',
     'race_hispanic or latino',
     'race_prefer not to answer',
     'race_white / caucasian']:
        participants = dfs['baseline'].loc[dfs['baseline'][race]==1].index.get_level_values('participant_id')
        use_participants = set(participants).intersection(set(dfs['survey'].index.get_level_values('participant_id')))
        print(race, len(use_participants))
        
        if len(use_participants) < 1000:
            continue
        
        flucovid_results = read_fluvey(dfs, participants=use_participants, n_cpu=5)
        #calculate mean across all results
        sens = flucovid_results['sens. @ 0.7 sens'].mean()
        sens_lower = flucovid_results['sens. @ 0.7 sens'].quantile(0.025)
        sens_upper = flucovid_results['sens. @ 0.7 sens'].quantile(0.975)
        spec = flucovid_results['spec. @ 0.7 sens'].mean()
        spec_lower = flucovid_results['spec. @ 0.7 sens'].quantile(0.025)
        spec_upper = flucovid_results['spec. @ 0.7 sens'].quantile(0.975)

        print(f'{race} - ili sensitivity is {sens} ({sens_lower}-{sens_upper}, 95\% CI) and specificity is {spec} ({spec_lower}-{spec_upper}, 95\% CI')


#         flucovid_results = read_fluvey(dfs, agg_days=7)
#         #calculate mean across all results
#         sens = flucovid_results['Combined sens. @ 0.7 sens'].mean()
#         sens_lower = flucovid_results['Combined sens. @ 0.7 sens'].quantile(0.025)
#         sens_upper = flucovid_results['Combined sens. @ 0.7 sens'].quantile(0.975)
#         spec = flucovid_results['Combined spec. @ 0.7 sens'].mean()
#         spec_lower = flucovid_results['Combined spec. @ 0.7 sens'].quantile(0.025)
#         spec_upper = flucovid_results['Combined spec. @ 0.7 sens'].quantile(0.975)

#         print(f'{race} - flucovid cumulative sensitivity is {sens} ({sens_lower}-{sens_upper}, 95\% CI) and specificity is {spec} ({spec_lower}-{spec_upper}, 95\% CI')
        
    
        
    for gender in ['gender_female','gender_male','gender_other',]:
        participants = dfs['baseline'].loc[dfs['baseline'][gender]==1].index.get_level_values('participant_id')
        use_participants = set(participants).intersection(set(dfs['survey'].index.get_level_values('participant_id')))
        print(gender, len(use_participants))
        
        if len(use_participants) < 1000:
            continue
        
        flucovid_results = read_fluvey(dfs, participants=use_participants, n_cpu=5)
        #calculate mean across all results
        sens = flucovid_results['sens. @ 0.7 sens'].mean()
        sens_lower = flucovid_results['sens. @ 0.7 sens'].quantile(0.025)
        sens_upper = flucovid_results['sens. @ 0.7 sens'].quantile(0.975)
        spec = flucovid_results['spec. @ 0.7 sens'].mean()
        spec_lower = flucovid_results['spec. @ 0.7 sens'].quantile(0.025)
        spec_upper = flucovid_results['spec. @ 0.7 sens'].quantile(0.975)

        print(f'{gender} - ili sensitivity is {sens} ({sens_lower}-{sens_upper}, 95\% CI) and specificity is {spec} ({spec_lower}-{spec_upper}, 95\% CI')


#         flucovid_results = read_fluvey(dfs, agg_days=7)
#         #calculate mean across all results
#         sens = flucovid_results['Combined sens. @ 0.7 sens'].mean()
#         sens_lower = flucovid_results['Combined sens. @ 0.7 sens'].quantile(0.025)
#         sens_upper = flucovid_results['Combined sens. @ 0.7 sens'].quantile(0.975)
#         spec = flucovid_results['Combined spec. @ 0.7 sens'].mean()
#         spec_lower = flucovid_results['Combined spec. @ 0.7 sens'].quantile(0.025)
#         spec_upper = flucovid_results['Combined spec. @ 0.7 sens'].quantile(0.975)

#         print(f'{gender} - flucovid cumulative sensitivity is {sens} ({sens_lower}-{sens_upper}, 95\% CI) and specificity is {spec} ({spec_lower}-{spec_upper}, 95\% CI')
    
    

        
        
        
    
    

if __name__=="__main__":
    main()
