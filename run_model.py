"""
"""

import torch, torch.optim, torch.nn as nn, torch.nn.functional as F, torch.nn.init as init
from torch.utils.data import DataLoader, Dataset, RandomSampler, SubsetRandomSampler

from radam import RAdam
try:
    from apex import amp
except ImportError:
    amp=None
from GRUD import GRUD
from models import *
from constants import *
import os
from tqdm import tqdm
import pickle
import json
import re
import sys
import glob
import gc

import pandas as pd
import numpy as np
import random
import sklearn.metrics
from sklearn.metrics import roc_auc_score
from copy import deepcopy
from itertools import repeat
from multiprocessing import Pool

import time
import traceback
from collections import Counter
from datetime import date, timedelta

from prospective_set_up import get_retro

from torch.utils.data.dataloader import default_collate


# This fixed the andata 0 received issue, pytorch constructed this strategy to deal with filesystems with a limit on the number of open files, see https://pytorch.org/docs/stable/multiprocessing.html for downsides of using this strategy over the default, etc.
torch.multiprocessing.set_sharing_strategy('file_system')


def fast_lags(df, num_days, diff=False, ldiff=False):
    holder = []
    cidx = df[('idx','cidx')]
    df = df.drop(columns=[('idx','cidx')])
    for ll in np.arange(num_days)+1:
        tmp_ll = df.shift(ll)
        tmp_ll.columns = pd.MultiIndex.from_frame(tmp_ll.columns.to_frame().assign(df_type=str(ll)+'days_ago'))
        tmp_ll.insert(0,('idx','cidx'),cidx)
        tmp_ll.loc[tmp_ll[('idx','cidx')] <= ll] = np.NaN
        tmp_ll.drop(columns=[('idx','cidx')],inplace=True)
        if diff:
            tmp_ll = df.values - tmp_ll
        if ldiff:
            tmp_ll = np.log((df.values+1) / (tmp_ll+1))
        holder.append(tmp_ll)
    tmp_ll = pd.concat(holder,1)
    tmp_ll = pd.concat([df, tmp_ll],1)
    return tmp_ll


def load_data(data_dir, regular=True, load_baseline=True, load_activity=True, load_survey=True, only_healthy=False, fname=None, verbose=False):
    """
    """
    if fname is None:
        f_pattern = '_daily_data_' + 'regular'*regular +'irregular'*(not(regular)) + '.hdf' 
        fnames = [f for f in os.listdir(data_dir) if f.endswith(f_pattern)]
        assert len(fnames) <= 1, "More than one file matches pattern, unsure which to load"
        assert len(fnames) > 0, "No file matched the pattern"
        fname = fnames[0]

    fname=os.path.join(data_dir, fname)
    
    assert os.path.exists(fname), fname
    
    if not(os.path.exists(fname)):
        fname='all_daily_data.hdf'
        fname=os.path.join(data_dir, fname)

    assert os.path.exists(fname), fname
    
    dfs={}
    if load_baseline:
        if verbose: print("loading baseline")
        dfs['baseline']=pd.read_hdf(fname, 'baseline')
    if load_activity:
        if verbose: print("loading activity")
        dfs['activity']=pd.read_hdf(fname, 'activity')
    if load_survey:
        if verbose: print("loading survey")
        dfs['survey']=pd.read_hdf(fname, 'survey')

    return dfs

def apply_limits(dfs, limit_df):
    """
    Apply date limits in limit_df to all data in dfs
    """
    limit_df = limit_df.set_index('participant_id')
    out_dfs={}
    for k, v in dfs.items():
        if 'date' in v.index.names:
            # apply the limits per participant
            # multiindex slicing with date
            
            out_dict={k:v.groupby('participant_id', group_keys=False).apply(lambda x: x.loc[(x.index.get_level_values('date')>=limit_df.loc[x.name, 'study_start_date'])
                                                    & (x.index.get_level_values('date')<=limit_df.loc[x.name, 'study_end_date'])])}
            out_dfs.update(out_dict)
        else:
            out_dfs.update({k:v})
    return out_dfs


def only_healthy(dfs, args):
    """
    Return a new df with only the observations before the first positive
    occurence of target.
    Assumes data is in the format as read in by the load_data function above.
    """
    result_df ={}
    main_target = args.target[0]
    tmp = dfs['survey'].reset_index(['date'])
    tmp2 = pd.DataFrame.join(tmp[['date']], tmp[tmp[main_target] == 1].groupby('participant_id')[['date']].transform(min).reset_index().drop_duplicates().set_index('participant_id'), rsuffix='_min')
    result_df['survey'] = dfs['survey'].reset_index(['date'])[(tmp2['date'] < tmp2['date_min'])].set_index(['date'], append=True)
    result_df['activity'] = dfs['activity'].reset_index(['date'])[(tmp2['date'] < tmp2['date_min'])].set_index(['date'], append=True)
    result_df['baseline'] = dfs['baseline']
    return result_df


def write_results(args, scores, dataloader, istestset=True, participant_order=None):
    csv_suffix = '_' + 'val'*(not(istestset)) + 'test'*istestset + 'set_results'+'.csv'
    wpath = os.path.join(args.output_dir, args.target[0] + csv_suffix)
    
    if isinstance(scores, pd.DataFrame):
        print('Writing DataFrame')
        scores = scores.reset_index()
        scores.to_csv(wpath,index=False)
        return
    
    if not isinstance(scores, dict):
        #If scores is not a dict then assume there is a single target
        tmp = {} 
        # not all test participants are sampled.
        if participant_order is None:
            p = dataloader.dataset.participants 
        else:
            p = participant_order
        tmp[args.target[0]] = dataloader.dataset.outcomes[args.target[0]].sort_index().reindex(p, level='participant_id')
        
        scores=np.concatenate([l.ravel() for l in scores], axis=0)
        score_key = args.target[0]+'_score'
        tmp[score_key] = pd.Series(scores)
        
        # get the thresholds (or apply them
        if istestset:
            #load and check thresholds
            if os.path.exists( os.path.join( args.output_dir, 'thresholds_activity.json')):
                with open(os.path.join( args.output_dir, 'thresholds_activity.json')) as f:
                    threshold = json.load(f)
                # apply this to the data and get the scores.
                for k, v in threshold.items():
                    tmp[args.target[0]+'_pred_'+k] = np.asarray(scores >= float(v) ).astype(np.int32)
                    
        else:
            # if this is validation set, and valid start/valid end date are defined, we must subselect those dates.
            if args.validation_start_date  is not None:
                keep_inds = pd.to_datetime(tmp[args.target[0]].index.get_level_values('date'))>=args.validation_start_date
                tmp[args.target[0]] = tmp[args.target[0]].loc[keep_inds , :]
                # If keep_inds is the same length as the scores then the scores were not filtered prior to passing them to this function so remove the extra dates now.
                if tmp[score_key].shape[0] == keep_inds.shape[0]:
                    tmp[score_key] = tmp[score_key].loc[keep_inds]
            if args.validation_end_date is not None:
                keep_inds = pd.to_datetime(tmp[args.target[0]].index.get_level_values('date'))<=args.validation_end_date
                tmp[args.target[0]] = tmp[args.target[0]].loc[keep_inds, :]
                if tmp[score_key].shape[0] == keep_inds.shape[0]:
                    tmp[score_key] = tmp[score_key].loc[keep_inds]
            
            threshold={}
            fpr, tpr, thresholds = sklearn.metrics.roc_curve( tmp[args.target[0]], tmp[score_key])

            # 98% specificity
            specificity=1-fpr
            for spec in [0.98, 0.95]:
                target_specificity = min(specificity[specificity>=spec])
                # index of target fpr
                index = list(specificity).index(target_specificity)
                threshold['{}_spec'.format(int(spec*100))] = thresholds[index]

            # 98% sensitivity
            target_tpr = min(tpr[tpr>=0.98])
            # index of target fpr
            index = list(tpr).index(target_tpr)

            threshold['98_sens'] = thresholds[index]
            
            # 70% sensitivity
            target_tpr = min(tpr[tpr>=0.7])
            # index of target fpr
            index = list(tpr).index(target_tpr)

            threshold['70_sens'] = thresholds[index]
            
            # 50% sensitivity
            target_tpr = min(tpr[tpr>=0.5])
            # index of target fpr
            index = list(tpr).index(target_tpr)

            threshold['50_sens'] = thresholds[index]

            # threshold to json
            with open(os.path.join(args.output_dir, 'thresholds_activity.json'), 'w') as f:
                f.write(json.dumps({k:str(v) for k,v in threshold.items()}))
                
            for k, v in threshold.items():
                tmp[args.target[0]+'_pred_'+k] = np.asarray(tmp[score_key] >= float(v) ).astype(np.int32)
            
        tmp[score_key] = tmp[score_key].values
        #TODO: 'ili_pred_98_spec has not been the same length as scores after filtering by time for the validation set
        pd.DataFrame(tmp).to_csv(wpath)
    else:
        tmp = dataloader.dataset.outcomes[list(args.target)].copy(deep=True).sort_index()
        for k in scores.keys():
            tmp.insert(len(tmp.columns), k + '_score', scores[k])
            
            # get/load thresholds
            if istestset:
                #load and check thresholds
                if os.path.exists( os.path.join( args.output_dir, k+'thresholds_activity.json')):
                    with open(os.path.join( args.output_dir, k+'thresholds_activity.json')) as f:
                        threshold = json.load(f)
                    # apply this to the data and get the scores.
                    for k2, v in threshold.items():
                        tmp[k+'_pred_'+k2] = np.asarray(scores >= float(v) ).astype(np.int32)

            else:
                # if this is validation set, and valid start/valid end date are defined, we must subselect those dates.
                if args.validation_start_date is not None:
                    tmp = tmp.loc[tmp.index.get_level_values('date')>=args.validation_start_date, :]
                if args.validation_end_date is not None:
                    tmp = tmp.loc[tmp.index.get_level_values('date')<=args.validation_end_date, :]
                threshold={}
                fpr, tpr, thresholds = sklearn.metrics.roc_curve( tmp[k].values, tmp[k+'_score'].values)

                # specificity thresholds
                specificity=1-fpr
                for spec in [0.98, 0.95]:
                    target_specificity = min(specificity[specificity>=spec])
                    # index of target fpr
                    index = list(specificity).index(target_specificity)
                    threshold['{}_spec'.format(spec*100)] = thresholds[index]

                # 98% sensitivity
                target_tpr = min(tpr[tpr>=0.98])
                # index of target fpr
                index = list(tpr).index(target_tpr)

                threshold['98_sens'] = thresholds[index]

                # 70% sensitivity
                target_tpr = min(tpr[tpr>=0.7])
                # index of target fpr
                index = list(tpr).index(target_tpr)

                threshold['70_sens'] = thresholds[index]
                
                # 50% sensitivity
                target_tpr = min(tpr[tpr>=0.5])
                # index of target fpr
                index = list(tpr).index(target_tpr)

                threshold['50_sens'] = thresholds[index]

                # threshold to json
                with open(os.path.join(args.output_dir, k+'thresholds_activity.json'), 'w') as f:
                    f.write(json.dumps({k2:str(v) for k2,v in threshold.items()}))

                for k2, v in threshold.items():
                    tmp[k+'_pred_'+k2] = np.asarray(scores >= float(v) ).astype(np.int32)
        tmp.to_csv(wpath)
        
        
def resample_fairly(participants, dfs, sample_col='ili'):
    """
    increase the occurence of participants based on their ILI incidenc, week, and region.
    Inputs:
        participants (list): list of the participant ids to be involved.
        dfs (dict): all of the dataframes from load_data
    Returns:
        (list): list of the participants upsampled.
    """
    idx=pd.IndexSlice

    # we want to balance interaction between week and region
    
    region_dict = {'Northeast': ['CT', 'ME', 'MA', 'NH', 'RI', 'VT', 'NJ', 'NY', 'PA'],
    'Midwest': ['IN', 'IL', 'MI', 'OH', 'WI', 'IA', 'KS', 'MN', 'MO', 'NE', 'ND', 'SD'],
    'South': ['DE', 'DC', 'FL', 'GA', 'MD', 'NC', 'SC', 'VA', 'WV', 'AL', 'KY', 'MS', 'TN', 'AR', 'LA', 'OK', 'TX', 'PR'],
    'West': ['AZ', 'MT', 'CO', 'ID', 'NM',  'UT', 'NV', 'WY', 'AK', 'CA', 'HI', 'OR', 'WA']}
    
    region_dict2 = {item: key for key, value in region_dict.items() for item in value}

    baseline_df = dfs['baseline'].copy()
    print('converting states')
    baseline_df['region'] =baseline_df['state'].replace(region_dict2)
    baseline_df
    
    df=dfs['survey'].copy()
#     df['weekofyear']=df.index.get_level_values('date').weekofyear
    if 'weekofyear' not in df.columns:
        df['weekofyear']=df.index.get_level_values('date').isocalendar().week
    
    # get the number of occurences for each person in the training data
    participant_df = df.groupby('participant_id').apply(lambda x: x[sample_col].gt(x[sample_col].shift()).cumsum().max() + x[x.index.get_level_values('date') == x.index.get_level_values('date').min()][sample_col]) # todo, just do this once, them multiply by counts in list of participants
    
    print('grouping participant_id and weekofyear')
    tmp_df = df[['weekofyear', sample_col]].groupby(['participant_id', 'weekofyear']).max()
    tmp_df = tmp_df.join(baseline_df['region'], on='participant_id', how='left')
    
    
    ineligible_upsamples = tmp_df.loc[~tmp_df['region'].isin(region_dict.keys())].index.get_level_values('participant_id').tolist()
    ineligible_downsamples = [] 
    max_week_df = tmp_df.groupby(['weekofyear', 'region']).sum()
    
    
    region_weeks = zip(max_week_df.index.get_level_values('region'), max_week_df.index.get_level_values('weekofyear'), max_week_df[sample_col])
    region_weeks=sorted(region_weeks, key=lambda x: x[2])
    
    print("maximum week ", region_weeks[-1])
    
    # get the participants in the max week so that we don't upsample them.
    max_week_participants = list(set(tmp_df.loc[(tmp_df.index.get_level_values('weekofyear')==region_weeks[-1][1])\
                                       &(tmp_df['region']==region_weeks[-1][0])\
                                       &(tmp_df[sample_col]==1), :].index.get_level_values('participant_id')))
    
    
    region_weeks = [item for item in region_weeks if item[2]>10] # skip the weeks where there are few cases/surveys
    
    while len(region_weeks)>1:
        print(len(region_weeks))

        time_old = time.time()  
        time_older=time.time()
        
        amplifier = pd.Series(Counter(participants)) # get the number of unique instances for each participant
        amplifier.index.name='participant_id'
        amplifier.name='multiplier'
        
        tmp_df2 = tmp_df.join(amplifier, how='left', on='participant_id')
        tmp_df2.loc[:, 'ili_multiplier'] = tmp_df2[sample_col]*tmp_df2['multiplier']
        
        # sum all the participants with ILI in a given week+region
        max_week_df = tmp_df2.groupby(['weekofyear', 'region']).sum()
        # remove the data we've already covered
        max_week_df = max_week_df.loc[[(item[1], item[0]) for item in region_weeks]]
        # make it into a tuple
        region_weeks = zip(max_week_df.index.get_level_values('region'), max_week_df.index.get_level_values('weekofyear'), max_week_df['ili_multiplier'])
        region_weeks=sorted(region_weeks, key=lambda x: x[2])
        
        # now we are always popping the minimum region/week combo
        region_week=region_weeks.pop(-2)
        print(region_week)
        if region_week[2]==0:
            # there are no ili incidence this week
            continue        
        
        participant_df_sample = participant_df.loc[participants].groupby('participant_id').sum() # this reflects their prevalence in oversampleing.

        # particpants not in max_region_week or who have already been upsamples get upsampled
        eligible_samples = tmp_df2.loc[(tmp_df2.index.get_level_values('weekofyear')==region_week[1])\
                                      &(tmp_df['region']==region_week[0])\
                                      &(~tmp_df2.index.get_level_values('participant_id').isin( max_week_participants+ineligible_upsamples))\
                                      &(tmp_df2['ili_multiplier']==1), :]
                
        if len(eligible_samples)==0:
            print(region_week)
            continue
            raise Exception(f'ran out of eligible samples for week {region_week[1]} and region {region_week[0]}')
            
        # sample participants based on the inverse of their frequency in other weeks
        weights=(1/participant_df_sample.loc[participant_df_sample.index.isin(eligible_samples.index.get_level_values('participant_id'))].values)/\
                        sum( 1/participant_df_sample.loc[ participant_df_sample.index.isin(eligible_samples.index.get_level_values('participant_id'))].values)
        
        # finally add the upsampled participants
        participants += eligible_samples.sample(n=int(region_weeks[-1][2]-region_week[2]), weights=weights, replace=True).index.get_level_values('participant_id').tolist()     
                
        # now remove samples from elibile samples
        ineligible_upsamples += eligible_samples.index.get_level_values('participant_id').tolist()
        
        
        
        
    amplifier = pd.Series(Counter(participants)) # get the number of unique instances for each participant
    amplifier.index.name='participant_id'
    amplifier.name='multiplier'

    tmp_df2 = tmp_df.join(amplifier, how='left', on='participant_id')
    tmp_df2.loc[:, 'ili_multiplier'] = tmp_df2[sample_col]*tmp_df2['multiplier']
        
        
    # add some checks
    max_week_df = tmp_df2.loc[idx[participants, :], :].groupby(['weekofyear', 'region']).sum()
    region_weeks = zip(max_week_df.index.get_level_values('region'), max_week_df.index.get_level_values('weekofyear'), max_week_df['ili_multiplier'])
    region_weeks=sorted(region_weeks)
    print(region_weeks)
    # show the mean by region
    for region in sorted(list(set(max_week_df.index.get_level_values('region')))):
        print(region, 'mean: ', np.mean([item[2] for item in region_weeks if item[0]==region]), 'std: ', np.std([item[2] for item in region_weeks if item[0]==region]))
    # show the mean by week
    for weekofyear in sorted(list(set(max_week_df.index.get_level_values('weekofyear')))):
        print(weekofyear, 'mean: ', np.mean([item[2] for item in region_weeks if item[1]==weekofyear]), 'std: ', np.std([item[2] for item in region_weeks if item[1]==weekofyear]))
    
    
    
    return participants
    
    

class ILIDataset(Dataset):
    def __init__(self, dfs, args, full_sequence=False, feat_subset=False, participants=None):
        """
        feat_subset: boolean, use only the subset of features used by Raghu 
        which improved performance in the xgboost and ridge regression models 
        in the 'measurement' dataframe.
        """
        self.max_seq_len=args.max_seq_len
        self.numerics = dfs['activity']
        self.statics = dfs['baseline']
        self.outcomes = dfs['survey']
        self.feat_subset = feat_subset
         
        self.subset = ['heart_rate_bpm', 'walk_steps', 'sleep_seconds',
                       'steps__rolling_6_sum__max', 'steps__count', 
                       'steps__sum', 'steps__mvpa__sum', 'steps__dec_time__max',
                       'heart_rate__perc_5th', 'heart_rate__perc_50th', 
                       'heart_rate__perc_95th', 'heart_rate__mean',
                       'heart_rate__stddev', 'active_fitbit__sum',
                       'active_fitbit__awake__sum', 'sleep__asleep__sum',
                       'sleep__main_efficiency', 'sleep__nap_count',
                       'sleep__really_awake__mean', 
                       'sleep__really_awake_regions__countDistinct']
        
        if args.no_imputation:
            self.activity_sub_df = 'measurement_noimp'
        else:
            self.activity_sub_df = 'measurement'
        print(self.activity_sub_df, '*'*30)

        # Using np.unique returning the indices to preserve the order the participants are in the original 
        # self.participants=list(np.unique(dfs['baseline'].index.get_level_values('participant_id')))
        #self.participants=np.unique(dfs['baseline'].index.get_level_values('participant_id')).tolist()
              
        if participants is None:
            # only resample for training data
            self.participants = list(np.unique(dfs['survey'].index.get_level_values('participant_id')))
        else:
            self.participants = participants
        
        self.full_sequence=full_sequence
        
        if args.weekofyear:
            ACTIVITY_COLS.append('weekofyear')
        self.ACTIVITY_COLS = ACTIVITY_COLS

        self.cat_mask_measure_time=False
        if 'modeltype' in vars(args).keys():
            if args.modeltype=='gru_simple':
    #             input('running with concatenation')
                self.cat_mask_measure_time=True
        
    def __len__(self):
        return len(self.participants)
    
    def __getitem__(self, item):
        """
        """
        participant = self.participants[item]
#         print(type(participant))
        idx = pd.IndexSlice
        
        # get the data for these participants
        
#         print('ind:', self.numerics.index.names)
#         print('cols:', self.numerics.columns.names)
        time_df = self.numerics.loc[idx[participant, :], idx['time', :]].values
        if self.feat_subset:
            measurement_df = self.numerics.loc[idx[participant, :], 
                                    idx[self.activity_sub_df, self.subset]].values
            measurement_z_df = self.numerics.loc[idx[participant, :], 
                    idx['measurement_z', self.subset]].values
        else:
            measurement_df = self.numerics.loc[idx[participant, :], 
                    idx[self.activity_sub_df, self.ACTIVITY_COLS]].values
            measurement_z_df = self.numerics.loc[idx[participant, :], 
                    idx['measurement_z', self.ACTIVITY_COLS]].values
        mask_df =self.numerics.loc[idx[participant, :], idx['mask', self.ACTIVITY_COLS]].values.astype(np.int32)
                
        

        # get the outcomes
        ili_outcomes=self.outcomes.loc[idx[participant, :], 'ili'].apply(int).values
        ili_24_outcomes=self.outcomes.loc[idx[participant, :], 'ili_24'].apply(int).values
        ili_48_outcomes=self.outcomes.loc[idx[participant, :], 'ili_48'].apply(int).values
        
        covid_outcomes=self.outcomes.loc[idx[participant, :], 'covid'].apply(int).values
        covid_24_outcomes=self.outcomes.loc[idx[participant, :], 'covid_24'].apply(int).values
        covid_48_outcomes=self.outcomes.loc[idx[participant, :], 'covid_48'].apply(int).values
        fever_outcomes=self.outcomes.loc[idx[participant, :], 'symptoms__fever__fever'].apply(int).values
        if 'loss_mask' in self.outcomes.columns:
            loss_mask=self.outcomes.loc[idx[participant, :], 'loss_mask'].apply(int).values
        else:
            loss_mask=None

        return_dict = {'measurement_z':measurement_z_df, 'measurement':measurement_df, 'mask':mask_df,  'time':time_df, 'ili':ili_outcomes, 'ili_24': ili_24_outcomes, 'ili_48':ili_48_outcomes, 'covid':covid_outcomes, 'covid_24':covid_24_outcomes, 'covid_48': covid_48_outcomes, 'symptoms__fever__fever':fever_outcomes, 'obs_mask':np.ones(len(measurement_df))}
        
        if loss_mask is not None:
            return_dict.update({'loss_mask':loss_mask})
        
        assert sum(return_dict['obs_mask'])>0
        
        if self.full_sequence:
            
            return_dict = {k:torch.tensor(v).float() for k,v in return_dict.items()}
            
            if self.cat_mask_measure_time:
                return_dict.update({'measurement':
                               torch.cat((return_dict['measurement'], return_dict['mask'], return_dict['time']), dim=-1)})
            return return_dict, participant
        
        
        # pad them to the maximum sequence length
        if len(time_df)>self.max_seq_len:
            # make sure 50% of the time there is a 1 in the data...
            # find minimum index of 1 in labels
            min_index= list(return_dict['ili']).index(1) if sum(return_dict['ili'])>0 else 0
            random_val= random.random() if sum(return_dict['ili'])>0 else 1 # this is mostly not the case, but to be sure we can remove patients without ILI in the start.
            
            if random_val<0.5:
                # includes a 1
                assert max(0, min_index-self.max_seq_len-1)<(len(time_df)-self.max_seq_len), (max(0, min_index-self.max_seq_len-1),(len(time_df)-self.max_seq_len))
                random_start = random.randint(max(0, min_index-self.max_seq_len-1), len(time_df)-self.max_seq_len)
                # sample and crop
                max_time = min(random_start+self.max_seq_len, len(return_dict['ili']))
                return_dict={k:v[random_start:max_time] for k,v in return_dict.items()}
                
            else:
                # does not include a 1
                random_start = random.randint(0, min_index-1) if (min_index-1 )>0 else 0
                # sample and crop
                max_time = min(random_start+self.max_seq_len, len(return_dict['ili']))
                return_dict={k:v[random_start:max_time] for k,v in return_dict.items()}
            
        if len(return_dict['measurement'])<self.max_seq_len:
            # pad to length         
            for k, v in return_dict.items():
#                 print(k)
                shape=list(v.shape)
                shape[0]=self.max_seq_len
#                 print(v)
                obj = np.zeros(shape)
                obj[:len(v)]=v
#                 print(v)
                return_dict[k]=obj
    
#         for k,v in return_dict.items():
#             print(k, v.dtype)
            
                
        return_dict = {k:torch.tensor(v).float() for k,v in return_dict.items()}
#         for item in return_dict.values():
            
#             assert len(item)==self.max_seq_len, print(len(item), self.max_seq_len)

        if self.cat_mask_measure_time:
            return_dict.update({'measurement':
                               torch.cat((return_dict['measurement'], return_dict['mask'], return_dict['time']), dim=-1)})
                
        
        return return_dict, participant

# Solution on Pytorch forum from justusschock for a custom collate function
# which will return an id with each batch.
def id_collate(batch):
    new_batch = []
    ids = []
    for _batch in batch:
        new_batch.append(_batch[:-1])
        ids.append(_batch[-1])
    return default_collate(new_batch), ids


def merge_fn(batch):
    """
    Train/Val DataLoader collate_fn (to merge the different datasets into a batch with pack_sequence)
    """
    batch = sorted(batch, key=lambda x: -len(x[0])) #sort by length of sequence
    data, target = zip(*batch)
    
    batch = sorted(batch, key=lambda x: -len(x[0]['measurement']))
    ids = [b[-1] for b in batch]
    new_batch = {k: [item[0][k] for item in batch] for k in batch[0][0].keys()}
    new_batch = {k : nn.utils.rnn.pack_sequence(v) for k, v in new_batch.items()}
    
    return new_batch, ids


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

            
def apply_standard_scaler(dfs, args, scaler=None):
    """
    apply sklearn standard scaler.
    Takes args to know what sub dataframe to use.
    """
    if args.no_imputation:
        sub_df = 'measurement_noimp'
    else:
        sub_df = 'measurement'

    idx=pd.IndexSlice
    if scaler is not None:
        assert isinstance(scaler, dict), print('Expected a dict for arguement scaler')
    else:
        print(set(sorted(dfs['activity'].loc[:, idx[sub_df, :]].columns.tolist()))-set(dfs['activity'].loc[:, idx[sub_df, :]].sample(5000).mean().index.tolist()))
        scaler={'mean': dfs['activity'].loc[:, idx[sub_df, :]].sample(5000).mean().values,
                'std': dfs['activity'].loc[:, idx[sub_df, :]].sample(5000).std().values,
               }
    # apply to measurement column in X and X_ffilled  
    
    assert scaler['mean'][np.newaxis, :].shape[1:] == scaler['std'][np.newaxis, :].shape[1:], print(scaler['mean'][np.newaxis, :].shape, scaler['std'][np.newaxis, :].shape[1:])
    assert dfs['activity'].loc[:, idx[sub_df, :]].values.shape[1:] == scaler['std'][np.newaxis, :].shape[1:], print(dfs['activity'].loc[:, idx[sub_df, :]].values.shape, scaler['std'][np.newaxis, :].shape) 
    dfs['activity'].loc[:, idx[sub_df, :]] = (dfs['activity'].loc[:, idx[sub_df, :]].values - scaler['mean'][np.newaxis, :])/scaler['std'][np.newaxis, :]
                                                                        
    return dfs, scaler


def cnn_lklhd(mus, Sigma, x):
    """
    Helper function for cnn model, returns the loglikelihood for each observation using pytorchs logprob for multivariate gaussian distribution.
    """
    loglklhds = []
    for i in range(x.shape[0]):
        dist = torch.distributions.MultivariateNormal(mus[i, :], scale_tril=Sigma[i, :, :])
        loglklhds.append(dist.log_prob(x[i,:]).unsqueeze(dim=0))
        
    return torch.cat(loglklhds, axis=0)
                                                                    
def old_cnn_lklhd(mus, logstds, x, k):
    """
    Helper function for cnn model, returns the loglikelihood for a observation given the 
    parameters returned by the 1dcnn model.
    TODO: Should this function be put into the cnn model class, or it's own file?
    """
    loglklhds = []
    stds = torch.exp(logstds)
    for i in range(x.shape[0]):
        Sigma = torch.diag(stds[i,:]**2)
        detSigma = torch.prod(stds[i,:]**2) + 1e-20
        dif = x[i:(i+1),:] - mus[i:(i+1), :]
        difT = torch.transpose(dif, 0, 1)
        invSigma = torch.inverse(Sigma)
        
        # Tested the pytorch implementation I'm getting the same loglikelihoods with
        # my calculation.
        #dist = torch.distributions.MultivariateNormal(mus, Sigma)
        #torch_lklhd = dist.log_prob(x)
        print('Det(Sigma): ' + str(detSigma))
        min_var = torch.min(stds[i,:]**2).detach().numpy()
        print('Min_var: ' + str(min_var))
        print('which_is_min: ' + str(np.where((stds[i,:]**2).detach().numpy() == min_var)))
        assert detSigma != 0, 'Singular covariance matrix'
        left_term = torch.log(detSigma)
        #Note for mid_term the transpose is backwards, because of
        # the shape of x being (batch, features).
        mid_term = torch.mm(torch.mm(dif,invSigma), difT)
        right_term = k*torch.log(torch.tensor(2*math.pi))
        loglklhds.append(-0.5*(left_term + mid_term + right_term))
    return torch.cat(loglklhds, axis=0).squeeze(dim = 1)
             

def add_shifts(inp, days_to_shift):
    """ Using `days_ago` arguement to set number of past days to add to features"""
    pid, df = inp
    df = df.droplevel(0).sort_index()  # Assumes it is of `datetype` type #pd.DatetimeIndex(df['date']))
    keep_cols = df.columns

    for day_shift in [(q, 'days_ago') for q in range(1, days_to_shift + 1)]:
        tmp = df.loc[:, keep_cols].shift(day_shift[0])
        tmp.columns = pd.MultiIndex.from_product([[str(day_shift[0])+day_shift[1]],
                                                  keep_cols.get_level_values('value').to_list()])
#         tmp.columns = tmp.columns.set_levels([str(day_shift[0]) + day_shift[1]], level=0)
        df = df.join(tmp, how='left')
    df.index = pd.MultiIndex.from_product([[pid], df.index])
    return df
                                   

def get_close_dates(df, time, time_col, time_delta, group_col=None, mask=None):
    """
    Takes a dataframe, a date, a time delta, Series of time deltas or dataframe
    with a column of time deltas, and a time column name, then it filters all 
    rows to only those close enough to the date by the time_delta in the 
    time_col of the df. time_delta can also be container of two elements 
    indicating a previous different acceptable length for before and after 
    time. Optionally it can take a group_col which will index a different time 
    per member of the group, the keys for the time object must be the group 
    names in the group_col. Also it can take a mask which will then drop 
    arbitrary dates which are close enough to time as well, mask times are 
    relative to time and must all fall within the time_delta bound. It's 
    assumed there is a time in the df except in the case where time has a 
    group_col, then if the group id does not have an entry in the time 
    dataframe it will return an empty dataframe with the same index and column 
    names as df.

    df: pandas dataframe
    time: pandas datetime or pandas dataframe
    time_col: str, column name of df which is datetime type
    time_delta: pandas timedelta or tuple of pandas timedelta
    group_col: str, column name for the time dataframe to extract a datetime per group
    mask: list of ints
    """
    # First deal with if the group_col is a column or an index by reseting the index.
    index_cols = list(df.index.names)
    if group_col in index_cols:
        index_cols.remove(group_col)

    df = df.reset_index()

    if not (group_col is None):
        group = np.unique(df[group_col])
        assert len(group) == 1, 'More than one value in df[group_col], might be df is not grouped or group_col is not the column grouped by'
        group = group[0]
        time = time[time[group_col] == group][time_col]
        # Check that the group id is in the time dataframe, if not return an 
        # empty dataframe. This is a case in our experiments where a participant
        # has no date of onset in the truncated dataframe, due to the change in 
        # timeframe, thus returning an empty dataframe is the correct output.
        if len(time) == 0:
            return df.set_index(index_cols).drop(columns=group_col, errors='ignore')[:0]
    
    # Get separate bounds for the time_delta
    if len(time_delta) == 2:
        assert time_delta[0] <= time_delta[1], 'lesser time_delta must be smaller than upper time_delta'
        lower_delta, upper_delta = time_delta
    elif len(time_delta) == 1:
        lower_delta = time_delta[0]
        upper_delta = time_delta[0]

    keep_date = pd.concat([df[time_col] - pd.to_datetime(t) for t in time], axis = 1).apply(lambda x: (x >= lower_delta) & (x <= upper_delta) & (~x.isin(mask)) ).any(axis = 1)

    res_df = df.set_index(index_cols).drop(columns=group_col, errors='ignore')[keep_date.values]

    return res_df


def boundary_helper(participant_id, recovery_dates, ili_diffs):
    """
    A helper function to turn a list comprehension into something more readable.
    Concats the maximum date when the ili_diffs is missing a recovery date, 
    because the event terminates after the dataframes maximum date.
    """
    
    tmp_dict = {'participant_id':participant_id,
                'date':ili_diffs.loc[participant_id].index.get_level_values('date').max()}
    tmp_df = pd.DataFrame(tmp_dict, index=[0])
    cur_recov_df = recovery_dates[recovery_dates['participant_id'] == participant_id]
    
    return pd.concat([cur_recov_df, tmp_df], ignore_index=True)


def onset_recovery_diff(x):
    """
    Only works on a Series, handles missing value occuring at the beginning and end of a pandas diff call, instead of returning NA, it will use the value that was there prior to the diff. ie, Series([1,2,3]) becomes Series([1,1,1]). 
    """
    x.index = x.index.droplevel('participant_id')
    didx = pd.Series(x.index.get_level_values('date'))
    diffed = x.reindex(pd.DatetimeIndex([didx.min() - pd.Timedelta('1d')] + didx.to_list() + [didx.max() + pd.Timedelta('1d')]), fill_value=0).diff()
    
    return diffed


def create_lagged_data(dataloader, args, bound_events, use_features=None, target=['ili']):
    """
    TODO: Not tested with target having length greater than 1...
    Inputs:
        dataloader (torch.utils.dataset.DataLoader): An instance of wrapped ILIDataset.
        forecast_features (list): a list of column headings to restrict features to.
        target: the objective feature.
    Returns:
        X (np.array): ??
        y (np.array): ??
    """
    assert isinstance(target, list), "target must be a list of target column names"
    idx=pd.IndexSlice
    # ILIDataset has attributes self.numerics, self.statics, self.outcomes, self.participants
    measurement_col = 'measurement' + args.zscore*'_z' + args.no_imputation*'_noimp' 
    forecast_features = ['time'] + [str(q) + 'days_ago' for q in range(1, args.days_ago + 1)]
    print(dataloader.dataset.numerics.loc[:, idx[measurement_col, :]].head())
    if use_features is None:
        #use_features =      dataloader.dataset.numerics.columns.get_level_values('value').tolist()
        use_features =dataloader.ACTIVITY_COLS # hard coded for safety
    label_colz = ['ili', 'ili_24', 'ili_48', 'covid', 'covid_24', 'covid_48', 'symptoms__fever__fever', 'flu_covid', 'time_to_onset']
    
    target_col = ('label', target[0])
    # add measurement col?
    forecast_features += [measurement_col]  # add today's measurements as part of feature set
        
    
    ### Load survey dataframe (for ILI labels)
    print('Loading survey data ...')
    survey = dataloader.dataset.outcomes.copy()
    survey = survey.sort_index()
    
    if len(survey.columns.names)==1:
        pass #target_col = target[0]
        #         survey.columns = pd.MultiIndex.from_product([['label'],survey.columns.tolist()])

    if bound_events and not 'time_to_onset' in survey.columns:
        # column orig_ili should be created if bound_events has been set, this is to preserve the original ili events for appropriate dropping of events too close together.
        ili_diffs = survey['orig_ili'].groupby('participant_id').apply(onset_recovery_diff)
        ili_diffs.index = ili_diffs.index.rename(['participant_id', 'date'])

        assert len(target) == 1, 'TODO: not implemented for targets lists of length greater than 1.'
        # Restricting all labels to near an ILI event. 
        t = 'ili'

        # get onset dates for each participant which are far enough apart.
        onset_dates = ili_diffs[ili_diffs == 1].reset_index()[['participant_id', 'date']]
        onset_dates_diff = onset_dates.groupby('participant_id').diff()
        is_first_onset_date = onset_dates_diff['date'].isnull()

        # Get the recovery dates to ensure the next event is far enough away.
        recovery_dates = ili_diffs[ili_diffs == -1].reset_index()[['participant_id', 'date']]
        # This assert is probably extra, I think because of how diffs is defined there couldn't be any additional recovery dates when given a binary sequence, so it is really just double checking we got a binary sequence for ILI.
        assert len([p for p in np.unique(ili_diffs.index.get_level_values('participant_id')) if not recovery_dates[recovery_dates['participant_id'] == p].shape[0] - onset_dates[onset_dates['participant_id'] == p].shape[0] in [-1, 0]]) == 0, 'The number of recovery dates and onset dates differs by more than one for some participants.'
        recovery_dates = pd.concat([boundary_helper(p, recovery_dates, ili_diffs) if recovery_dates[recovery_dates['participant_id'] == p].shape[0] - onset_dates[onset_dates['participant_id'] == p].shape[0] != 0 else recovery_dates[recovery_dates['participant_id'] == p] for p in np.unique(ili_diffs.index.get_level_values('participant_id'))])
        
        # length of events
        event_lens = pd.concat([recovery_dates['participant_id'], recovery_dates['date'] - onset_dates['date'].values], axis=1, ignore_index=True)
        event_lens = event_lens.rename(columns={0:'participant_id', 1:'length'})
        # shift by one period
        event_lens['length'] = event_lens['length'].shift()

        onset_distance = pd.to_timedelta(min(args.bound_labels)*-1 + args.days_ago, 'd') + event_lens['length']
        is_far_from_onset_date = onset_dates_diff['date'] >= onset_distance.reset_index(drop=True)
        onset_dates = onset_dates[is_first_onset_date | is_far_from_onset_date]

        #leave enough previous days for first day wanted to be in the bound to have all previous days worth of data.
        larger_bounds = (args.bound_labels[0] - args.days_ago, args.bound_labels[1])
        # final bound will restrict the dates to just the ones we want to make a prediction on and not keep the days necessary to fill the days ago columns.
        final_bounds = (args.bound_labels[0], args.bound_labels[1])

        final_indices = survey.groupby('participant_id').apply(get_close_dates, onset_dates, 'date', pd.to_timedelta(final_bounds, 'd'), 'participant_id', pd.to_timedelta(args.mask_labels, 'd')).index

    #select down the survey dataframe to the appropriate dates.
        survey = survey.groupby('participant_id').apply(get_close_dates, onset_dates, 'date', pd.to_timedelta(larger_bounds, 'd'), 'participant_id', pd.to_timedelta(args.mask_labels, 'd'))
    elif bound_events:
        # Use the time_to_onset column instead of calculating it. Survey has enough back data for the features to have 7 days worth of data.
        survey = survey[(survey['time_to_onset'] >= min(args.bound_labels) - args.days_ago) & (survey['time_to_onset'] <= max(args.bound_labels)) & (~survey['time_to_onset'].isin(args.mask_labels))]
        # final_indices will restrict to the dates we actually want to make predictions on.
        final_indices = survey[(survey['time_to_onset'] >= min(args.bound_labels)) & (survey['time_to_onset'] <= max(args.bound_labels)) & (~survey['time_to_onset'].isin(args.mask_labels))].index
    survey.columns = pd.MultiIndex.from_product([[target_col[0]], survey.columns]) # add multi-index columns to survey
    activity = dataloader.dataset.numerics.loc[:, idx[measurement_col, use_features]]
    activity = activity.loc[survey.index]
    
    print('survey shape: %s' % (survey.shape,))
    
    ### Create day-shifted features - for a subset of 20 features
    #   print(len(use_features))
    print('Using cores=', args.num_dataloader_workers)
    print('Creating day-shifted dataset ...')
    p = Pool(args.num_dataloader_workers)
    out = pd.concat(p.starmap(add_shifts,
                              zip(activity.groupby('participant_id'),
                                  repeat(args.days_ago))))
    out.index.names = ['participant_id', 'date']
    print('Shifted-features dataframe:', out.shape)
    
    ### Merge with day-level ILI labels and adding time features
    out = out.join(survey.loc[:, idx[[target_col[0]], label_colz]], how='left')
#     out['time', 'day_of_week'] = pd.Series(out.index.get_level_values(1)).dt.weekday.values
    if args.weekofyear:
        if ('time','weekofyear') not in out.columns.tolist():
            print(out.columns.tolist())
            # out['time', 'weekofyear'] = pd.Series(out.index.get_level_values(1)).dt.weekofyear.values # potentially remove this
            out['time', 'weekofyear'] = pd.Series(out.index.get_level_values(1)).dt.isocalendar().week.astype(np.int32).values # potentially remove this
    print('out shape: %s' % (out.shape,))
    print('out columns: %s' % out.columns)
    
    #filter out to have the final indices only, if bounding observations around events.
    if bound_events:
        out = out.loc[final_indices]

    X = out.loc[:,idx[forecast_features, :]]
    y = out.loc[:, target_col]
    
    if args.resample:
        participant_count= Counter(dataloader.dataset.participants)
        sample_weight = [participant_count[item] for item in y.index.get_level_values('participant_id')]
    else:
        sample_weight=None
    
    
    return X, y, sample_weight


def train_torch(model, train_dataloader, valid_dataloader, args):
    """
    Train the models that require pytorch dataloader in a serperate function
    """
    
    return

def get_latest_event_onset(dfs):
    """
    input:
        dfs (dict): a dict of pandas dataframes of the data.
    Returns:
        (list): a list of tuples of participant_id, and latest event onset date.
    """
    df=dfs['survey'].copy()
    df['ili_count'] = df['ili'].groupby('participant_id').apply(lambda x:x.rolling(2).apply(lambda y:y[0]<y[-1], raw=True).cumsum())
    df['ili_count'] = df['ili_count'].fillna(0)
    tmp_df = df['ili'].groupby('participant_id').first()
    df = df.join(tmp_df, on=['participant_id'],how='left', rsuffix='_first')
    # if it starts with 1, we also want to coutn it as a case.
    df['ili_count']=df['ili_count']+df['ili_first'] 
    df = df.loc[df.groupby('participant_id')['ili_count'].transform(max) ==df['ili_count']] # get only th maximum ili per participant

    # get the onset dat for the maximum ili per participant
    df = df.reset_index().loc[df.reset_index().groupby('participant_id')['date'].transform(min)==df.reset_index()['date']] 

    # get the index of the onset of the latest event for a partiicpant.
    participant_ili_tuples = df.set_index(['participant_id', 'date']).index.tolist()
    
    return participant_ili_tuples


def split_helper(participants, train_size=0.7, val_size=0.15, test_size=0.15):
    
    assert (train_size + val_size + test_size) == 1, "split sizes don't add up to one"
    random.shuffle(participants)
    train_participants = participants[:int(0.7*len(participants))]
    valid_participants = participants[int(0.7*len(participants)):int(0.85*len(participants))]
    test_participants = participants[int(0.85*len(participants)):]

    return train_participants, valid_participants, test_participants


def get_onset_dates(x, label_col):
    """
    Get the first instance sick day for a participants label series, fixes case
    where onset is the first day of the Series.
    """
    didx = pd.Series(x.index.get_level_values('date'))
            
    diff = x.droplevel(0).reindex(pd.DatetimeIndex([didx.min() - pd.Timedelta('1d')] + didx.to_list() + [didx.max() + pd.Timedelta('1d')]), fill_value=0).diff()
    #drop the added dates to fix end point corner case for diff
    diff = diff[1:-1]

    onset_dates = x.droplevel(0)[(diff == 1).values]

    return onset_dates


def change_labels(dataframe, relative_changes, label_column):
    """
    Assumes the label in the dataloader is binary, and will then make the positive labels relative to the first label in each sequence of the positive class, in our case the date of onset for a disease.
    dataframe : pandas dataframe with a binary label in the dataset.
    relative_changes : list of values for labels relative to the first positive label which will be the new positive labels. ex: if old label column is [0 0 0 1 1 1 0 1] with relative change being (-1, 0, 1), then the output label column will be [0 0 1 1 1 0 1 1].
    """
    onset_dates = dataframe[label_column].groupby('participant_id').apply(get_onset_dates, label_col=label_column)
    # When the difference is greater than 0 the previous row must have been 0 indicating that it is a day of onset.

    new_dataframe = dataframe
    new_dataframe[label_column] = 0
    for change in relative_changes:
        date_offset = pd.to_timedelta(change, unit='d')
        relative_dates = onset_dates.copy()
        relative_dates = relative_dates.reset_index()
        relative_dates['date'] = relative_dates['date'] + date_offset
        relative_dates = relative_dates.set_index(['participant_id', 'date']).index
        relative_dates = relative_dates[relative_dates.isin(new_dataframe.index)]
        new_dataframe.loc[relative_dates] = 1

    return new_dataframe


def restricted_float(s):
    try:
        f = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError("{} not a floating-point literal".format(s))

    if f < 0.0 or f > 1.0:
        raise argparse.ArgumentTypeError("{} not a value between 0 and 1".format(s))
    
    return f


def my_roll(tbl):
    tbl = tbl.droplevel(0).rolling('8d', min_periods=1).sum()
    return tbl


def restrict_missing(df, args):
    """
    Function to calculate the missingness for each day. Then remove 
    observations which have missingness larger than the maximum as 
    dictated in the arguments. Assumes index of df is multiindex with two levels participant_id and date, date must be of type pd.datetime, and participant_id is of type str.

    TODO: Implement a way to read the original activity dataset from a non harcoded path
    """
    keep_cols = ['active_fitbit__awake__sum', 'active_fitbit__sum',
                 'heart_rate__asleep__max', 'heart_rate__asleep__mean',
                 'heart_rate__asleep__stddev', 'heart_rate__awake__max',
                 'heart_rate__awake__mean', 'heart_rate__awake__stddev', 
                 'heart_rate__mean', 'heart_rate__not_moving__max', 
                 'heart_rate__not_moving__mean', 'heart_rate__not_moving__stddev', 
                 'heart_rate__perc_25th', 'heart_rate__perc_50th', 
                 'heart_rate__perc_5th', 'heart_rate__perc_75th', 
                 'heart_rate__perc_95th', 'heart_rate__stddev', 
                 'sleep__asleep__mean', 'sleep__asleep__sum',
                 'sleep__awake__mean', 'sleep__awake__sum',
                 'sleep__awake_regions__countDistinct', 'sleep__really_awake__mean',
                 'sleep__really_awake__sum',
                 'sleep__really_awake_regions__countDistinct',
                 'sleep__sleeping__sum', 'steps__count', 
                 'steps__dec_time__max', 'steps__dec_time__min', 
                 'steps__dec_time_max_rolling_6__first', 
                 'steps__light_activity__sum', 'steps__mvpa__sum',
                 'steps__not_moving__sum', 'steps__rolling_6_sum__max',
                 'steps__sedentary__sum', 'steps__streaks__countDistinct', 'steps__sum']
    
    if not restrict_missing.loaded:
        orig_activity = pd.read_feather('/datasets/evidationdata/wave2/day_level_activity_features.feather')    
        tmp = orig_activity
        tmp['date'] = pd.to_datetime(tmp['date'])
        tmp = tmp.set_index(['participant_id', 'date']).sort_index()[keep_cols].isnull().groupby(level='participant_id').apply(my_roll)
        num_cols = tmp.shape[1]
        tmp = tmp.sum(axis = 1, min_count=num_cols)/(num_cols*8)
        tmp.name = ('Missingness', 'Percent Missing')
        restrict_missing.miss_tbl = tmp
        restrict_missing.loaded = True

    new_df = df.copy()
    new_df = new_df.join(restrict_missing.miss_tbl, how='left')
    
    new_df = new_df[new_df[('Missingness', 'Percent Missing')] <= args.max_miss]

    return new_df
restrict_missing.miss_tbl = None
restrict_missing.loaded = False


def add_loss_mask(df_in, args):
    """
    """
#     from pandarallel import pandarallel
#     pandarallel.initialize(nb_workers=args.num_dataloader_workers, progress_bar=True)
    df = df_in.copy()
    df['mask']=np.nan
#     print(len(df))

    if not('date' in df.index.names):
        df = df.set_index(['participant_id', 'date'])
    original_index = df.index
    
    

    print("before 7 day restriction", len(df))

    # reindex to regularly indexed
    print(df['ili'].dtype)
    print(set(df['ili'].values))
    try:
        min_inds= df.loc[df['ili']==1].reset_index('date').groupby('participant_id').min().set_index(['date'], append=True).index
    except:
        min_inds= df.loc[df['ili']==1].reset_index('date').groupby('participant_id')[['ili', 'date']].min().set_index(['date'], append=True).index
        
    try:
        absolute_min_inds= df.reset_index('date').groupby('participant_id').min().set_index(['date'], append=True).index
    except:
        absolute_min_inds= df.reset_index('date').groupby('participant_id')[['ili', 'date']].min().set_index(['date'], append=True).index
    
    try:
        max_inds= df.loc[df['ili']==1].reset_index('date').groupby('participant_id').max().set_index(['date'], append=True).index
    except:
        max_inds= df.loc[df['ili']==1].reset_index('date').groupby('participant_id')[['ili', 'date']].max().set_index(['date'], append=True).index

    person, min_i = zip(*sorted(min_inds.tolist()))
    min_i = [m - pd.Timedelta(days=7+7+21) for m in min_i]
    
    
    print(len(min_inds), len(absolute_min_inds))
    absolute_min_inds=[item for item in absolute_min_inds.tolist() if item[0] in person]
    #absolute_min_inds = sorted(absolute_min_inds, key=lambda x: person.index(x[0]))    
    person2, min_i2 = zip(*sorted(absolute_min_inds))
    assert all( item[0]==item[1] for item in zip(person, person2))
    min_i2 = [m - pd.Timedelta(days=7+7+21) for m in min_i2]
    min_i = [max(item[0], item[1]) for item in zip(min_i, min_i2)]
    
    person, max_i = zip(*sorted(max_inds.tolist()))
    max_i = [m + pd.Timedelta(days=1) for m in max_i]
    all_inds=zip(person, min_i, max_i)

    result_inds=[]
    for ind in all_inds:
        dates = pd.date_range(start=ind[1], end=ind[2])
        result_inds+=list(zip([ind[0]]* len(dates), list(dates)))
    result_inds = pd.MultiIndex.from_tuples(result_inds, names = ['participant_id', 'date'])        

    df = df.reindex(result_inds)
    
    print("after 7 day restriction", len(df))

    # now set the min ili index to 0 if it is na
    df.loc[min_inds, 'ili']=df.loc[min_inds, 'ili'].fillna(0)
    df['ili']=df['ili'].ffill()

    print("pre mask")
#     display(df.iloc[18:30])

    df['mask'] =df.groupby('participant_id', as_index=False)['ili'].transform('diff')
    
    print('0')

#     df['mask'] = df['ili'].reset_index('date').groupby('participant_id', as_index=False).parallel_apply(lambda x: pd.DataFrame({'date':x.date, 'mask': x['ili']-x['ili'].shift()})).set_index('date', append=True)
    
#     print('1')


    # delete events that are too close by reducing the index of the dataframe

    # 1 first calculate where the data windows overlap
    df['mask']=df['mask'].fillna(0)
    
    df['intermediate'] = df.groupby('participant_id')['mask'].rolling(window=(21+7+3+7+1), min_periods=1).sum().reset_index(0, drop=True)
    
#     df['intermediate'] = df['mask'].reset_index('date').groupby('participant_id').parallel_apply(lambda x: pd.DataFrame({'date':x.date, 'mask': x['mask'].rolling(window=(21+7+3+7+1), min_periods=1).sum()})).set_index('date', append=True)
    
    print('2')
    
    df['date_prime'] = np.nan
    df.loc[df['intermediate']>1, 'data_prime']  = 1
    df['date_prime']  = df.groupby('participant_id')['date_prime'].ffill()
    assert(df['date_prime'].isna().mean()>0)
    df['date_prime']=df['date_prime'] .fillna(0)

    max_inds2 = df.loc[df['date_prime']==1].reset_index('date').groupby('participant_id')['date'].transform(min)#.reset_index(0, drop=True)
    # max_inds2 = max_inds2-pd.Timedelta(days=1)

    max_inds3=df.loc[df['date_prime']==0].reset_index( 'date').groupby('participant_id')['date'].transform(max)#.reset_index(0, drop=True)

    max_inds2 = max_inds2.append(max_inds3).groupby('participant_id').min()

    max_inds2=max_inds2.to_frame().set_index('date', append=True).index


    # 2 calculate the max index of each patient considering overlaps

    
#     print('2.5')

#     # 2 calculate the max index of each patient considering overlaps
#     max_inds= df['intermediate'].reset_index('date').groupby('participant_id').parallel_apply(lambda x: x.loc[(x['intermediate']>1), 'date'].min()-pd.Timedelta('1D') if any(x['intermediate']>1) else x['date'].max() )
#     max_inds.name='date'
#     max_inds=max_inds.to_frame().set_index('date', append=True).index
    
    print('3')

    # 3 reindex the dataframe with the appropriate indices.
    person, min_i = zip(*sorted(min_inds.tolist()))
    person, max_i = zip(*sorted(max_inds.tolist()))
    all_inds=zip(person, min_i, max_i)

    result_inds=[]
    for ind in all_inds:
        dates = pd.date_range(start=ind[1], end=ind[2])
        result_inds+=list(zip([ind[0]]* len(dates), list(dates)))
    result_inds = pd.MultiIndex.from_tuples(result_inds, names = ['participant_id', 'date'])        
    df = df.reindex(result_inds)
    
    print("after max overlap restriction", len(df))





    print('first mask')
#     display(df.iloc[18:30])
    df.loc[df['mask']!=1, 'mask']=np.nan
#     print('set nan')
#     display(df.iloc[18:30])

    expanded_index = df.loc[df['mask']==1].index.tolist()
    participants, dates = zip(*expanded_index)
    # # calculate the -1 day for bfill
    expanded_index_1d_ago = pd.MultiIndex.from_tuples(list(zip(participants, pd.to_datetime(dates).shift(-1, freq='D'))), names=['participant_id', 'date']).intersection(df.index)
    # calculate the -7 days for 1 bfill
    expanded_index_7d_ago = pd.MultiIndex.from_tuples(list(zip(participants, pd.to_datetime(dates).shift(-8, freq='D'))), names=['participant_id', 'date']).intersection(df.index)
    # # calculate the -21-7 days for 0 bfill
    # expanded_index_28d_ago = pd.MultiIndex.from_tuples(list(zip(participants, pd.to_datetime(dates).shift(-28, freq='D'))), names=['participant_id', 'date']).intersection(df.index)

#     display(df.reset_index().groupby('participant_id').ffill(limit=2)['mask'].values)
    df['mask_forward'] = df.reset_index().groupby('participant_id').ffill(limit=2)['mask'].values


    # shift index, create a new column.
    df['mask_backward']=np.nan
    df.loc[expanded_index_7d_ago,'mask_backward']=1

    print('second mask')
#     display(df.iloc[18:30])

    df['mask_backward'] = df.groupby('participant_id')['mask_backward'].bfill(limit=21).values
    # df['mask_backward'] = df.groupby('participant_id')['mask'].bfill(1, limit=21)
#     print('bfill second')
#     display(df.iloc[18:30])


    ##############
    # betweens 0s and 1s
    df['skip_mask_backward']=np.nan
    df.loc[expanded_index_1d_ago,'skip_mask_backward']=1
    df['skip_mask_backward'] = df.groupby('participant_id')['skip_mask_backward'].bfill(limit=6).values
#     display(df.iloc[18:30])


    ##############
    # after 1s
    # find the 1-0 transitions
    df['skip_mask_forward']=np.nan
    df['skip_mask_forward'] = df.reset_index('date').groupby('participant_id').apply(lambda x: pd.DataFrame({'date':x.date, 'mask': x['ili'].shift()-x['ili']})).set_index('date', append=True)
    df.loc[df['skip_mask_forward']!=1, 'skip_mask_forward']=np.nan
    df['skip_mask_forward'] = df.groupby('participant_id')['skip_mask_forward'].ffill(limit=6).values

    df['skip_mask_forward']
    df.loc[(df['mask_forward'].isna())&(df['ili']==1), 'skip_mask_forward']=1



    ################
    # overall mask

    # ((mask_forward XOR mask_backward) AND NOT(skip_mask_backward)) AND NOT(skip_mask_forward))
    df['loss_mask'] = np.logical_and(np.logical_and(np.logical_xor(df['mask_forward'].fillna(0).values,df['mask_backward'].fillna(0).values), df['skip_mask_backward'].isna()), df['skip_mask_forward'].isna())

    # finally take the intersection of the original index, and the new_index

#     print(original_index[:5])
#     print(df.index[:5])

    df = df.reindex(original_index.intersection(df.index)).sort_index()
    
    print("after logic restriction", len(df))

    df=df.drop(['intermediate', 'mask', 'mask_forward', 'mask_backward', 'skip_mask_backward', 'skip_mask_forward'],axis='columns')
    # df=df.drop(['intermediate', 'mask', 'mask_forward', 'mask_backward', 'skip_mask_backward', 'skip_mask_forward'],axis='columns')
    print("double check return", len(df))
    
    
    return df


def multi_class_column_scores(ys, X, dataloader, args):
    """
    Function to create readable dataframe of results if a single column has 
    multiple classes in it. Initially solution for xgboost model results, 
    without breaking backwards compatibility for write_results.
    """
    scores = pd.DataFrame(ys)
    column_dict = {}
    for c in scores.columns:
        column_dict[c] = '{}_{}'.format(c, args.target[0])
    scores = scores.rename(columns=column_dict)
    scores.index = X.index
    scores = scores.join(dataloader.dataset.outcomes[[args.target[0], 'time_to_onset']])
    return scores


def main(args):
    """
    """
    import time
    
    print("Loading data.")
    # check if the datasets are already made
    time_old=time.time()

    try:
        # Try to read in the dictionary detailing the retrospective data
        d_name = 'split_dict_' + args.regularly_sampled*'regular' + (not args.regularly_sampled)*'irregular' + '.pkl'
        with open(os.path.join(args.data_dir, d_name), 'rb') as f:
            retro_specs = pickle.load(f)
        # If the load hasn't failed we know that the dictionary file exists, check for a full_dataframe file, if both exist raise an error because it's unclear what the program should have done.
        f_name = 'split_daily_data_' + args.regularly_sampled*'regular' + (not args.regularly_sampled)*'irregular' + '.hdf'
        assert not os.path.exists(os.path.join(args.data_dir, f_name)), 'Found a retrospective dictionary file and full dataframe file in {}, uncertain which should be used.'.format(args.data_dir)
        #Load the most recent dataset using the DATA_PATH_DICTIONARY_FILE
        with open(DATA_PATH_DICTIONARY_FILE, 'rb') as f:
            tmp = pickle.load(f)
        path = tmp[GET_PATH_DICT_KEY(args)]
#         # temp override
#         path = '/datasets/evidationdata/test/all_daily_data_regular_merged_nov29.hdf'
        d = os.path.dirname(path)
        f = os.path.basename(path)
        dfs=load_data(d, regular=args.regularly_sampled, only_healthy=args.only_healthy, fname=f)
        dfs = get_retro(dfs, retro_specs)

        # Clean up temporary objects
        del path, d, f, tmp

    except OSError: 
        # No retrospective file, try to just load all the data as default
        print('no retrospective specifications found, loading all latest data')
        with open(DATA_PATH_DICTIONARY_FILE, 'rb') as f:
            tmp = pickle.load(f)
        path = tmp[GET_PATH_DICT_KEY(args)]
        d = os.path.dirname(path)
        f = os.path.basename(path)
        dfs=load_data(d, regular=args.regularly_sampled, only_healthy=args.only_healthy, fname=f)

    if args.weekofyear:
        idx=pd.IndexSlice
        dfs['activity']['weekday']=(dfs['activity'].index.get_level_values('date').weekday<5).astype(np.int32) # add week of year
#         dfs['activity'].loc[:, idx['measurement', 'weekofyear']]=dfs['activity'].index.get_level_values('date').weekofyear.astype(np.int32)
        dfs['activity'].loc[:, idx['measurement', 'weekofyear']]=dfs['activity'].index.get_level_values('date').isocalendar().week .astype(np.int32).values
        # dfs['activity'].loc[:, idx['measurement_z', 'weekofyear']]=dfs['activity'].index.get_level_values('date').weekofyear.astype(np.int32)
        dfs['activity'].loc[:, idx['measurement_z', 'weekofyear']]=dfs['activity'].index.get_level_values('date').isocalendar().week.astype(np.int32).values
        dfs['activity'].loc[:, idx['measurement_noimp', 'weekofyear']]=dfs['activity'].index.get_level_values('date').isocalendar().week .astype(np.int32).values

        dfs['activity'].loc[:, idx['mask', 'weekofyear']]=np.ones(len(dfs['activity'])).astype(np.int32)
        dfs['activity'].loc[:, idx['time', 'weekofyear']]=np.ones(len(dfs['activity'])).astype(np.int32)
        
    print("Done loading data.", time.time() - time_old)

    # load or make participants
    if args.ignore_participants:
        # use everybody
        participants = list(np.unique(dfs['baseline'].index.get_level_values('participant_id')))
        # make sure these participants are also in activity
        participants = list(set(participants).intersection(set(dfs['activity'].index.get_level_values('participant_id'))))
        train_participants=participants.copy()
        valid_participants=participants.copy()
        test_participants=participants.copy()
        print("len of train_participants is: ", len(train_participants))
    elif os.path.exists(os.path.join(args.output_dir, f'train_participants.csv')) & os.path.exists(os.path.join(args.output_dir, 'valid_participants.csv')):
        print('Pre-loading train/val/test IDs')
        train_participants = pd.read_csv(os.path.join(args.output_dir,'train_participants.csv')).values.ravel().tolist()
        valid_participants = pd.read_csv(os.path.join(args.output_dir,'valid_participants.csv')).values.ravel().tolist()
        test_participants = pd.read_csv(os.path.join(args.output_dir,'test_participants.csv')).values.ravel().tolist()
        
    else:
        print('Generating train/val/test IDs from scratch')
        # set random_seed
        random.seed(args.seed)

        # apply train val test splits
        time_old=time.time()
        #participants = list(set(dfs['baseline'].index.get_level_values('participant_id')))
        participants = list(np.unique(dfs['baseline'].index.get_level_values('participant_id')))
        # make sure these participants are also in activity
        participants = list(set(participants).intersection(set(dfs['activity'].index.get_level_values('participant_id'))))
        
        if not args.temporally_split_participants:
            
            print('Guaranteeing positive and negative cases for label ' + args.target[0])
            pos_participants = set(np.unique(dfs['survey'][dfs['survey'][args.target[0]] == 1].index.get_level_values('participant_id')))
    
            neg_participants = set(participants) - pos_participants

            pos_participants = list(pos_participants)
            neg_participants = list(neg_participants)
        
            pos_splits = split_helper(pos_participants)
            neg_splits = split_helper(neg_participants)

            train_participants, valid_participants, test_participants = [pos + neg for pos, neg in zip(pos_splits, neg_splits)]

    #       # todo  train, valid, test splits to csv
        else:
            # we can define the train_val_test participants from the args.
#             df=dfs['survey'].copy()
#             df['ili_count'] = df['ili'].groupby('participant_id').apply(lambda x:x.rolling(2).apply(lambda y:y[0]<y[-1], raw=True).cumsum())
#             df['ili_count'] = df['ili_count'].fillna(0)
#             tmp_df = df['ili'].groupby('participant_id').first()
#             df = df.join(tmp_df, on=['participant_id'],how='left', rsuffix='_first')
#             # if it starts with 1, we also want to coutn it as a case.
#             df['ili_count']=df['ili_count']+df['ili_first'] 
#             df = df.loc[df.groupby('participant_id')['ili_count'].transform(max) ==df['ili_count']] # get only th maximum ili per participant
            
#             # get the onset dat for the maximum ili per participant
#             df = df.reset_index().loc[df.reset_index().groupby('participant_id')['date'].transform(min)==df.reset_index()['date']] 
            
#             # get the index of the onset of the latest event for a partiicpant.
#             participant_ili_tuples = df.set_index(['participant_id', 'date']).index.tolist() 
#             del tmp_df
#             del df
            participant_ili_tuples = get_latest_event_onset(dfs)
            
            # if test participants are defined get them
            if args.test_start_date is None:
                test_participants=[]
            else:
                test_participants = [p for p, d in participant_ili_tuples if (d>=args.test_start_date)&(d<=args.test_end_date)]

            # if validation participants are defined
            valid_participants = list(set([p for p, d in participant_ili_tuples if (d>=args.validation_start_date)&(d<=args.validation_end_date)]))
            if args.train_start_date is None:
                train_participants = list(set([p for p, d in participant_ili_tuples if (d<=args.train_end_date)]))
            else:
                train_participants = list(set([p for p, d in participant_ili_tuples if (d>=args.train_start_date)&(d<=args.train_end_date)]))

        pd.DataFrame(train_participants).to_csv( os.path.join(args.output_dir,'train_participants.csv'), index=False)
        pd.DataFrame(valid_participants).to_csv( os.path.join(args.output_dir,'valid_participants.csv'), index=False)
        pd.DataFrame(test_participants).to_csv( os.path.join(args.output_dir,'test_participants.csv'), index=False)
            
        print("Done creating splits.", time.time() - time_old)

    print("Making dataloaders")
    time_old=time.time()
    idx=pd.IndexSlice
    # subset the dataframes to their respective participants
    train_dfs = {k:v.loc[v.index.get_level_values('participant_id').isin(train_participants), :] for k, v in dfs.items()}
    valid_dfs = {k:v.loc[v.index.get_level_values('participant_id').isin(valid_participants), :] for k, v in dfs.items()}
    test_dfs = {k:v.loc[v.index.get_level_values('participant_id').isin(test_participants), :] for k, v in dfs.items()}
    

    if (args.modeltype=='grud')&(not(args.test)):
        # TODO: Add this as a flag to args. should be reconciled with args. positive_labels
        print(len(train_dfs['activity'] ))
        #train_dfs['survey'] = add_loss_mask(train_dfs['survey'], args)
        print(len(train_dfs['activity']))
        print(train_dfs['activity'].index[-10:])
        print(train_dfs['survey'].index[-10:])

        train_index = train_dfs['activity'].index.intersection( train_dfs['survey'].index)
        train_dfs['activity'] = train_dfs['activity'].reindex(train_index).sort_index()
        print(len(train_dfs['activity']))
        train_dfs['survey'] = train_dfs['survey'].reindex(train_index).sort_index()
        print(len(train_dfs['survey'] ))
#         raise

        train_participants = list(np.unique(train_dfs['baseline'].index.get_level_values('participant_id')))
        # make sure these participants are also in activity
        train_participants = list(set(train_participants).intersection(set(train_dfs['activity'].index.get_level_values('participant_id'))))
        
        #valid_dfs['survey'] = add_loss_mask(valid_dfs['survey'], args)
        valid_index = valid_dfs['activity'].index.intersection( valid_dfs['survey'].index)
        valid_dfs['activity'] = valid_dfs['activity'].reindex(valid_index).sort_index()
        valid_dfs['survey'] = valid_dfs['survey'].reindex(valid_index).sort_index()
        #test_dfs['survey'] = add_loss_mask(test_dfs['survey'], args)
        valid_participants = list(np.unique(valid_dfs['baseline'].index.get_level_values('participant_id')))
        # make sure these participants are also in activity
        valid_participants = list(set(valid_participants).intersection(set(valid_dfs['activity'].index.get_level_values('participant_id'))))
    
    # Preserve the original ili labels for use with create_lagged_data to ensure dropping of events which are too close together.
    if not (args.positive_labels is None) or not (args.bound_labels is None):
        # Keep the original ili event column for use with the function create_lagged_data 
        train_dfs['survey'].loc[:,'orig_ili'] = train_dfs['survey']['ili']
    
    # Change the positive_labels relative to date of onset for the training data only.
    if not (args.positive_labels is None):
        # TODO: Passing in this flag is not fully implemented
        # positive_labels is not None, thus change the positive labels relative to the dates of for the given labels.
        # Assumes that the occurence of two classes 
        for t in args.target:
            classes = np.unique(train_dfs['survey'][t])
            one_hot_classes = []
            for c in classes:
                if c == 0:
                    continue
                one_hot_classes.append((train_dfs['survey'][t] == c).astype(int).to_frame())
            changed_labels = []
            for s in one_hot_classes:
                changed_labels.append(change_labels(s, args.positive_labels, t))
            for i, c in zip(range(len(changed_labels)), classes[1:]):
                changed_labels[i] = changed_labels[i][t]*c
            new_labels = pd.concat(changed_labels, axis=1).max(axis=1)
            
            train_dfs['survey'].loc[:, t] = new_labels
    
    if args.add_missingness:
        # add missingness to entire row.
        test_dfs['numerics'].loc[:, idx['mask', :]]=test_dfs['numerics'].loc[:, idx['mask', :]].values*(np.random.randint(0, 2, len(test_dfs['numerics']))[:, np.newaxis])
    
    print(args.train_start_date, args.train_end_date, args.validation_end_date, args.test_end_date)
    
    # if train dates are given, restrict data further into the training dates
    if args.train_start_date is not None:
        print(len(train_dfs['activity']))
        train_dfs={k:(v.loc[v.index.get_level_values('date')>=args.train_start_date] if 'date' in v.index.names else v) for k, v in train_dfs.items()}
        print(len(train_dfs['activity']))

    if args.train_end_date is not None:
        print(len(train_dfs['activity']))
        train_dfs={k:(v.loc[v.index.get_level_values('date')<=args.train_end_date] if 'date' in v.index.names else v) for k, v in train_dfs.items()}
        print(len(train_dfs['activity']))
        
    if args.validation_end_date is not None:
        print(args.validation_end_date)
        print(len(valid_dfs['activity']))
        valid_dfs={k:(v.loc[(v.index.get_level_values('date')<=args.validation_end_date)] if 'date' in v.index.names else v) for k, v in valid_dfs.items()}
        print(len(valid_dfs['activity']))
        # some participants get dropped here, so we must get all remaining ones
        valid_participants = list(np.unique(valid_dfs['baseline'].index.get_level_values('participant_id')))
        # make sure these participants are also in activity
        valid_participants = list(set(valid_participants).intersection( set(valid_dfs['activity'].index.get_level_values('participant_id'))))
        
    if args.test_end_date is not None:
        print(len(test_dfs['activity']))
        test_dfs={k:(v.loc[(v.index.get_level_values('date')<=args.test_end_date)] if 'date' in v.index.names else v) for k, v in test_dfs.items()}
        print(len(test_dfs['activity']))
        
        # some participants get dropped here, so we must get all remaining ones
        test_participants = list(np.unique(test_dfs['baseline'].index.get_level_values('participant_id')))
        # make sure these participants are also in activity
        test_participants = list(set(test_participants).intersection(set(test_dfs['activity'].index.get_level_values('participant_id'))))                                  
                                  

    if (args.max_miss < 1)&(not(args.test)):
        # None of the observations with too much missingness gets removed in the validation or test sets, if the performance is wanted with those observations removed as well then that will need to be done manually.
        # Calculate the missingness for each date per participant
        print(f'restricting to dates with less than {args.max_miss} missingness')
        train_dfs['activity'] = restrict_missing(train_dfs['activity'], args)
        print('done missingness restriction')
        train_dfs['survey'] = train_dfs['survey'].loc[train_dfs['activity'].index]
#        train_dfs['baseline'] = train_dfs['baseline'].loc[train_dfs['activity'].index.get_level_values('participant_id')]

    if args.only_healthy:
        train_dfs = only_healthy(train_dfs, args)
    else:
#         print(args.target)
#         print(args.target[0])
#         print(sorted(train_dfs['survey'].columns.tolist()))
        # assert the target is in all of the train val and test sets
        assert train_dfs['survey'][args.target[0]].values.sum() >0, "The target does not exist in the train set." 
        assert valid_dfs['survey'][args.target[0]].values.sum() >0, "The target does not exist in the validation set." 
        assert test_dfs['survey'][args.target[0]].values.sum() >0, "The target does not exist in the test set." 
        
    if not(args.zscore):
        # zscore is already normalised, so we do not need to apply the scalar
        print('train scaler')
        train_dfs, scaler = apply_standard_scaler(train_dfs, args, scaler=None)
        print(len(train_dfs), len(train_participants))
        print(time.time() - time_old)
       
        with open(os.path.join(args.output_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)

        print('valid scaler')
        valid_dfs, _ = apply_standard_scaler(valid_dfs, args, scaler=scaler)
        print(len(valid_dfs), len(valid_participants))
        print(time.time() - time_old)
        
        print('test scaler')
        test_dfs, _ = apply_standard_scaler(test_dfs, args, scaler=scaler)
        print(len(test_dfs), len(test_participants))
        print("Done creating split dfs", time.time() - time_old)
        

    # get the average values for the dataframes (GRU-D needs this)
    time_old=time.time()
    if os.path.exists(os.path.join(args.output_dir, 'train_means.pickle')):
        with open(os.path.join(args.output_dir, 'train_means.pickle'), 'rb') as f:
            train_means = pickle.load(f)
    else:
        inds = train_dfs['survey'].loc[train_dfs['survey']['ili_48']==0, :].index.tolist()
        print('train_means: got index')
        if args.zscore:
            #train_means = train_dfs['activity'].loc[inds, idx[ 'measurement_z',:]].sample(5000).mean()
            # these should decay to 0
            train_means = np.zeros(len(train_dfs['activity'].loc[inds, idx[ 'measurement_z',:]].columns.tolist()))
        elif args.no_imputation:
            train_means = train_dfs['activity'].loc[inds, idx['measurement_noimp',:]].sample(5000).mean()
        else:
            # print(len(inds))
            
            # print(set(train_dfs['activity'].index.tolist()).difference(set(inds)))
            # print(set(inds).difference(set(train_dfs['activity'].index.tolist())))
            train_means = train_dfs['activity'].loc[inds, idx[ 'measurement',:]].sample(5000).mean()
        print('train_means: got mean')
        pickle.dump(train_means, open(os.path.join(args.output_dir, 'train_means.pickle'), 'wb'))
        
    print(train_means.shape)
    assert train_means.shape[0]==args.num_feature +1*args.weekofyear
    print("Done creating train_means.", time.time() - time_old)
    
    
    # make the torch Dataset and DataLoader for each of train, val, and test pandas DataFrames
    time_old=time.time()
    if args.resample:
        train_participants = list(np.unique(train_dfs['baseline'].index.get_level_values('participant_id')))
        train_participants = resample_fairly(train_participants, train_dfs, args.target[0])
        # todo save for checkpointing.
    else:
        train_participants=None
#         train_participants = list(np.unique(train_dfs['baseline'].index.get_level_values('participant_id')))

    collate_fn=id_collate
    if (args.modeltype in ['lstm', 'gru_simple', 'gru']) and (args.batch_size>1):
        collate_fn = merge_fn
        
    print(train_participants)
    
    train_dataset=ILIDataset(train_dfs, args, full_sequence=True, feat_subset=args.feat_subset, participants=train_participants)
    train_dataloader=DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_dataloader_workers, collate_fn=collate_fn)
    
    if args.test&(args.modeltype=='grud'):
        if args.ignore_participants:
            # save space:o
            del train_dfs
            del train_dataset
            del train_dataloader
    
    # TODO: This should happen near criterion.    
    if args.modeltype not in ['cnn', 'prob_nn']:
        try:
            pos_weight = 1 / train_dfs['survey']['ili'].mean()
            pos_weight = torch.tensor(pos_weight)
        except:
            pos_weight = None
            
            
    # we must make sure the loss is 0 before the validation start date.
    if args.validation_start_date is not None:
        if 'loss_mask' in valid_dfs['survey'].columns:
            valid_dfs['survey'].loc[(valid_dfs[ 'survey'].index.get_level_values('date')< args.validation_start_date), 'loss_mask']=0
            assert valid_dfs['survey'].loc[(valid_dfs[ 'survey'].index.get_level_values('date')< args.validation_start_date), 'loss_mask'].values.sum()==0, print('loss mask was not updated')
        else:
            valid_dfs['survey']['loss_mask']=1
            valid_dfs['survey'].loc[(valid_dfs[ 'survey'].index.get_level_values('date')< args.validation_start_date), 'loss_mask']=0
            assert 'loss_mask' in valid_dfs['survey'].columns

    valid_dataset=ILIDataset(valid_dfs, args, full_sequence=True, feat_subset=args.feat_subset, participants=valid_participants)
    valid_dataloader=DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=max(args.num_dataloader_workers-3,1), collate_fn=id_collate)
    
    test_dataset=ILIDataset(test_dfs, args, full_sequence=True, feat_subset=args.feat_subset, participants=test_participants)
    test_dataloader=DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_dataloader_workers, collate_fn=id_collate)
    
    if not(args.test)&(args.modeltype=='grud'):
        if args.ignore_participants:
            # save space:o
            del test_dfs
            del test_dataset
            del test_dataloader

    print("Done creating dataloaders/datasets", time.time() - time_old)
    
    # if there is a forecasting task, determine if they should be tte, or ae.
    print(type(args.target))
    

    if any([('24' in t) or ('48' in t) for t in args.target]):
        if len(args.target)>1:
            raise Exception("Incompatible timetoevent window with multitask leaning")
#         if args.forecast_type == 'timetoevent':
#             # convert the labels and data here.
#             print('Subsetting to timetoevent')
#             out = out.groupby(participant_col).apply(
#                 lambda x: x.iloc[:x.reset_index().loc[:, idx[:, target_col[1]]].idxmax().values[0] + 1])
#             out = out.droplevel(0)
#             out[target_col] = out[target_col].astype(int)
#             assert np.all(out[target_col].groupby(participant_col).sum() == 1)
#             print('out shape: %s' % (out.shape,))

     
    # TODO: fixed the conflict by commenting these out. Break these out into functions and have any necessary data modifiers in the model construction area.
    use_features = ['heart_rate_bpm', 'walk_steps', 'sleep_seconds',
                'steps__rolling_6_sum__max', 'steps__count', 'steps__sum', 'steps__mvpa__sum',
                'steps__dec_time__max',
                'heart_rate__perc_5th', 'heart_rate__perc_50th', 'heart_rate__perc_95th', 'heart_rate__mean',
                'heart_rate__stddev',
                'active_fitbit__sum', 'active_fitbit__awake__sum',
                'sleep__asleep__sum', 'sleep__main_efficiency', 'sleep__nap_count', 'sleep__really_awake__mean',
                'sleep__really_awake_regions__countDistinct', 'weekday']
    if args.weekofyear:
        use_features.append('weekofyear')
    
    # ********************** make the model ************************
    print("Making models")
    device='cuda' if torch.cuda.is_available() else 'cpu'
    
    input_size = args.num_feature + 1*args.weekofyear    
    cell_size = args.grud_hidden #67 # replace with args.grud_size
    hidden_size = args.grud_hidden # 67 # replace with args.grud_size
    if args.modeltype=='grud':
        model=GRUD( input_size, cell_size, hidden_size, train_means, device, fp16=not(args.opt_level=='O0'))
        # model modifiers
        # optimizer
        # criterion
        # train
    elif args.modeltype=='linear':
        model = LogisticRegression(input_size)
    elif args.modeltype=='lstm':
        model = LSTMmodel(input_size, hidden_size)
    elif args.modeltype=='gru_simple':
        model = GRUmodel(input_size*3, hidden_size)
    elif args.modeltype=='gru':
        model = GRUmodel(input_size, hidden_size)
    elif args.modeltype=='cnn':
        # shorten sequence length by 1 to not fit identity function
        cnn_seq_len = args.max_seq_len - 1
        model = CNN1D(input_size, seq_len=cnn_seq_len, num_feature=args.num_feature)
    elif args.modeltype=='prob_nn':
        prob_seq_len = args.max_seq_len - 1
        model = prob_NN(input_size, seq_len=prob_seq_len, num_feature=args.num_feature)
    elif args.modeltype=='fastgam':
        mdl = fastgam_mdl(args)
        if 'GAM_mdl.pickle' in os.listdir(args.output_dir):
            print('Model has already been trained, loading')
            mdl.load(args.output_dir)
        else:
            print('Training model for the first time')
            mdl.fit(train_dfs)
            mdl.save(args.output_dir)
        y_test, eta_test, idx_test = mdl.predict(test_dfs)
#         res_test = pd.DataFrame({'y':y_test, 'eta':eta_test}, index=idx_test).reset_index().assign(woy=lambda x: pd.to_datetime(x.date).dt.weekofyear)
        res_test = pd.DataFrame({'y':y_test, 'eta':eta_test}, index=idx_test).reset_index().assign(woy=lambda x: pd.to_datetime(x.date).dt.isocalendar().week )
        res_test.to_csv(os.path.join(args.output_dir, 'res_test.csv'), index=False)
        # Check model performance
        woy = res_test.groupby('woy').y.var()[res_test.groupby('woy').y.var() > 0].index.to_list()
#         auc_woy = res_test[res_test.woy.isin(woy)].assign(woy=lambda x: x.date.dt.weekofyear).groupby('woy').apply(lambda x: roc_auc_score(x.y, x.eta))
        auc_woy = res_test[res_test.woy.isin(woy)].assign(woy=lambda x: x.date.dt.isocalendar().week).groupby('woy').apply(lambda x: roc_auc_score(x.y, x.eta))
        auc_all = roc_auc_score(res_test.y, res_test.eta)
        print('AUROC for all days: %0.1f, for week-of-year: %0.1f' % (auc_all*100, auc_woy.mean()*100))

    elif args.modeltype=='ar_mdl':
        print('Using full batch set up for ar_mdl')
        # Assign to the different values
        survey_train, survey_valid, survey_test = train_dfs['survey'], valid_dfs['survey'], test_dfs['survey']
        # Use the measurement_z or measurement given as the z_score argument.
        col_set = 'measurement' + args.zscore*'_z' + args.no_imputation*'_noimp'
        activity_train, activity_valid, activity_test = train_dfs['activity'], valid_dfs['activity'], test_dfs['activity']
        activity_train = activity_train.loc[:,idx[col_set, use_features]]
        activity_train.insert(0,('idx','cidx'),activity_train.groupby('participant_id').cumcount().values+1)
        activity_valid = activity_valid.loc[:,idx[col_set, use_features]]
        activity_valid.insert(0,('idx','cidx'),activity_valid.groupby('participant_id').cumcount().values+1)
        activity_test = activity_test.loc[:,idx[col_set, use_features]]
        activity_test.insert(0,('idx','cidx'),activity_test.groupby('participant_id').cumcount().values+1)
        # Get the lags by ID...
        stime = time.time()
        print('Creating fast lags')
        do_diff = True

        out_train = fast_lags(activity_train, args.days_ago, diff=do_diff)
        out_valid = fast_lags(activity_valid, args.days_ago, diff=do_diff)
        out_test = fast_lags(activity_test, args.days_ago, diff=do_diff)
        print('Create target dataframes')
        target_train = pd.DataFrame(survey_train.loc[:,args.target[0]].astype(int))
        target_train.columns = pd.MultiIndex.from_tuples([['label',args.target[0]]],names=['df_type', 'value'])
        target_valid = pd.DataFrame(survey_valid.loc[:,args.target[0]].astype(int))
        target_valid.columns = pd.MultiIndex.from_tuples([['label',args.target[0]]],names=['df_type', 'value'])
        target_test = pd.DataFrame(survey_test.loc[:,args.target[0]].astype(int))
        target_test.columns = pd.MultiIndex.from_tuples([['label',args.target[0]]],names=['df_type', 'value'])

        print('Merging')
        out_train = out_train.join(target_train)
        out_valid = out_valid.join(target_valid)
        out_test = out_test.join(target_test)
        etime = time.time()
        print(f'Time: {(etime - stime) / 60:.2f} mins')
   
        if args.forecast_type == 'timetoevent':
            print('Subetting for time2event')
            out_train = out_train.join(out_train.assign(cidx=out_train.groupby('participant_id').cumcount()).loc[out_train[('label',args.target[0])]==1].groupby('participant_id').head(1)[['cidx']].droplevel(1))
            out_train = out_train[out_train.groupby('participant_id').cumcount() <= out_train.cidx].drop(columns=['cidx'])
            out_valid = out_valid.join(out_valid.assign(cidx=out_valid.groupby('participant_id').cumcount()).loc[out_valid[('label',args.target[0])]==1].groupby('participant_id').head(1)[['cidx']].droplevel(1))
            out_valid = out_valid[out_valid.groupby('participant_id').cumcount() <= out_valid.cidx].drop(columns=['cidx'])
            out_test = out_test.join(out_test.assign(cidx=out_test.groupby('participant_id').cumcount()).loc[out_test[('label',args.target[0])]==1].groupby('participant_id').head(1)[['cidx']].droplevel(1))
            out_test = out_test[out_test.groupby('participant_id').cumcount() <= out_test.cidx].drop(columns=['cidx'])
        
        # Remove any person that does have at least lags+1
        nids_train = out_train.index.get_level_values('participant_id').value_counts()
        nids_valid = out_valid.index.get_level_values('participant_id').value_counts()
        nids_test = out_test.index.get_level_values('participant_id').value_counts()
        out_train = out_train[out_train.index.get_level_values('participant_id').isin(idx[nids_train[nids_train >= args.days_ago + 1].index])]
        out_valid = out_valid[out_valid.index.get_level_values('participant_id').isin(idx[nids_valid[nids_valid >= args.days_ago + 1].index])]
        out_test = out_test[out_test.index.get_level_values('participant_id').isin(idx[nids_test[nids_test >= args.days_ago + 1].index])]
        # Fill missing (because last row has no missing values, groupby is not necessay)
        assert out_train.groupby('participant_id').tail(1).notnull().all().all()
        assert out_valid.groupby('participant_id').tail(1).notnull().all().all()
        assert out_test.groupby('participant_id').tail(1).notnull().all().all()
        print(out_test.loc[:,idx[:,'walk_steps']].head(2))

        #TODO: Jaryd, are we still using model construction?
#         from funs_lin import ILIDataset_fcast, lin_mdl
#         model = lin_mdl(args=args, num_features=48)
#         model.fit(dataloader=train_dataloader)
#         model.tune(dataloader=valid_dataloader)
        
#         # save model
#         try:
#             lin_mdl.save()
#         except:
#             raise Exception("lin_mdl.save() is not implemented yet.")
        
#         # save weights
#         scores = mdl.predict(dataloader=valid_dataloader)
#         (args=args, scores=scores, dataloader=valid_dataloader, istestset=False)
#         scores = mdl.predict(dataloader=test_dataloader)
#         write_results(args=args, scores=scores, dataloader=test_dataloader, istestset=True)

        from glmnet_python import glmnet
        print('---------- Linear model -----------')
        # Fill diff=True with zeros
        out_train = out_train.fillna(0) #(method='backfill') 
        out_valid = out_valid.fillna(0) #(method='backfill')
        out_test = out_test.fillna(0) #(method='backfill')

        print('Splitting X/y')
        X_train, y_train = out_train.drop(columns=[('label', args.target[0])]).values, out_train[('label', args.target[0])].values
        X_valid, y_valid = out_valid.drop(columns=[('label', args.target[0])]).values, out_valid[('label', args.target[0])].values
        X_test, y_test = out_test.drop(columns=[('label', args.target[0])]).values, out_test[('label', args.target[0])].values

        # Transform the data    
        enc = processor()
        X_train2 = enc.transform(X_train)
        X_valid2 = enc.transform(X_valid)
        X_test2 = enc.transform(X_test)
        
        print(enc.scaler.mean_)

        with open(os.path.join(args.output_dir, 'processor.pkl'), 'wb') as f:
            pickle.dump(enc, f)

        mdl = ar_mdl(args, nlambda=50)
        mdl.train(X_train2, y_train, X_valid2, y_valid)
        with open(os.path.join(args.output_dir, 'ar_model.bin'), 'wb') as f:
            pickle.dump(mdl, f)              
        
        # Store the coefficient info
        df_bhat = out_train.columns.to_frame().reset_index(None,True).rename(columns={'df_type':'tt','value':'cn'})
        df_bhat = df_bhat[df_bhat.tt!='label'].assign(trans='none')
        df_bhat = pd.concat([df_bhat, df_bhat.assign(trans='square')]).reset_index(None,True)
        df_bhat = df_bhat.assign(bhat = mdl.bhat_star)

        df_bhat.to_csv(os.path.join(args.output_dir,'df_bhat.csv'),index=False)
        
        # Save the risk scores for later
        eta_train = mdl.predict(X_train2)
        eta_valid = mdl.predict(X_valid2)
        eta_test = mdl.predict(X_test2)
        auc_train = roc_auc_score(y_train, eta_train)
        auc_valid = roc_auc_score(y_valid, eta_valid)
        auc_test = roc_auc_score(y_test, eta_test)
        print('AUROC by dataset: train=%0.1f, valid=%0.1f, test=%0.1f' % 
             (auc_train*100, auc_valid*100, auc_test*100))
        # Merge and save
        score_train = out_train.index.to_frame().reset_index(None,True).assign(lbl=y_train, score=eta_train,tt='train')
        score_valid = out_valid.index.to_frame().reset_index(None,True).assign(lbl=y_valid, score=eta_valid,tt='valid')
        score_test = out_test.index.to_frame().reset_index(None,True).assign(lbl=y_test, score=eta_test,tt='test')
        score_all = pd.concat([score_train, score_valid, score_test]).rename(columns = {'lbl':args.target[0], 
                                                                                        'score':args.target[0]+'_score'})
        score_test = score_all[score_all.tt=='test']
        print('Writing result')
        write_results(args=args, scores=score_test, dataloader=test_dataloader, istestset=True)
        write_results(args=args, scores=score_all, dataloader=valid_dataloader, istestset=False)
        return
    
    elif args.modeltype=='xgboost':
        from xgboost import XGBClassifier

        # No need to fill missing
        print('Splitting X/y')
#         X_train, y_train = out_train.drop(columns=[('label', args.target[0])]).values, out_train[('label', args.target[0])].values
#         X_valid, y_valid = out_valid.drop(columns=[('label', args.target[0])]).values, out_valid[('label', args.target[0])].values
#         X_test, y_test = out_test.drop(columns=[('label', args.target[0])]).values, out_test[('label', args.target[0])].values
        # create lagged data
        use_features = ['heart_rate_bpm', 'walk_steps', 'sleep_seconds',
                 'steps__rolling_6_sum__max', 'steps__count', 'steps__sum', 
                 'steps__mvpa__sum', 'steps__dec_time__max',
                 'heart_rate__perc_5th', 'heart_rate__perc_50th', 
                 'heart_rate__perc_95th', 'heart_rate__mean',
                 'heart_rate__stddev', 'active_fitbit__sum', 
                 'active_fitbit__awake__sum', 'sleep__asleep__sum', 
                 'sleep__main_efficiency', 'sleep__nap_count', 
                 'sleep__really_awake__mean', 
                 'sleep__really_awake_regions__countDistinct', 'weekday']
        
        fake_features = [c[1] for c in train_dataloader.dataset.numerics.columns if 'fake' in c[1]]
        use_features = use_features + fake_features

        if args.feat_regex != '':
            use_features = [f for f in use_features if re.search(args.feat_regex, f)]

        if args.weekofyear:
            use_features.append('weekofyear')       
        
        # Assume if the final dataframes were generated the others were also generated.
        if os.path.exists(os.path.join(args.checkpt_dir, 'X_test.pkl')):
            with open(os.path.join(args.checkpt_dir, 'X_train.pkl'), 'rb') as f:
                X_train = pickle.load(f)
            with open(os.path.join(args.checkpt_dir, 'y_train.pkl'), 'rb') as f:
                y_train = pickle.load(f)
            with open(os.path.join(args.checkpt_dir, 'sample_weight.pkl'), 'rb') as f:
                sample_weight = pickle.load(f)
            with open(os.path.join(args.checkpt_dir, 'X_valid.pkl'), 'rb') as f:
                X_valid = pickle.load(f)
            with open(os.path.join(args.checkpt_dir, 'y_valid.pkl'), 'rb') as f:
                y_valid = pickle.load(f)
            with open(os.path.join(args.checkpt_dir, 'X_test.pkl'), 'rb') as f:
                X_test = pickle.load(f)
            with open(os.path.join(args.checkpt_dir, 'y_test.pkl'), 'rb') as f:
                y_test = pickle.load(f)
        #Check if args.bound_labels doesn't evaluate to False as a boolean. It should be safe to assume args.bound_labels is None or a list.
        else:
            X_train, y_train, sample_weight = create_lagged_data(train_dataloader, args, use_features=use_features, target=args.target, bound_events=bool(args.bound_labels))
            X_valid, y_valid, _ = create_lagged_data(valid_dataloader, args, use_features=use_features, target=args.target, bound_events=False)
            X_test, y_test, _ = create_lagged_data(test_dataloader, args, use_features=use_features, target=args.target, bound_events=False)
            if args.checkpt_dir is None:
                args.checkpt_dir = args.output_dir
            with open(os.path.join(args.checkpt_dir, 'X_train.pkl'), 'wb') as f:
                pickle.dump(X_train, f, protocol=4)
            with open(os.path.join(args.checkpt_dir, 'y_train.pkl'), 'wb') as f:
                pickle.dump(y_train, f, protocol=4)
            with open(os.path.join(args.checkpt_dir, 'sample_weight.pkl'), 'wb') as f:
                pickle.dump(sample_weight, f, protocol=4)
            with open(os.path.join(args.checkpt_dir, 'X_valid.pkl'), 'wb') as f:
                pickle.dump(X_valid, f, protocol=4)
            with open(os.path.join(args.checkpt_dir, 'y_valid.pkl'), 'wb') as f:
                pickle.dump(y_valid, f, protocol=4)
            with open(os.path.join(args.checkpt_dir, 'X_test.pkl'), 'wb') as f:
                pickle.dump(X_test, f, protocol=4)
            with open(os.path.join(args.checkpt_dir, 'y_test.pkl'), 'wb') as f:
                pickle.dump(y_test, f, protocol=4)
        
        # After using all previous data to build validation and test features, restrict the dates evaluated on, same purpose as the mask for the grud.
        if args.validation_start_date is not None:
            X_valid = X_valid[X_valid.index.get_level_values('date') >= args.validation_start_date]
            y_valid = y_valid[y_valid.index.get_level_values('date') >= args.validation_start_date]
        if args.test_start_date is not None:
            X_test = X_test[X_test.index.get_level_values('date') >= args.test_start_date]
            y_test = y_test[y_test.index.get_level_values('date') >= args.test_start_date]

        
        participants = np.unique(X_test.index.get_level_values('participant_id'))
        pd.DataFrame(participants).to_csv(os.path.join(args.output_dir, 'test_batch_order.csv'))

        print(X_train.columns)

        if args.unbalanced:
            SCALE_POS_WEIGHT = 1
        else:
            try:
                print(y_train)
                SCALE_POS_WEIGHT = (y_train == 0).sum() / (y_train == 1).sum()  # using train-set ratio
            except:
                SCALE_POS_WEIGHT=1
        
        if args.test:
            print('Loading model')
            print(os.path.join(args.output_dir.replace('_bounded', ''), 'xgboost_model.bin'))
#             with open(os.path.join(args.output_dir.replace('_bounded', ''), 'xgboost_model.bin'), 'rb') as f:
#                 model = pickle.load(f) 
            with open(os.path.join(args.output_dir, 'xgboost_model.bin'), 'rb') as f:
                model = pickle.load(f) 
        else:
            #train model
            print('Training model')
            stime = time.time()
            
            if os.path.exists(os.path.join(args.checkpt_dir, 'xgb_rand_params.pkl')):
                with open(os.path.join(args.checkpt_dir, 'xgb_rand_params.pkl'), 'rb') as f:
                    space = pickle.load(f)
            else:
                space = []
                for i in range(100):
                    space.append({
                            #'objective' : 'multi:softmax',
                            'n_estimators' : np.random.choice(np.arange(10, 200, 10), 1)[0],
                            #'eta' : np.random.choice(np.arange(0.01, 0.1, 0.02), 1)[0],
                            'max_depth' : np.random.choice(np.arange(1, 10), 1)[0],
                            #'min_child_weight' : np.random.choice(np.arange(1, 6, 1), 1)[0],
                            'subsample' : np.random.choice(np.arange(0.5, 1, 0.05), 1)[0],
                            'gamma' : np.random.choice(np.arange(0.5, 1, 0.05), 1)[0],
                            'colsample_bytree' : np.random.choice(np.arange(0.5, 1, 0.05), 1)[0],
                            #'silent' : 1,
                            #'tree_method' : 'gpu_hist',
                            #add positive class weight up-scaling
                            'scale_pos_weight' : SCALE_POS_WEIGHT,
                            'seed' : np.random.choice(np.arange(1,10000),1)[0],
                            'nthread' : 5,
                            'tree_method' : args.xgb_method,
                            'scale_pos_weight' : sample_weight
                            })
                    with open(os.path.join(args.checkpt_dir, 'xgb_rand_params.pkl'), 'wb') as f:
                        pickle.dump(space, f, protocol=4)

            cv_performance = -1
            i = 0
            if os.path.exists(os.path.join(args.checkpt_dir, 'xgb_iter.pkl')):
                with open(os.path.join(args.checkpt_dir, 'xgb_iter.pkl'), 'rb') as f:
                    i = pkl.load(f)
            
            if os.path.exists(os.path.join(args.checkpt_dir, 'best_perf.pkl')):
                with open(os.path.join(args.checkpt_dir, 'best_perf.pkl'), 'rb') as f:
                    cv_performance = pickle.load(f)
            
            while i < len(space):
                params = space[i]
                print(i)
                tmp_model = XGBClassifier(**params)
                tmp_model.fit(X_train, y_train)
                #yh_train = tmp_model.predict_proba(X_train)
                yh_valid = tmp_model.predict_proba(X_valid)
                # Check if it's a binary classification, this changes the correct arguments for skm.roc_auc_score
                if len(np.unique(y_train)) == 2:
                    yh_valid = yh_valid[:, 1]
                tmp_perf = skm.roc_auc_score(y_valid, yh_valid, multi_class='ovo')
                if tmp_perf > cv_performance:
                    cv_performance = tmp_perf
                    model = tmp_model
                    with open(os.path.join(args.checkpt_dir, 'best_model.pkl'), 'wb') as f:
                        pickle.dump(model, f)
                    with open(os.path.join(args.checkpt_dir, 'best_perf.pkl'), 'wb') as f:
                        pickle.dump(cv_performance, f)
                # fix memory leak with repeated xgboost fit calls.        
                gc.collect()
                i = i + 1
                with open(os.path.join(args.checkpt_dir, 'xgb_iter.pkl'), 'wb') as f:
                    pkl.dump(i, f)
            # load the best model from the cur_model.pkl file. 
            # Necessary if the preemption happens and the best model was trained in a previous session.
            with open(os.path.join(args.checkpt_dir, 'best_model.pkl'), 'rb') as f:
                model = pickle.load(f)

            pickle.dump(model, open(os.path.join(args.output_dir, "xgboost_model.bin"), "wb"))
            etime = time.time()
            print('Took %i seconds to fit XGboost' % (etime - stime))
    #         model = nn.ModuleDict({'model':model})
        if np.array_equal(np.unique(y_train), [0,1]):
            # Binary classifier only need probability class 1
            ys_train = model.predict_proba(X_train)[:, 1]
            ys_valid = model.predict_proba(X_valid)[:, 1]
            ys_test = model.predict_proba(X_test)[:, 1]
        else:
            # Require all class probabilities for multiclass
            ys_train = model.predict_proba(X_train)
            ys_train = multi_class_column_scores(ys_train, X_train, train_dataloader, args) 
            ys_valid = model.predict_proba(X_valid)
            ys_valid = multi_class_column_scores(ys_valid, X_valid, valid_dataloader, args)
            ys_test = model.predict_proba(X_test)
            ys_test = multi_class_column_scores(ys_test, X_test, test_dataloader, args)
        
        # save outputs
        write_results(args=args, scores=ys_valid, dataloader=valid_dataloader, istestset=False, participant_order=np.unique(X_valid.index.get_level_values('participant_id')))
        write_results(args=args, scores=ys_test, dataloader=test_dataloader, istestset=True,
                participant_order=np.unique(X_test.index.get_level_values('participant_id')))
        
        

        return
        

    else:
        raise NotImplementedError
        
        
    
    
    # for multitask learning objectives add the additional prediction heads
    if len(args.target) > 1:
        model = model.update(nn.ModuleDict({key: nn.Linear(hidden_size, 2) for key in args.target}))
        
    
    model.to(device)
    model.apply(weight_init)
        
    optimizer=RAdam(model.parameters(), lr=args.learning_rate)
    
    # apply torch amp for faster training
    if amp is not None:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)
    
    best_err=10e9
    start_epoch=0
    do_train = not(args.test) 
    if args.reload:
        # check if there is a model in output_dir
        if os.path.exists(os.path.join( args.output_dir, 'checkpoint_best.pth')):
            #reload the checkpoint and recreate the starting point.
            checkpoint_best = torch.load(os.path.join( args.output_dir, 'checkpoint_best.pth'), map_location=device)
            best_epoch = checkpoint_best['epoch']
            if os.path.exists(os.path.join(args.checkpt_dir, 'checkpt_cur.pth')):
                checkpoint_cur = torch.load(os.path.join(args.checkpt_dir, 'checkpt_cur.pth'), map_location=device)
                cur_epoch = checkpoint_cur['epoch']
            elif os.path.exists(os.path.join(args.checkpt_dir, 'checkpoint_cur.pth')):
                checkpoint_cur = torch.load(os.path.join(args.checkpt_dir, 'checkpoint_cur.pth'), map_location=device)
                cur_epoch = checkpoint_cur['epoch']
            else:
                cur_epoch = 0
                print('No checkpoint found at ', os.path.join(args.checkpt_dir, 'checkpt_cur.pth'))
            if best_epoch >= cur_epoch:
                checkpoint = checkpoint_best
                print('Only loading checkpoint from ', os.path.join( args.output_dir, 'checkpoint_best.pth'))
            else:
                checkpoint = checkpoint_cur
                print('Using checkpoint from ', os.path.join(args.checkpt_dir, 'checkpoint_cur.pth'))
        

            checkpoint['state_dict']= {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
            print(checkpoint['state_dict'].keys())
            print(checkpoint)
            
            try:
                model.load_state_dict(checkpoint['state_dict'])
            except:
                print(checkpoint['state_dict'].keys())
                model['model'].load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if amp is not None:
                amp.load_state_dict(checkpoint['amp'])
            start_epoch = checkpoint['epoch']
            best_err = checkpoint['best_acc']
            obj = checkpoint.copy()
            if not(args.test) and ('done_training' in checkpoint.keys()):
                do_train = not(checkpoint['done_training']) # only train if we are not telling to test and if we are not done training
        else:
            # if there is no checkpoint, make sure we delete any old existing logs.
            if os.path.exists(os.path.join(args.output_dir, 'val_err.log')):
                os.remove(os.path.join(args.output_dir, 'val_err.log'))
            
            
    
    
    # loss function
    if args.modeltype not in ['cnn', 'prob_nn']:
        criterion=nn.BCEWithLogitsLoss(pos_weight = pos_weight)
        criterion_valid=nn.BCEWithLogitsLoss(pos_weight = pos_weight, reduction='none')
    else:
        criterion=nn.BCELoss(reduction='none') # for grud have no reduction
        criterion_valid=nn.BCELoss(reduction='none') # for grud have no reduction
        #criterion=nn.MSELoss(reduction='mean')
        #criterion_valid=nn.MSELoss(reduction='mean')
        #criterion=nn.SmoothL1Loss(reduction='mean')
        #criterion_valid=nn.SmoothL1Loss(reduction='mean')
    
    main_target=sorted(list(args.target))[0]
    assert isinstance(main_target, str)
    col_set = 'measurement' + args.zscore*'_z' + args.no_imputation*'_noimp'
    if do_train:    
        print("Start training")
        print(len(train_dataloader.dataset.participants))
        print(len(set(train_dataloader.dataset.numerics.index.get_level_values('participant_id'))))
        #weight_idx = 0
        #weights = {'conv_1': {}, 'conv_2': {}, 'fc1': {}, 'fc2': {}, 'fc3': {}}
        epochs=tqdm(range(start_epoch, args.epochs), desc='epoch')
        for epoch in epochs:
            model.train()
            optimizer.zero_grad()
            batches=tqdm(enumerate(train_dataloader), total=len(train_dataloader))
            for batch_num, batch in batches:
                participants = batch[-1]
                try:batch = batch[0][0]
                except: batch = batch[0]
                batch={k:v.to(device) for k, v in batch.items()}

                prediction, hidden_states = model(batch[col_set].float(),
                                                  batch[col_set].float(), 
                                                  batch['mask'].float(),
                                                  batch['time'].float(), 
                                                  pad_mask= batch['obs_mask'],
                                                  return_hidden=True)
                if len(args.target)==1:# todo check this test
                
                    if args.batch_size==1:
                        prediction = prediction.unsqueeze(0)

                    if args.modeltype not in ['cnn', 'prob_nn']:
                        
                        #loss=criterion(prediction[:,:,1].squeeze(-1)[batch['obs_mask']==1], batch[main_target][batch['obs_mask']==1]).sum()
                        if train_dataloader.batch_size ==1:
#                             loss = criterion(prediction[:,:,1].reshape((args.batch_size,-1))[batch['obs_mask']==1], batch[main_target][batch['obs_mask']==1]).sum()
                            loss = criterion(prediction[:,:,1].reshape((args.batch_size,-1))[batch['obs_mask']==1], batch[main_target][batch['obs_mask']==1])# reduce mean ==None
                            if 'loss_mask' in batch.keys():
                                loss = (loss*batch['loss_mask']).mean()
                            else:
                                loss= loss.mean()
                        else:
                            loss = criterion(prediction[:, 1].view(-1)[batch['obs_mask'].data==1], batch[main_target].data[batch['obs_mask'].data==1]).sum()
                    else:
                        #for layer in weights.keys():
                            #weights[layer][weight_idx] = getattr(model['model'], layer).weight.data.numpy().flatten()
                            #pd.DataFrame(weights[layer]).to_csv(os.path.join(args.output_dir, layer + '_weights.csv'))
                        mus = prediction[0]
                        #print("mus: " + str(torch.min(torch.mean(mus))))
                        sigma = prediction[1]
                        #print("logstds: " + str(torch.min(torch.mean(logstds))))
                        x = batch[col_set].float()[:, -1, :] 
                        log_lklhd = cnn_lklhd(mus, sigma, x)
                        # using sigmoid because it needs the score to be between zero and one.
                        # This will guarantee that property.
                        score = torch.sigmoid(log_lklhd)
                        assert torch.sum(torch.isnan(mus)) == 0, 'missing values in mus'
                        try:
                            loss=-torch.mean(log_lklhd)
                            #loss=criterion(score, torch.ones(score.shape[0]))
                        except:
                            print("got error")
                            print(score)
                            raise
                        #weight_idx = (weight_idx + 1) % 10
                else:
                    # it is multitask:
                    loss=0
                    for k, pred_head in model.items():
                        predictions = pred_head(hidden_states)
                        if args.batch_size==1:
                            prediction = prediction.unsqueeze(0)
                        loss+=criterion(prediction[:,:,1].reshape((args.batch_size, -1))[batch['obs_mask']==1], batch[k][batch['obs_mask']==1])
                        
        
                
                assert len(loss.shape)<2
                
                # backprop the loss                
                if amp is not None:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                
                if batch_num % args.batches_per_gradient == args.batches_per_gradient-1:
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    
                # update TQDM
                batches.set_description(desc=f'{loss.cpu().data.numpy():.4f}')
                
                if (batch_num==10) &(args.small):
                    break
                
                
                
            # ******************valid *******************
            
            losses_epoch_valid = {k:[] for k in args.target}
            optimizer.zero_grad()
            model.eval()
            if args.modeltype in ['cnn', 'prob_nn']:
                scores = []
                labels = []
            batches=tqdm(valid_dataloader, total=len(valid_dataloader))
            for batch in batches:
                participants = batch[-1]
                batch = batch[0][0]
                batch={k:v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    prediction, hidden_states = model(batch[col_set].float(),
                                                  batch[col_set].float(), 
                                                  batch['mask'].float(),
                                                  batch['time'].float(), 
                                                  pad_mask= batch['obs_mask'],
                                                  return_hidden=True)
 
                    if len(args.target)==1:# todo check this test

                        if args.batch_size==1:
                            prediction = prediction.unsqueeze(0)

                        if args.modeltype in ['cnn', 'prob_nn']: 
                            
                            mus = prediction[0]
                            sigma = prediction[1]
                            x = batch[col_set].float()[:, -1, :]
                            log_lklhd = cnn_lklhd(mus, sigma, x)
                            scores.append(log_lklhd)
                            labels.append(batch[main_target][:, -1])
                            loss=-torch.mean(log_lklhd)
                            #loss=criterion(torch.sigmoid(log_lklhd), torch.ones(log_lklhd.shape[0]))
                        else:
                            # loss = criterion_valid(prediction[:,:,1].reshape((valid_dataloader.batch_size, -1))[batch['obs_mask']==1], batch[main_target][batch['obs_mask']==1]).sum()
                            
                            loss = criterion_valid(prediction[:,:,1].reshape((valid_dataloader.batch_size, -1))[batch['obs_mask']==1], batch[main_target][batch['obs_mask']==1])
                            if 'loss_mask' in batch.keys():
                                loss = (loss*batch['loss_mask']).mean()
                            else:
                                loss = loss.mean()
 
                        losses_epoch_valid[main_target] += loss.cpu().data.numpy().ravel().tolist()
                    else:
                        # it is multitask:
                        loss_dict={}
                        for k, pred_head in model.items():
                            predictions = pread_head(hidden_states)
                            if args.batch_size==1:
                                prediction = prediction.unsqueeze(0)
                            loss_dict[k]=criterion_valid(prediction[:,:,1].squeeze(-1)[batch['obs_mask']==1], batch[k][batch['obs_mask']==1]).cpu().data.numpy().ravel()
                                            
                        losses_epoch_valid={k:v+list(loss_dict[k]) for k, v in losses_epoch_valid.items()}
                        
            if isinstance(losses_epoch_valid, dict):
                losses_epoch_valid = np.asarray([v for k,v in losses_epoch_valid.items()]).ravel()
            else:
                losses_epoch_valid = np.concatenate(losses_epoch_valid)
            
            if args.modeltype in ['cnn', 'prob_nn']:
                try: 
                    times_written = times_written + 1
                except NameError:
                    times_written = 0

                results = {}
                results['score'] = np.concatenate(scores, axis=0)
                results[args.target[0]] = np.concatenate(labels, axis=0)
                pd.DataFrame(results).to_csv(os.path.join(args.output_dir, args.target[0] + '_val_' + str(times_written) + '.csv'))
            if (np.mean(losses_epoch_valid) <= best_err) or epoch==0:
                best_err = np.mean(losses_epoch_valid)
                # If we save using the predefined names, we can load using `from_pretrained`
                print('val_err: ', best_err, ', epoch: ', epoch+1, '; improved' )
                # Adding persistent log through preemptions to see changes in val_err.
                with open(os.path.join(args.output_dir, 'val_err.log'), 'a') as f:
                    f.write('val_err: ' + str(best_err) + ', epoch: ' + str(epoch+1) + '; improved\n')
                output_model_file = os.path.join(args.output_dir, 'pytorch_model.bin')
                output_config_file = os.path.join(args.output_dir, 'config.json')

                if hasattr(model, 'module'):
                    if amp is not None:
                        obj = {
                            'epoch': epoch+1,
                            'state_dict': model.module.state_dict(),
                            'best_acc': np.mean(losses_epoch_valid) ,
                            'optimizer' : optimizer.state_dict(),
                            'amp': amp.state_dict(),
                            'done_training':False,
                        }
                    else:
                        obj = {
                            'epoch': epoch+1,
                            'state_dict': model.module.state_dict(),
                            'best_acc': np.mean(losses_epoch_valid) ,
                            'optimizer' : optimizer.state_dict(),
                            'done_training':False,
                        }
                    torch.save(obj, os.path.join(args.output_dir,'checkpoint_best.pth'))
                else:
                    if amp is not None:
                        obj = {
                            'epoch': epoch+1,
                            'state_dict': model.state_dict(),
                            'best_acc': np.mean(losses_epoch_valid) ,
                            'optimizer' : optimizer.state_dict(),
                            'amp': amp.state_dict(),
                            'done_training':False,
                        }
                    else:
                        obj = {
                           'epoch': epoch+1,
                           'state_dict': model.state_dict(),
                           'best_acc': np.mean(losses_epoch_valid) ,
                           'optimizer' : optimizer.state_dict(),
                           'done_training':False,
                        }
                    torch.save(obj, os.path.join(args.output_dir,'checkpoint_best.pth'))
            elif epoch-obj['epoch']-1 > args.patience:
                checkpoint = torch.load(os.path.join(args.output_dir,'checkpoint_best.pth'))
                checkpoint['done_training']=True
                torch.save(checkpoint, os.path.join(args.output_dir,'checkpoint_best.pth'))
                break
            else:
                cur_err = np.mean(losses_epoch_valid)
                # If we save using the predefined names, we can load using `from_pretrained`
                print('val_err: ', cur_err, ', epoch: ', epoch+1 )
                # Adding persistent log through preemptions to see changes in val_err.
                with open(os.path.join(args.output_dir, 'val_err.log'), 'a') as f:
                    f.write('val_err: ' + str(cur_err) + ', epoch: ' + str(epoch+1) + '\n')
                output_model_file = os.path.join(args.checkpt_dir, 'pytorch_model.bin')
                output_config_file = os.path.join(args.checkpt_dir, 'config.json')

                if hasattr(model, 'module'):
                    if amp is not None:
                        obj2 = {
                            'epoch': epoch+1,
                            'state_dict': model.module.state_dict(),
                            'best_acc': best_err ,
                            'optimizer' : optimizer.state_dict(),
                            'amp': amp.state_dict(),
                            'done_training':False,
                        }
                    else:
                        obj2 = {
                            'epoch': epoch+1,
                            'state_dict': model.module.state_dict(),
                            'best_acc': best_err ,
                            'optimizer' : optimizer.state_dict(),
                            'done_training':False,
                        }
                    torch.save(obj2, os.path.join(args.checkpt_dir,'checkpoint_cur.pth'))
                else:
                    if amp is not None:
                        obj2 = {
                            'epoch': epoch+1,
                            'state_dict': model.state_dict(),
                            'best_acc': best_err ,
                            'optimizer' : optimizer.state_dict(),
                            'amp': amp.state_dict(),
                            'done_training':False,
                        }
                    else:
                        obj2 = {
                           'epoch': epoch+1,
                           'state_dict': model.state_dict(),
                           'best_acc': best_err,
                           'optimizer' : optimizer.state_dict(),
                           'done_training':False,
                        }
                    torch.save(obj2, os.path.join(args.checkpt_dir,'checkpoint_cur.pth'))
            epochs.set_description(desc='')
            
        #return
                
    # ********************************** now test the model **********************************
    try:
        del train_dataloader
        del train_dataset
    except:
        pass
    
    
    
    proba_projection=torch.nn.Softmax(dim=1)
    if len(args.target)>1:
        scores={}
    else:
        scores=[]
    labels=[]
    participants = []
    model.eval()
    batches=tqdm(valid_dataloader, total=len(valid_dataloader))
    for batch in batches:
        assert valid_dataloader.batch_size == 1, "batch size for valid_dataloader must be one."
        participants.append(batch[-1][0])
        batch = batch[0][0]
        batch={k:v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            if args.modeltype in ['cnn', 'prob_nn']:
                predictions = []
                i = args.max_seq_len
                num_predictions = batch[col_set].shape[1]

                while i < num_predictions:
                    X = batch[col_set][:,(i-args.max_seq_len):i,:]
                    
                    prediction, hidden = model(X,
                                                      batch[col_set].float(),
                                                      batch['mask'].float(),
                                                      batch['time'].float(),
                                                      pad_mask=batch['obs_mask'],
                                                      return_hidden=True)
                            
                    mus = prediction[0]
                    sigma = prediction[1]
                    x = batch[col_set].float()[:, i, :]
                    log_lklhd = cnn_lklhd(mus, sigma, x)
                    predictions.append(log_lklhd) 
                    i += 1
                    score = torch.sigmoid(log_lklhd)
                    batch_size = batch[col_set].shape[0]
                    loss=torch.mean(log_lklhd)

                if len(predictions) == 0:
                    prediction = torch.tensor([np.nan]*num_predictions, dtype=torch.float32)
                    scores.append(prediction.view(1, prediction.shape[0]).detach().cpu().numpy())
                else:
                    #TODO: test this section...
                    num_miss = num_predictions - len(predictions)
                    missing = torch.tensor([np.nan]*num_miss, dtype=torch.float32).view(1, num_miss)
                    prediction = proba_projection(torch.cat(predictions, dim=0).view(1, len(predictions)))
                    prediction = torch.cat([missing, prediction], dim = 1)
                    scores.append(prediction.detach().cpu().numpy())

                labels.append(batch[main_target][:,args.max_seq_len:].detach().cpu().numpy())
            else:
                prediction, hidden_states = model(batch[col_set].float(),
                                          batch[col_set].float(), 
                                          batch['mask'].float(), 
                                          batch['time'].float(), 
                                          pad_mask= batch['obs_mask'],
                                          return_hidden=True)
            
                if len(args.target)>1:
                    for k, pred_head in model.items():
                        predictions = model[k](hidden_states)
                        scores.append(proba_projection(prediction.view(-1, 2))[:, -1].detach().cpu().numpy())
                else:          
                    scores.append(proba_projection(prediction.view(-1, 2))[:, -1].detach().cpu().numpy()) # I just learned how to spell detach
                labels.append(batch[main_target].detach().cpu().numpy())
    
    #TODO: remove this order check
    # write the order of participants seen in the test set to a file
    pd.DataFrame(participants).to_csv(os.path.join(args.output_dir, 'valid_batch_order.csv'))

    labels=np.concatenate([l.ravel() for l in labels], axis=0)
    scores=np.concatenate([l.ravel() for l in scores], axis=0)

    # save outputs
    write_results(args=args, scores=scores, dataloader=valid_dataloader, istestset=False, participant_order=participants)#X_valid.index.get_level_values('participant_id')))
    del valid_dataloader
    del valid_dataset
    
    
    proba_projection=torch.nn.Softmax(dim=1)
    if len(args.target)>1:
        scores={}
    else:
        scores=[]
    labels=[]
    participants = []
    model.eval()
    batches=tqdm(test_dataloader, total=len(test_dataloader))
    for batch in batches:
        assert test_dataloader.batch_size == 1, "batch size for test_dataloader must be one."
        participants.append(batch[-1][0])
        batch = batch[0][0]
        batch={k:v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            if args.modeltype in ['cnn', 'prob_nn']:
                predictions = []
                i = args.max_seq_len
                num_predictions = batch[col_set].shape[1]

                while i < num_predictions:
                    X = batch[col_set][:,(i-args.max_seq_len):i,:]
                    
                    prediction, hidden = model(X,
                                                      batch[col_set].float(),
                                                      batch['mask'].float(),
                                                      batch['time'].float(),
                                                      pad_mask=batch['obs_mask'],
                                                      return_hidden=True)
                            
                    mus = prediction[0]
                    sigma = prediction[1]
                    x = batch[col_set].float()[:, i, :]
                    log_lklhd = cnn_lklhd(mus, sigma, x)
                    predictions.append(log_lklhd) 
                    i += 1
                    score = torch.sigmoid(log_lklhd)
                    batch_size = batch[col_set].shape[0]
                    loss=torch.mean(log_lklhd)

                if len(predictions) == 0:
                    prediction = torch.tensor([np.nan]*num_predictions, dtype=torch.float32)
                    scores.append(prediction.view(1, prediction.shape[0]).detach().cpu().numpy())
                else:
                    #TODO: test this section...
                    num_miss = num_predictions - len(predictions)
                    missing = torch.tensor([np.nan]*num_miss, dtype=torch.float32).view(1, num_miss)
                    prediction = proba_projection(torch.cat(predictions, dim=0).view(1, len(predictions)))
                    prediction = torch.cat([missing, prediction], dim = 1)
                    scores.append(prediction.detach().cpu().numpy())

                labels.append(batch[main_target][:,args.max_seq_len:].detach().cpu().numpy())
            else:
                prediction, hidden_states = model(batch[col_set].float(),
                                          batch[col_set].float(), 
                                          batch['mask'].float(), 
                                          batch['time'].float(), 
                                          pad_mask= batch['obs_mask'],
                                          return_hidden=True)
            
                if len(args.target)>1:
                    for k, pred_head in model.items():
                        predictions = model[k](hidden_states)
                        scores.append(proba_projection(prediction.view(-1, 2))[:, -1].detach().cpu().numpy())
                else:          
                    scores.append(proba_projection(prediction.view(-1, 2))[:, -1].detach().cpu().numpy()) # I just learned how to spell detach
                labels.append(batch[main_target].detach().cpu().numpy())
    
    #TODO: remove this order check
    # write the order of participants seen in the test set to a file
    pd.DataFrame(participants).to_csv(os.path.join(args.output_dir, 'test_batch_order.csv'))
    
    labels=np.concatenate([l.ravel() for l in labels], axis=0)
    scores=np.concatenate([l.ravel() for l in scores], axis=0)
    
    write_results(args=args, scores=scores, dataloader=test_dataloader, istestset=True, participant_order=participants)#.index.get_level_values('participant_id')))

    if args.calculate_metrics:
        def get_metrics(labels, scores):
            
            AUC=sklearn.metrics.roc_auc_score(labels, scores.ravel())
            AUPR=sklearn.metrics.average_precision_score(labels, scores.ravel())

            print("AUC: ", AUC)
            print("AUPR: ", AUPR)



            # Get the thresholds
            precision, recall, thresholds = sklearn.metrics.precision_recall_curve(labels, scores.ravel())

            print(recall)

            # find the threshold where FPR == 2%
            print(thresholds.shape, recall.shape, len(scores), len(labels))
            threshold = thresholds[recall[1:] == np.min(recall[recall>0.98])] #[0]
            if len(threshold)>1:
                threshold=threshold[0]
            print(threshold)
            print(sklearn.metrics.classification_report(labels, scores.ravel()>threshold))
            precision, recall, thresholds = sklearn.metrics.precision_recall_curve(labels, scores.ravel())
        
            return AUC, AUPR, precision, recall
        
        if isinstance(scores, dict):
            dict_args=deepcopy(vars(args))
            for k, s in scores:
                AUC, AUPR, precision, recall = get_metrics(labels[k], v)
                dict_args[f'auc_{args.target}']=AUC
                dict_args[f'aupr_{args.target}']=AUPR
                dict_args[f'recall_{args.target}']=recall
                dict_args[f'precision_{args.target}']=precision
        else:
            AUC, AUPR, precision, recall = get_metrics(labels, scores)
            dict_args=deepcopy(vars(args))
            dict_args[f'auc_{args.target}']=AUC
            dict_args[f'aupr_{args.target}']=AUPR
            dict_args[f'recall_{args.target}']=recall
            dict_args[f'precision_{args.target}']=precision
    
    # ndarray not serializable error, may need to try using .tolist() for it?
    # Problem with that is tolist does not scale well for large ndarrays...
    #with open(os.path.join(args.output_dir, 'test_perf.pkl'), mode='w') as f:
    #    f.write(json.dumps(dict_args))
        
        
     
    # Getting no attribute ravel issue in plotting, commenting out for now. 
    # save the precision and recall curves
    #import matplotlib.pyplot as plt
    
    #fig=plt.figure()
    #precision, recall, thresholds = sklearn.metrics.precision_recall_curve(labels, scores.ravel())
    #plt.plot(recall, precision)
    #plt.ylim([0,1])
    #plt.xlim([0,1])
    #plt.ylabel('precision')
    #plt.xlabel('recall')
    #plt.title(f'AUPR = {AUPR:.3f}')
    
    #plt.savefig(os.path.join(args.output_dir,'aupr.png'))
    
    
    #fpr, tpr, _  =sklearn.metrics.roc_curve(labels, scores.ravel())
    
    #fig=plt.figure()
    #plt.plot(fpr, tpr)
    #plt.ylim([0,1])
    #plt.xlim([0,1])
    #plt.ylabel('tpr')
    #plt.xlabel('fpr')
    #plt.title(f'ROC = {AUC:.3f}')
    
#     write_results(args, scores, test_dataloader)
    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--target', type=str, default=('ili',), nargs='+', choices=['ili', 'ili_24', 'ili_48', 'covid', 'covid_24', 'covid_48', 'symptoms__fever__fever', 'flu_covid'], help='The target')
    parser.add_argument('--modeltype', type=str, default='grud', choices=['grud', 'linear', 'ar_mdl', 'lstm', 'cnn', 'prob_nn', 'xgboost', 'lancet_rhr','fastgam', 'gru', 'gru_simple'], help='The model to train.')

    parser.add_argument("--max_seq_len", type=int, default=48, help="maximum number of timepoints to feed into the model")
    parser.add_argument("--output_dir", type=str, required=True, help='save dir.')
    parser.add_argument('--data_dir', type=str, required=True, help='Explicit dataset path (else use rotation).')
    parser.add_argument("--checkpt_dir", type=str, default=None, help='checkpointing directory, required if script might be preempted.')
    parser.add_argument("--reload",  action='store_true', help='Option to load the latest model from the output_dir if it exists.')
    parser.add_argument("--test",  action='store_true', help='Option to load the latest model from the output_dir if it exists.')
    parser.add_argument("--small",  action='store_true', help='debug by stepping through just a few batches')

    parser.add_argument('--seed', type=int, default=0)
    
    parser.add_argument('--num_dataloader_workers', type=int, default=4, help='# dataloader workers.')
    parser.add_argument('--opt_level', type=str, default='O1', choices=['O0', 'O1'], help='The model to train.')

    parser.add_argument("--epochs", type=int, default=50, help="number of training epochs")
    parser.add_argument(
        '--batches_per_gradient', type=int, default = 1,
        help='Accumulate gradients over this many batches.'
        )
    parser.add_argument('--patience', type=int, default = 5, help='Early stopping patience')
    parser.add_argument("--batch_size", type=int, default=32, help="batch size for train, test, and eval")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate for the model")
    parser.add_argument("--grud_hidden", type=int, default=67, help="learning rate for the model")
    parser.add_argument("--regularly_sampled", action='store_true', help="Set this flag to have regularly sampled data rather than irregularly sampled data.")
    parser.add_argument("--zscore", action='store_true', help="Set this flag to train a model using the z score (assume forward fill imputation for missing data)")
    parser.add_argument("--only_healthy", action='store_true', help='Set this flag to train the model on only healthy measurements before the first onset of the target illness')
    parser.add_argument("--calculate_metrics", action='store_true', help='Calculate AUROC and other metrics using the scores calculated before writing them to a csv file.')
    parser.add_argument("--feat_subset", action='store_true', help='in the measurement and measurement_z dataframes only use the subset of features found to work better for xgboost and ridge regression') 
    parser.add_argument("--feat_regex", type=str, default="", help="A regex pattern to filter the features by. If a feature matches the pattern with a call to re.match then the feature will be used. If an empty string is given then all features will be used.")
    parser.add_argument("--num_feature", type=int, default=48, help="number of features passed to the model.") 
    parser.add_argument('--forecast_type', type=str, default='allevent', choices=[ 
        'allevent', 'timetoevent'], help='The target')
    parser.add_argument('--days_ago', type=int, default=7, help='Number of days in the past to include in feature set for XGBOOST')
    parser.add_argument("--unbalanced",  action='store_true', help='no task balancing in loss function') #todo add to all models.
    parser.add_argument("--weekofyear",  action='store_true', help='add week of year') #todo add to all models.
    parser.add_argument("--resample",  action='store_true', help='upsample the data to correct for the week-of-year and region bias. It is recommended to use in conjunction to correct for weekofyear feature.') #todo add to all models.
    parser.add_argument("--add_missingness",  action='store_true', help='add 50% corruption to testing data') #todo add to all models.
    parser.add_argument("--xgb_method", type=str, default='auto', choices=['auto', 'gpu_hist', 'hist'], help='The tree_method argument to be passed to XGBClassifier, only necessary if training the xgboost model. Useful if training with GPUs')
    parser.add_argument("--no_imputation", action='store_true', help="If set use the data prior to imputation preserving missing values as NA.")
    parser.add_argument("--max_miss", type=restricted_float, default=1, help="A value between 0 and 1, which restricts all observations for the model to be trained on have at most that percentage of activity data can be missing values on the current day and the past 7 days inclusive.")
#TODO: test the three features directly below.
    parser.add_argument("--positive_labels", default=None, nargs='+', type=int, help='Pass in a tuple of integers to indicate which dates around a given event onset should stay positive labels in the training data, for example 0 will keep the date of onset as 1, whereas (0, -1, 1) will keep the day before the day of and the day after as 1, in both cases all other labels will be zero. If None no changes to the labels will take place.')
    parser.add_argument("--mask_labels", default=None, nargs='+', type=int, help='Pass in a tuple of intergers to indicate which dates around a given event onset should be dropped from the training data. For example 0 will mask the date of onset, whereas (-3, -2, 1) will mask the third day before onset, the second day before onset and the day after.')
    parser.add_argument("--bound_labels", default=None, nargs=2, type=int, help='Pass in two numbers, which indicate the bounds relative to a date of onset to keep in the training data, for example argument is -21 7 then it will keep dates between 22 days before onset and 8 days after onset, i.e. it is a closed set of days. Note, both could be positive or negative in which case the date of onset would not be included.')
    parser.add_argument("--fake_data", action='store_true', help='pass this argument if you are fitting the model with the fake data, simulated signal, in the fake data folder.') 
    # add dataset splitting functions to here:
    parser.add_argument("--ignore_participants", action='store_true', help='Just do regular training without splitting participants into differnt test train groups (i.e. only split with time)') 
    parser.add_argument("--train_start_date",  type=str, default='', help='start date for training data (yyyy-mm-dd)')
    parser.add_argument("--train_end_date",  type=str, default='', help='The last day of training data (inclusive)')
    parser.add_argument("--validation_start_date",  type=str, default='', help='start date for training data')
    parser.add_argument("--validation_end_date",  type=str, default='', help='The last day of validation data (inclusive)')
    parser.add_argument("--validation_set_len",  type=int, help='Alternatively provide the len of the desired validation set')
    parser.add_argument("--test_start_date",  type=str, default='', help='start date for test data')
    parser.add_argument("--test_end_date",  type=str, default='', help='The last day of test data (inclusive)')    
    parser.add_argument("--temporally_split_participants", action='store_true', help="If set split the participants temporally according to the train, validation, and test dates provided. Otherwise split the participants randomly.") 

    args = parser.parse_args()
    
    if args.opt_level=='O0':
        amp=None
        
    # Raise error if unimplemented combinations of arguments are passed.
    if args.zscore and args.no_imputation:
        print('combination of --zscore and --no_imputation has not been implemented.')
        raise NotImplementedError
    
    # Raise error if no checkpoint directory given and the modeltype is GRUD.
    assert not (not args.checkpt_dir and args.modeltype == 'grud'), 'If the modeltype is grud, then a checkpt_dir must be provided for checkpointing.'
    if args.checkpt_dir:
        if not os.path.exists(args.checkpt_dir):
            os.mkdir(args.checkpt_dir)
        
    # Check bound_labels is a non empty bound of days if provided.
    if not (args.bound_labels is None):
        assert len(args.bound_labels) == 2, 'If label_bounds is provided it must be only 2 values, a lower bound and an upper bound'
        assert args.bound_labels[0] <= args.bound_labels[1], 'Lower bound is greater than the upper bound, must be a non-empty set of days.'
    if not (args.bound_labels is None) and not (args.positive_labels is None):
        # This assertion 
        assert (min(args.positive_labels) >= min(args.bound_labels)) and (max(args.positive_labels) <= max(args.bound_labels)), 'positive labels must be contained within the bound labels if both are provided'
    # todo reload args
    
    print(vars(args))
    if args.test:
        if args.modeltype in ['grud', 'gru', 'lstm']:
            if not(os.path.exists(os.path.join(args.output_dir, 'checkpoint_best.pth'))):
                if not(os.path.exists(os.path.join(args.checkpt_dir, 'checkpoint_best.pth'))):
                    print('No checkpoint found, training from scratch')
                    args.test=False
#         elif args.modeltype in ['xgboost']:
#             assert os.path.exists(os.path.join(args.output_dir, 'xgboost_model.bin')), os.path.join(args.output_dir, 'xgboost_model.bin')
        args.reload=True
    if not (os.path.exists(args.output_dir)):
        os.mkdir(args.output_dir)
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
              f.write(json.dumps(vars(args)))
            
    # change all of the args dates into pandas datetimes.
    # train
    args.train_start_date= pd.to_datetime(args.train_start_date) if args.train_start_date!='' else None
    args.train_end_date= pd.to_datetime(args.train_end_date) if args.train_end_date!='' else None
#     print(args.train_start_date)
#     print(not(args.train_start_date is None))
    if not(args.train_start_date is None) and not(args.train_end_date is None):
        assert args.train_end_date>=args.train_start_date, "The train end date must be after the train start date"
        
        
    # validation
    args.validation_start_date= pd.to_datetime(args.validation_start_date) if args.validation_start_date!='' else None
    args.validation_end_date= pd.to_datetime(args.validation_end_date) if args.validation_end_date!='' else None
    if hasattr(args,'validation_set_len'):
        if args.validation_set_len is not None:
            assert args.validation_end_date is None, "You cannot define both the validation_end_date and the validation_set_len"
    
    if args.validation_start_date is not None:
        assert (hasattr(args,'validation_set_len') or args.validtion_end_date!=''), "Ether validation_set_len or validation_end_date must be defined if you define validation_start_date"
        
    if not(args.validation_start_date is None):
        args.validation_end_date = args.validation_start_date+timedelta(args.validation_set_len) if (args.validation_set_len is not None) else pd.to_datetime(args.validation_end_date)
        assert args.validation_end_date>=args.validation_start_date, "The validation_end_date must be after the validation_start_date"
        
    # if training dates are defined, we should also define the validation dates.
#     if args.train_end_date is not None:
#         assert args.validation_start_date is not None, "If training set is defined, you must also define a validation set."
        
    # test
    args.test_start_date= pd.to_datetime(args.test_start_date) if args.test_start_date!='' else None
    args.test_end_date= pd.to_datetime(args.test_end_date) if args.test_end_date!='' else None
    if not(args.test_start_date is None) and not(args.test_end_date is None):
        assert args.test_end_date>=args.test_start_date, "The test_end_date must be after the test_start_date"    
    
    try:
        main(args)
    except Exception as e:
        with open(os.path.join(args.output_dir, 'failed.txt'), 'w') as f:
              f.write('Marked as failure. Check Logs.')
        print(e)
        traceback.print_exc()

