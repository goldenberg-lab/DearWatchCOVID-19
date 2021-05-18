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

from prospective_set_up import get_retro, get_prosp
# import pdb

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


def load_data(data_dir, regular=True, load_baseline=True, load_activity=True, load_survey=True, fname=None, all_survey=True):
    """
    """
    if fname is None:
        f_pattern = '_daily_data_' + 'allsurvey_'*args.all_survey + 'regular'*regular +'irregular'*(not(regular)) + '.hdf' 
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
    if load_baseline: dfs['baseline']=pd.read_hdf(fname, 'baseline')
    if load_activity: dfs['activity']=pd.read_hdf(fname, 'activity')
    if load_survey: dfs['survey']=pd.read_hdf(fname, 'survey')

    return dfs


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
    wpath = os.path.join(args.output_dir, args.target[0][1] + csv_suffix)
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
            if isinstance(p[0], list)|isinstance(p[0], tuple):
                p, _ = zip(*p)
                p2=[]
                for item in p:
                    if item not in p2:
                        p2.append(item)
                p=p2
        else:
            p = participant_order
        tmp[args.target[0][1]] = dataloader.dataset.outcomes[args.target[0]].sort_index().reindex(p, level='participant_id')
        
        scores=np.concatenate([l.ravel() for l in scores], axis=0)
        tmp[args.target[0][1] + '_score'] = scores
        
        # get the thresholds (or apply them
        if istestset:
            #load and check thresholds
            if os.path.exists( os.path.join( args.output_dir, 'thresholds_survey.json')):
                with open(os.path.join( args.output_dir, 'thresholds_survey.json')) as f:
                    threshold = json.load(f)
                # apply this to the data and get the scores.
                for k, v in threshold.items():
                    print(args.target[0][1])
                    tmp[args.target[0][1]+'_pred_'+k] = np.asarray(scores >= float(v) ).astype(np.int32)
                    
        else:
            # if this is validation set, and valid start/valid end date are defined, we must subselect those dates.
            if args.validation_start_date  is not None:
                tmp = tmp.loc[tmp.index.get_level_values('date')>=args.validation_start_date , :]
            if args.validation_end_date is not None:
                tmp = tmp.loc[tmp.index.get_level_values('date')<=args.validation_end_date, :]
                
            threshold={}
            fpr, tpr, thresholds = sklearn.metrics.roc_curve( tmp[args.target[0][1]], tmp[args.target[0][1]+'_score'])

            # 98% specificity
            specificity=1-fpr
            target_specificity = min(specificity[specificity>=0.98])
            # index of target fpr
            index = list(specificity).index(target_specificity)

            threshold['98_spec'] = thresholds[index]

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
            with open(os.path.join(args.output_dir, 'thresholds_survey.json'), 'w') as f:
                f.write(json.dumps({k:str(v) for k,v in threshold.items()}))
                
            for k, v in threshold.items():
                tmp[args.target[0][1]+'_pred_'+k] = np.asarray(scores >= float(v) ).astype(np.int32)
            
        pd.DataFrame(tmp).to_csv(wpath)
    else:
        tmp = dataloader.dataset.outcomes[list(args.target)].copy(deep=True).sort_index()
        for k in scores.keys():
            tmp.insert(len(tmp.columns), k + '_score', scores[k])
            
            # get/load thresholds
            if istestset:
                #load and check thresholds
                if os.path.exists( os.path.join( args.output_dir, k+'thresholds_survey.json')):
                    with open(os.path.join( args.output_dir, k+'thresholds_survey.json')) as f:
                        threshold = json.load(f)
                    # apply this to the data and get the scores.
                    for k2, v in threshold.items():
                        tmp[k+'_pred_'+k2] = np.asarray(scores >= float(v) ).astype(np.int32)

            else:
                # if this is validation set, and valid start/valid end date are defined, we must subselect those dates.
                if args.validation_start_date  is not None:
                    tmp = tmp.loc[tmp.index.get_level_values('date')>=args.validation_start_date , :]
                if args.validation_end_date is not None:
                    tmp = tmp.loc[tmp.index.get_level_values('date')<=args.validation_end_date, :]
                threshold={}
                fpr, tpr, thresholds = sklearn.metrics.roc_curve( tmp[k].values, tmp[k+'_score'].values)

                # 98% specificity
                specificity=1-fpr
                target_specificity = min(specificity[specificity>=0.98])
                # index of target fpr
                index = list(specificity).index(target_specificity)

                threshold['98_spec'] = thresholds[index]

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
                with open(os.path.join(args.output_dir, k+'thresholds_survey.json'), 'w') as f:
                    f.write(json.dumps({k2:str(v) for k2,v in threshold.items()}))

                for k2, v in threshold.items():
                    tmp[k+'_pred_'+k2] = np.asarray(scores >= float(v) ).astype(np.int32)
        tmp.to_csv(wpath)
        
        
def resample_fairly(participants, dfs, sample_col='ili'):
    """
    increase the occurence of participants based on their 
    incidenc, week, and region.
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
        self.survey=dfs['survey']# todo get feature set
        self.statics = dfs['baseline']
        self.outcomes = dfs['survey']
        self.feat_subset = feat_subset
        
        self.subset =  SURVEY_ROLLING_FEATURES+SURVEY_BASELINE_FEATURES+['weekofyear']*args.weekofyear
        
        assert len(self.subset)==args.num_feature+1*args.weekofyear, print(len(self.subset), args.num_feature)
        for item in self.subset:
            if item not in dfs['survey'].columns.get_level_values(level=1):
                print(item, " is missing")
        assert all([feat in dfs['survey'].columns.get_level_values(level=1) for feat in self.subset])
        
        
        self.outcomes.loc[:, ('measurement', 'ili')]=self.outcomes.loc[:, ('measurement', 'ili')].apply(int).values
        self.outcomes.loc[:, ('measurement', 'ili_24')]=self.outcomes.loc[:, ('measurement', 'ili_24')].apply(int).values
        self.outcomes.loc[:, ('measurement', 'ili_48')]=self.outcomes.loc[:, ('measurement', 'ili_48')].apply(int).values
        
        self.outcomes.loc[:, ('measurement', 'covid')]=self.outcomes.loc[:, ('measurement', 'covid')].apply(int).values
        self.outcomes.loc[:, ('measurement', 'covid_24')]=self.outcomes.loc[:, ('measurement', 'covid_24')].apply(int).values
        self.outcomes.loc[:, ('measurement', 'covid_48')]=self.outcomes.loc[:, ('measurement', 'covid_48')].apply(int).values
        self.outcomes.loc[:, ('measurement', 'symptoms__fever__fever')]=self.outcomes.loc[:, ('measurement', 'symptoms__fever__fever')].apply(int).values

        # Using np.unique returning the indices to preserve the order the participants are in the original 
        # self.participants=list(np.unique(dfs['baseline'].index.get_level_values('participant_id')))
        #self.participants=np.unique(dfs['baseline'].index.get_level_values('participant_id')).tolist()
              
        if participants is None:
            # only resample for training data
            self.participants = list(np.unique(dfs['survey'].index.get_level_values('participant_id')))
        else:
            self.participants = participants
        #assert len(self.participants)>0, print("There are no participants in this set")
        
        self.full_sequence=full_sequence
        
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
        time_df = self.survey.loc[participant, idx['time', :]].values.astype(np.int32)
        if self.feat_subset:
            multiindex_rolling_features= pd.MultiIndex.from_product((['measurement'], SURVEY_ROLLING_FEATURES+SURVEY_BASELINE_FEATURES+['weekofyear']*args.weekofyear))
#             measurement_df = self.survey.loc[idx[participant, :], 
#                                     idx['measurement', self.subset]].values
#             mask_df =self.survey.loc[idx[participant, :], idx['mask', self.subset]].values.astype(np.int32)
        
            measurement_df = self.survey.loc[participant, 
                                    multiindex_rolling_features].values.astype(float)
            mask_df =self.survey.loc[participant, multiindex_rolling_features].values.astype(np.int32)
        else:
            raise Exception("This currently is not set up")
            measurement_df = self.survey.loc[participant, 
                    idx['measurement', :]].values.astype(float)
            mask_df =self.survey.loc[participant, idx['mask', ]].values.astype(np.int32)
        
        

        # get the outcomes
        ili_outcomes=self.outcomes.loc[participant, ('measurement', 'ili')]
        ili_24_outcomes=self.outcomes.loc[participant, ('measurement', 'ili_24')]
        ili_48_outcomes=self.outcomes.loc[participant, ('measurement', 'ili_48')]
        
        covid_outcomes=self.outcomes.loc[participant, ('measurement', 'covid')]
        covid_24_outcomes=self.outcomes.loc[participant, ('measurement', 'covid_24')]
        covid_48_outcomes=self.outcomes.loc[participant, ('measurement', 'covid_48')]
        fever_outcomes=self.outcomes.loc[participant, ('measurement', 'symptoms__fever__fever')]

        return_dict = {'measurement':measurement_df, 'mask':mask_df,  'time':time_df, 'ili':ili_outcomes, 'ili_24': ili_24_outcomes, 'ili_48':ili_48_outcomes, 'covid':covid_outcomes, 'covid_24':covid_24_outcomes, 'covid_48': covid_48_outcomes, 'symptoms__fever__fever':fever_outcomes, 'obs_mask':np.ones(len(measurement_df))}
        
        assert sum(return_dict['obs_mask'])>0
        
        if self.full_sequence:
#             for k, v in return_dict.items():
#                 print(k, v.dtype)
            return {k:torch.tensor(v).float() for k,v in return_dict.items()}, participant
        
        
        # pad them to the maximum sequence length
        if len(time_df)>self.max_seq_len:
            # make sure 50% of the time there is a 1 in the data...
            # find minimum index of 1 in labels
            # min_index= list(return_dict['ili']).index(1) if sum(return_dict['ili'])>0 else 0
            # random_val= random.random() if sum(return_dict['ili'])>0 else 1 # this is mostly not the case, but to be sure we can remove patients without ILI in the start.
            
            random_start = random.randint(0, len(time_df)-self.max_seq_len)
            # sample and crop
            max_time = min(random_start+self.max_seq_len, len(return_dict['ili']))
            return_dict={k:v[random_start:max_time] for k,v in return_dict.items()}
            
            
#             if random_val<0.5:
#                 # includes a 1
#                 assert max(0, min_index-self.max_seq_len-1)<(len(time_df)-self.max_seq_len), (max(0, min_index-self.max_seq_len-1),(len(time_df)-self.max_seq_len))
#                 random_start = random.randint(max(0, min_index-self.max_seq_len-1), len(time_df)-self.max_seq_len)
#                 # sample and crop
#                 max_time = min(random_start+self.max_seq_len, len(return_dict['ili']))
#                 return_dict={k:v[random_start:max_time] for k,v in return_dict.items()}
                
#             else:
#                 # does not include a 1
#                 random_start = random.randint(0, min_index-1) if (min_index-1 )>0 else 0
#                 # sample and crop
#                 max_time = min(random_start+self.max_seq_len, len(return_dict['ili']))
#                 return_dict={k:v[random_start:max_time] for k,v in return_dict.items()}
            
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

            
def apply_standard_scaler(dfs, scaler=None):
    """
    apply sklearn standard scaler
    """
    idx=pd.IndexSlice
    multiindex_rolling_features= pd.MultiIndex.from_product((['measurement'], SURVEY_ROLLING_FEATURES+SURVEY_BASELINE_FEATURES+['weekofyear']*args.weekofyear))
    
#     for item in multiindex_rolling_features:
#         print(item)
        
    if scaler is not None:
        assert isinstance(scaler, dict), print('Expected a dict for arguement scaler')
    else:
#         print(set(sorted(dfs['survey'].loc[:, multiindex_rolling_features].columns.tolist()))-set(dfs['survey'].loc[:, multiindex_rolling_features].sample(5000).mean().index.tolist()))
        scaler={'mean': dfs['survey'].loc[:, multiindex_rolling_features].sample(5000).mean().values,
                'std': dfs['survey'].loc[:, multiindex_rolling_features].sample(5000).std().values,
               }
    # apply to measurement column in X and X_ffilled  
    
    print("There are this many zeros in the scaler:", np.sum(scaler['std']==0))
    scaler['std'][scaler['std']==0] = 1 # these columns do not get scaled.
    
    
    
    
    
    assert scaler['mean'][np.newaxis, :].shape[1:] == scaler['std'][np.newaxis, :].shape[1:], print(scaler['mean'][np.newaxis, :].shape, scaler['std'][np.newaxis, :].shape[1:])
    assert dfs['survey'].loc[:, multiindex_rolling_features].values.shape[1:] == scaler['std'][np.newaxis, :].shape[1:], print(dfs['survey'].loc[:, multiindex_rolling_features].values.shape, scaler['std'][np.newaxis, :].shape) 
    
    dfs['survey'].loc[:, multiindex_rolling_features] = (dfs['survey'].loc[:, multiindex_rolling_features].values - scaler['mean'][np.newaxis, :])/scaler['std'][np.newaxis, :]
                                                                        
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
                                   
    
def create_lagged_data(dataloader, args, use_features=None, target=['ili']):
    """
    Inputs:
        dataloader (torch.utils.dataset.DataLoader): An instance of wrapped ILIDataset.
        forecast_features (list): a list of column headings to restrict features to.
        target: the objective feature.
    Returns:
        X (np.array): ??
        y (np.array): ??
    """
    idx=pd.IndexSlice
    
    # ILIDataset has attributes self.numerics, self.statics, self.outcomes, self.participants
    measurement_col='measurement_z' if args.zscore else 'measurement' 
    forecast_features = ['time'] + [str(q) + 'days_ago' for q in range(1, args.days_ago + 1)]
    print(dataloader.dataset.numerics.head())
    if use_features is None:
        use_features =      dataloader.dataset.numerics.columns.get_level_values('value').tolist()
    label_colz = ['ili', 'ili_24', 'ili_48', 'covid', 'covid_24', 'covid_48', 'symptoms__fever__fever']
    target_col = ('label', target[0])
    # add measurement col?
    forecast_features += [measurement_col]  # add today's measurements as part of feature set
        
    
    ### Load survey dataframe (for ILI labels)
    print('Loading survey data ...')
    survey = dataloader.dataset.outcomes.copy()
    survey.columns = pd.MultiIndex.from_product([[target_col[0]], survey.columns]) # add multi-index columns to survey
    print('survey shape: %s' % (survey.shape,))
    
    ### Create day-shifted features - for a subset of 20 features
    #   print(len(use_features))
    print('Using cores=', args.num_dataloader_workers)
    print('Creating day-shifted dataset ...')
    p = Pool(args.num_dataloader_workers)
    out = pd.concat(p.starmap(add_shifts,
                              zip(dataloader.dataset.numerics.loc[:, idx[measurement_col, use_features]].groupby('participant_id'),
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
    # num levels might be more than one.
    
    df=dfs['survey'].copy()
    
    if 'ili' in df.columns:
        ili_col = 'ili'
    else:
        ili_col= ('measurement', 'ili')

    df['ili_count'] = df[ili_col].groupby('participant_id').apply(lambda x:x.rolling(2).apply(lambda y:y[0]<y[-1], raw=True).cumsum())

    df['ili_count'] = df['ili_count'].fillna(0)
    tmp_df = df[ili_col].groupby('participant_id').first()
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


def main(args):
    """
    """
    import time
    
    print("Loading data.")
    # check if the datasets are already made
    time_old=time.time()
    
    ############ temp
    # Try to read in the dictionary detailing the retrospective data
#     d_name = 'split_dict_' + args.regular_sampling*'regular' + (not args.regular_sampling)*'irregular' + '.pkl'
#     if not(args.regular_sampling) & args.override_splits:
#         d_name = 'split_dict_regular.pkl'

#     with open(os.path.join(args.home_dir, d_name), 'rb') as f:
#         retro_specs = pickle.load(f)

#     with open(os.path.join(args.home_dir, 'test', d_name), 'rb') as f:
#         prosp_specs = pickle.load(f)
#     # If the load hasn't failed we know that the dictionary file exists, check for a full_dataframe file, if both exist raise an error because it's unclear what the program should have done.
#     f_name = 'split_daily_data_' + args.regular_sampling*'regular' + (not args.regular_sampling)*'irregular' + '.hdf'
#     assert not os.path.exists(os.path.join(args.home_dir, f_name)), 'Found a retrospective dictionary file and full dataframe file in {}, uncertain which should be used.'.format(args.data_dir)
#     dfs=load_data(args.data_dir, regular=args.regular_sampling,  load_activity=False, all_survey=True, fname='all_daily_data_allsurvey_irregular_merged_nov29.hdf')

#     test_dfs = get_prosp(dfs, prosp_specs)
#     print(sorted(list(set(test_dfs['survey'].index.get_level_values('participant_id'))))[:10])
#     dfs = get_retro(dfs, retro_specs)
#     print("loaded correct splits")
    ############ temp
    try:
        # Try to read in the dictionary detailing the retrospective data
        d_name = 'split_dict_' + args.regular_sampling*'regular' + (not args.regular_sampling)*'irregular' + '.pkl'
        if not(args.regular_sampling)&args.override_splits:
            d_name = 'split_dict_regular.pkl'
        with open(os.path.join(args.home_dir, d_name), 'rb') as f:
            retro_specs = pickle.load(f)
            
        with open(os.path.join(args.home_dir, 'test', d_name), 'rb') as f:
            prosp_specs = pickle.load(f)
        # If the load hasn't failed we know that the dictionary file exists, check for a full_dataframe file, if both exist raise an error because it's unclear what the program should have done.
        f_name = 'split_daily_data_' + args.regular_sampling*'regular' + (not args.regular_sampling)*'irregular' + '.hdf'
        assert not os.path.exists(os.path.join(args.home_dir, f_name)), 'Found a retrospective dictionary file and full dataframe file in {}, uncertain which should be used.'.format(args.data_dir)
        dfs=load_data(args.data_dir, regular=args.regular_sampling,  load_activity=False, all_survey=True, fname='all_daily_data_allsurvey_irregular_merged_nov29.hdf')
        
        test_dfs = get_prosp(dfs, prosp_specs)
        dfs = get_retro(dfs, retro_specs)
        print("loaded correct splits")
        # create a flag to indicate if a prospective test set was successfully loaded
        prosp_test = True
    except OSError:
        # Note: if here then it must have failed on either line 764 or line 769, but if it didn't fail on line 764 then the assert on line 768 must stop it in the case that this would work, preventing a situation where it failed to load the full dataset when the dictionary existed but managed to continue on with a truncated file which may be incorrect. 
        dfs=load_data(args.data_dir, regular=args.regular_sampling,  load_activity=False, all_survey=True, fname='all_daily_data_allsurvey_irregular_merged_nov29.hdf')
        test_dfs = dfs.copy()
        # create a flag to indicate if a prospective test set was successfully loaded
        prosp_test = False
    if args.weekofyear:
        idx=pd.IndexSlice
        dfs['survey']['weekday']=(dfs['survey'].index.get_level_values('date').weekday<5).astype(np.int32) # add week of year
        dfs['survey'].loc[:, 'weekofyear']=dfs['survey'].index.get_level_values('date').isocalendar().week .astype(np.int32).values
        test_dfs['survey']['weekday']=(test_dfs['survey'].index.get_level_values('date').weekday<5).astype(np.int32) # add week of year
        test_dfs['survey'].loc[:, 'weekofyear']=test_dfs['survey'].index.get_level_values('date').isocalendar().week .astype(np.int32).values

        
#         dfs['survey'].loc[:, idx['measurement', 'weekofyear']]=dfs['survey'].index.get_level_values('date').isocalendar().week .astype(np.int32).values
        
        
#         dfs['survey'].loc[:, idx['measurement_z', 'weekofyear']]=dfs['survey'].index.get_level_values('date').isocalendar().week.astype(np.int32).values
#         dfs['survey'].loc[:, idx['mask', 'weekofyear']]=np.ones(len(dfs['activity'])).astype(np.int32)
#         dfs['survey'].loc[:, idx['time', 'weekofyear']]=np.ones(len(dfs['activity'])).astype(np.int32)

    try:
        dfs['survey']=dfs['survey'].join(dfs['baseline'][SURVEY_BASELINE_FEATURES], on='participant_id', how='left')
        test_dfs['survey']=test_dfs['survey'].join(test_dfs['baseline'][SURVEY_BASELINE_FEATURES], on='participant_id', how='left')
        
        
        args.target = [('measurement', t) for t in args.target]
    except:
        for col in SURVEY_BASELINE_FEATURES:
            if col in dfs['baseline'].columns:
                print(col)
            else:
                print("lacking: ", col) 
        print(dfs['baseline'].columns.tolist())
        raise
    dfs['survey']=dfs['survey'].fillna(0)
    test_dfs['survey']=test_dfs['survey'].fillna(0)
    
    idx=pd.IndexSlice
    
    if args.last_7:        
        dfs['survey'][SURVEY_ROLLING_FEATURES] = dfs['survey'][SURVEY_ROLLING_FEATURES].reset_index().groupby('participant_id').rolling('7D', min_periods=1, on='date').max().reset_index().set_index(['participant_id','date'])[SURVEY_ROLLING_FEATURES]
        test_dfs['survey'][SURVEY_ROLLING_FEATURES] = test_dfs['survey'][SURVEY_ROLLING_FEATURES].reset_index().groupby('participant_id').rolling('7D', min_periods=1, on='date').max().reset_index().set_index(['participant_id','date'])[SURVEY_ROLLING_FEATURES]

    dfs['survey'] = pd.concat({'measurement':dfs['survey'],  'mask':dfs['survey'].notna().apply(pd.to_numeric)
, 'time':dfs['survey'].isna().apply(pd.to_numeric)
}, axis=1, names=['df_type', 'value'])
    test_dfs['survey'] = pd.concat({'measurement':test_dfs['survey'], 'mask':test_dfs['survey'].notna().apply(pd.to_numeric)
, 'time':test_dfs['survey'].isna().apply(pd.to_numeric)
}, axis=1, names=['df_type', 'value'])
    
    print(dfs['survey'].loc[:, idx['mask',:]].mean().mean())
    assert dfs['survey'].loc[:, idx['mask',:]].mean().mean()==1, "values are missing!"

        
    print("Done loading data.", time.time() - time_old)
    
    
#     print(os.path.join(os.path.split(args.output_dir)[0], 'out_xgb_WOY_z'+'_reg_mm1_bounded'if (args.override_splits|args.regular_sampling) else '_mm1_bounded', f'train_participants.csv'))
#     print(os.path.join(os.path.split(args.output_dir)[0], 'out_xgb_WOY_z'+'_reg_mm1_bounded'if (args.override_splits|args.regular_sampling) else '_mm1_bounded', 'valid_participants.csv'))

    print(os.path.join(os.path.split(args.output_dir)[0], 'out_grud_WOY', f'train_participants.csv'))
    print(os.path.join(os.path.split(args.output_dir)[0], 'out_grud_WOY', 'valid_participants.csv'))    
    
    # load or make participants
    if args.ignore_participants:
        # use everybody
        participants = list(np.unique(dfs['baseline'].index.get_level_values('participant_id')))
        # make sure these participants are also in activity
        participants = list(set(participants).intersection(set(dfs['survey'].index.get_level_values('participant_id'))))
        train_participants=participants.copy()
        valid_participants=participants.copy()
        test_participants=participants.copy()
    elif os.path.exists(os.path.join(args.output_dir, f'train_participants.csv')) & os.path.exists(os.path.join(args.output_dir, 'valid_participants.csv')):
        # we want this for training and validation. however, the test participants will be the retrospective test participants.
        print('Pre-loading train/val/test IDs')
        train_participants = pd.read_csv(os.path.join(args.output_dir,'train_participants.csv')).values.ravel().tolist()
        valid_participants = pd.read_csv(os.path.join(args.output_dir,'valid_participants.csv')).values.ravel().tolist()
        test_participants = pd.read_csv(os.path.join(args.output_dir,'test_participants.csv')).values.ravel().tolist()
        
    elif os.path.exists(os.path.join(os.path.split(args.output_dir)[0], 'out_grud_WOY', f'train_participants.csv')) & os.path.exists(os.path.join(os.path.split(args.output_dir)[0], 'out_grud_WOY', 'valid_participants.csv')):
        # we will copy over participants from the other folders that already exist to make sure there is overlap with these train and validation participants.
        print('Pre-loading train/val/test IDs from ', os.path.join(os.path.split(args.output_dir)[0], 'out_grud_WOY'))
        train_participants = pd.read_csv(os.path.join(os.path.split(args.output_dir)[0], 'out_grud_WOY','train_participants.csv')).values.ravel().tolist()
        valid_participants = pd.read_csv(os.path.join(os.path.split(args.output_dir)[0], 'out_grud_WOY','valid_participants.csv')).values.ravel().tolist()
        try:
            #do
            test_participants = list(set(test_dfs['survey'].index.get_level_values('participant_id').tolist()))
            print("loaded test participants from test_dfs")
        except:
            test_participants = pd.read_csv(os.path.join(os.path.split(args.output_dir)[0], 'out_grud_WOY','test_participants.csv')).values.ravel().tolist()
            print('WARNING: loaded test_participants from ', os.path.join(os.path.split(args.output_dir)[0], 'out_grud_WOY','test_participants.csv'))
            
        

        
    else:
        print('Generating train/val/test IDs from scratch')
        # set random_seed
        random.seed(args.seed)

        # apply train val test splits
        time_old=time.time()
        #participants = list(set(dfs['baseline'].index.get_level_values('participant_id')))
        participants = list(np.unique(dfs['baseline'].index.get_level_values('participant_id')))
        # make sure these participants are also in activity
        participants = list(set(participants).intersection(set(dfs['survey'].index.get_level_values('participant_id'))))
        if args.train_end_date is None:
            
            print('Guaranteeing positive and negative cases for label ' + args.target[0][1])
            print(dfs['survey'].columns.tolist())
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

        pd.DataFrame(train_participants).to_csv(os.path.join(args.output_dir,'train_participants.csv'), index=False)
        pd.DataFrame(valid_participants).to_csv(os.path.join(args.output_dir,'valid_participants.csv'), index=False)
        pd.DataFrame(test_participants).to_csv(os.path.join(args.output_dir,'test_participants.csv'), index=False)
            
        print("Done creating splits.", time.time() - time_old)
    print(sorted(test_participants)[:10])

    print("Making dataloaders")
    time_old=time.time()
    idx=pd.IndexSlice
    print("test_participants", len(test_participants))
    # subset the dataframes to their respective participants
    if args.ignore_participants:
        train_dfs = {k:v.loc[v.index.get_level_values('participant_id').isin(train_participants), :] for k, v in dfs.items()}
    else:
        train_dfs = {k:v.loc[~v.index.get_level_values('participant_id').isin(valid_participants+test_participants), :] for k, v in dfs.items()}
    print('before train ', len(train_participants))
    train_participants = [p for p in train_participants if p in train_dfs['survey'].index.get_level_values('participant_id')]
    print('after train ', len(train_participants))
    
    valid_dfs = {k:v.loc[v.index.get_level_values('participant_id').isin(valid_participants), :] for k, v in dfs.items()}
    print('before valid ', len(valid_participants))
    valid_participants = [p for p in valid_participants if p in valid_dfs['survey'].index.get_level_values('participant_id')] # this is necessary because splits are based off of baseline dfs
    assert len(set(valid_participants)-set(valid_dfs['survey'].index.get_level_values('participant_id')))==0
    assert len(set(valid_dfs['survey'].index.get_level_values('participant_id'))-set(valid_participants))==0
    print('after valid ', len(valid_participants))
    if prosp_test:
        # If test_dfs exists already then a prospective test set was successfully loaded
        test_participants = np.unique(test_dfs['survey'].index.get_level_values('participant_id')).tolist()
    else:
        # In this case use a retrospective test set.
        test_dfs = {k:v.loc[v.index.get_level_values('participant_id').isin(test_participants), :] for k, v in dfs.items()}
        test_participants = [p for p in test_participants if p in test_dfs['survey'].index.get_level_values('participant_id')] # this is necessary because splits are based off of baseline dfs
    
    if args.add_missingness:
        # add missingness to entire row.
        test_dfs['numerics'].loc[:, idx['mask', :]]=test_dfs['numerics'].loc[:, idx['mask', :]].values*(np.random.randint(0, 2, len(test_dfs['numerics']))[:, np.newaxis])
    
    print(args.train_end_date, args.validation_end_date, args.test_end_date)
    
    # if train dates are given, restrict data further into the training dates
    if args.train_end_date is not None:
        print(len(train_dfs['survey']))
        train_dfs={k:(v.loc[v.index.get_level_values('date')<=args.train_end_date] if 'date' in v.index.names else v) for k, v in dfs.items()}
        print(len(train_dfs['survey']))
        
    if args.validation_end_date is not None:
        print(args.validation_end_date)
        print(len(valid_dfs['survey']))
        valid_dfs={k:(v.loc[(v.index.get_level_values('date')<=args.validation_end_date)] if 'date' in v.index.names else v) for k, v in dfs.items()}
        print(len(valid_dfs['survey']))
        
    if args.test_end_date is not None:
        print(len(test_dfs['survey']))
        test_dfs={k:(v.loc[(v.index.get_level_values('date')<=args.test_end_date)] if 'date' in v.index.names else v) for k, v in dfs.items()}
        print(len(test_dfs['survey']))

    
    
    # for survey, we want to train on only the ill participants.
#     train_dfs={k:v.loc[np.max(v['ili', 'ili_24', 'ili_48'].values ,axis=1)]  if 'ili' in v.columns else v for k,v in train_dfs.items()}
    orig_len = len(train_dfs['survey'])
#     print(train_dfs['survey'].head())
    print(('measurement', 'ili')in train_dfs['survey'].columns)
    print(train_dfs['survey'][('measurement', 'ili')].mean())
    train_dfs={k:v.loc[v[('measurement', 'ili')]==1]  if ('measurement', 'ili') in v.columns else v for k,v in train_dfs.items()}
#     train_dfs={k:v.loc[v['ili']==1]  if 'ili' in v.columns else v for k,v in train_dfs.items()}
    print(train_dfs['survey'][('measurement', 'ili')].mean())
    assert len(train_dfs['survey']) < orig_len
    valid_dfs={k:v.loc[v[('measurement', 'ili')]==1]  if ('measurement', 'ili') in v.columns else v for k,v in valid_dfs.items()}
    
    train_participants = [p for p in train_participants if p in train_dfs['survey'].index.get_level_values('participant_id')] # there were participants without ili?
    valid_participants = [p for p in valid_participants if p in valid_dfs['survey'].index.get_level_values('participant_id')] # there were participants without ili?
    

    # assert the target is in all of the train val and test sets
    print(train_dfs['survey'].head())
    assert train_dfs['survey'][args.target[0]].values.sum() >0, "The target does not exist in the train set." 
    assert valid_dfs['survey'][args.target[0]].values.sum() >0, "The target does not exist in the validation set." 
    assert test_dfs['survey'][args.target[0]].values.sum() >0, "The target does not exist in the test set." 
        
    if not(args.zscore):
        # zscore is already normalised, so we do not need to apply the scalar
        print('train scaler')
        train_dfs, scaler = apply_standard_scaler(train_dfs, scaler=None)
        print(len(train_dfs), len(train_participants))
        print(time.time() - time_old)
       
        with open(os.path.join(args.output_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)

        print('valid scaler')
        valid_dfs, _ = apply_standard_scaler(valid_dfs, scaler=scaler)
        print(len(valid_dfs), len(valid_participants))
        print(time.time() - time_old)
        
        print('test scaler')
        test_dfs, _ = apply_standard_scaler(test_dfs, scaler=scaler)
        print(len(test_dfs), len(test_participants))
        print("Done creating split dfs", time.time() - time_old)
        

    # get the average values for the dataframes (GRU-D needs this)
    time_old=time.time()
    if os.path.exists(os.path.join(args.output_dir, 'train_means.pickle')):
        with open(os.path.join(args.output_dir, 'train_means.pickle'), 'rb') as f:
            train_means = pickle.load(f)
    else:
        inds = train_dfs['survey'].index.tolist()
        print('train_means: got index')
        if args.zscore:
            # these should decay to 0
            raise NotImplementedError
            train_means = np.zeros(len(train_dfs['activity'].loc[inds, idx[ 'measurement_z',:]].columns.tolist()))
        else:
            multiindex_rolling_features= pd.MultiIndex.from_product([['measurement'], SURVEY_ROLLING_FEATURES+SURVEY_BASELINE_FEATURES + ['weekofyear']*args.weekofyear])
            train_means = train_dfs['survey'].loc[inds, multiindex_rolling_features].sample(5000).mean()
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
        try:train_participants
        except:train_participants=None

    if args.modeltype in ['linear']: 
        print(train_dfs[ 'survey'].index.get_level_values( 'participant_id'))
        train_dfs[ 'survey'].index.get_level_values( 'participant_id').isin(train_participants)
        train_participants = train_dfs[ 'survey'].loc[train_dfs[ 'survey'].index.get_level_values( 'participant_id').isin( train_participants)].index.tolist()
    train_dataset=ILIDataset(train_dfs, args, full_sequence=((args.batch_size==1)|(args.modeltype not in ['gru', 'grud', 'lstm', 'gru_simple'])), feat_subset=args.feat_subset, participants=train_participants)
    train_dataloader=DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_dataloader_workers, collate_fn=merge_fn if args.modeltype in ['gru', 'gru_simple', 'lstm'] else id_collate)
    
    # TODO: This should happen near criterion.    
    if args.modeltype not in ['cnn', 'prob_nn']:
        try:
            pos_weight = 1 / train_dfs['survey']['ili'].mean()
            pos_weight = torch.tensor(pos_weight)
        except:
            pos_weight = None

#     valid_dataset=ILIDataset(valid_dfs, args, full_sequence=args.batch_size==1, feat_subset=args.feat_subset, participants=valid_participants)
#     valid_dataloader=DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_dataloader_workers, collate_fn=merge_fn)
    if args.modeltype in ['linear']: valid_participants = valid_dfs['survey'].loc[valid_dfs['survey'].index.get_level_values('participant_id').isin(valid_participants)].index.tolist()
    valid_dataset=ILIDataset(valid_dfs, args, full_sequence=True, feat_subset=args.feat_subset, participants=valid_participants)
    valid_dataloader=DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=args.num_dataloader_workers, collate_fn=id_collate)
    
    
    
    if args.modeltype in ['linear']: test_participants = test_dfs['survey'].loc[test_dfs['survey'].index.get_level_values('participant_id').isin(test_participants)].index.tolist()
    test_dataset=ILIDataset(test_dfs, args, full_sequence=True, feat_subset=args.feat_subset, participants=test_participants)
    test_dataloader=DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_dataloader_workers, collate_fn=id_collate)

    print("Done creating dataloaders/datasets", time.time() - time_old)
    
    # if there is a forecasting task, determine if they should be tte, or ae.
    

    if any([('24' in t) or ('48' in t) for t in args.target]):
        if len(args.target)>1:
            raise Exception("Incompatible timetoevent window with multitask leaning")
     
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
    l1_regularise=args.l1_constant>0
    if args.modeltype=='grud':
        model=GRUD( input_size, cell_size, hidden_size, train_means, device, fp16=not(args.opt_level=='O0'))
    elif args.modeltype=='linear':
        model = LogisticRegression(input_size)
        l1_constant=args.l1_constant
    elif args.modeltype=='lstm':
        model = LSTMmodel(input_size, hidden_size)
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
        col_set = 'measurement' + args.zscore*'_z'
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
        
        if args.feat_regex != '':
            use_features = [f for f in use_features if re.search(args.feat_regex, f)]

        if args.weekofyear:
            use_features.append('weekofyear')

        X_train, y_train, sample_weight = create_lagged_data(train_dataloader, args, use_features=use_features, target=args.target)
        X_valid, y_valid, _ = create_lagged_data(valid_dataloader, args, use_features=use_features, target=args.target)
        X_test, y_test, _ = create_lagged_data(test_dataloader, args, use_features=use_features, target=args.target)

        print(X_train.columns)

        if args.unbalanced:
            SCALE_POS_WEIGHT = 1
        else:
            try:
                print(y_train)
                SCALE_POS_WEIGHT = (y_train == 0).sum() / (y_train == 1).sum()  # using train-set ratio
            except:
                SCALE_POS_WEIGHT=1
        
        #train model
        print('Training model')
        stime = time.time()
        model = XGBClassifier(scale_pos_weight=SCALE_POS_WEIGHT, nthread=5)
        model.fit(X_train, y_train, sample_weight)
        
        pickle.dump(model, open(os.path.join(args.output_dir, "xgboost_model.bin"), "wb"))
        etime = time.time()
        print('Took %i seconds to fit XGboost' % (etime - stime))
#         model = nn.ModuleDict({'model':model})
        ys_train = model.predict_proba(X_train)[:, 1]
        ys_valid = model.predict_proba(X_valid)[:, 1]
        ys_test = model.predict_proba(X_test)[:, 1]
        
        # save outputs
        write_results(args=args, scores=ys_valid, dataloader=valid_dataloader, istestset=False)
        write_results(args=args, scores=ys_test, dataloader=test_dataloader, istestset=True)
        
        

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
            checkpoint = torch.load(os.path.join( args.output_dir, 'checkpoint_best.pth'))
            checkpoint['state_dict']= {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
            print(checkpoint['state_dict'].keys())
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
            if not(args.test) and ('done_training' in checkpoint.keys()):
                do_train = not(checkpoint['done_training']) # only train if we are not telling to test and if we are not done training
            
            
    
    
    # loss function
    if args.modeltype not in ['cnn', 'prob_nn']:
        criterion=nn.BCEWithLogitsLoss(pos_weight = pos_weight)
        criterion_valid=nn.BCEWithLogitsLoss(pos_weight = pos_weight, reduction='none')
    else:
        criterion=nn.BCELoss(reduction='mean')
        criterion_valid=nn.BCELoss(reduction='mean')
        #criterion=nn.MSELoss(reduction='mean')
        #criterion_valid=nn.MSELoss(reduction='mean')
        #criterion=nn.SmoothL1Loss(reduction='mean')
        #criterion_valid=nn.SmoothL1Loss(reduction='mean')
    
    main_target=sorted(list(args.target))[0]
    assert isinstance(main_target, tuple)
    if do_train:    
        print("Start training")

        #weight_idx = 0
        #weights = {'conv_1': {}, 'conv_2': {}, 'fc1': {}, 'fc2': {}, 'fc3': {}}
        epochs=tqdm(range(start_epoch, args.epochs), desc='epoch')
        for epoch in epochs:
            model.train()
            optimizer.zero_grad()
            batches=tqdm(enumerate(train_dataloader), total=len(train_dataloader))
            for batch_num, batch in batches:
                participants = batch[-1]
                batch = batch[0]
                if not(isinstance(batch, dict)):
                    participants=batch[-1]
                    batch=batch[0]
#                 for k, v in batch.items():
#                     print(k, type(v))
                batch={k:v.to(device) for k, v in batch.items()}
                

                prediction, hidden_states = model(batch['measurement_z'].float() if args.zscore else batch['measurement'].float(),
                                                  batch['measurement'].float(), 
                                                  batch['mask'].float(),
                                                  batch['time'].float(), 
                                                  pad_mask= batch['obs_mask'],
                                                  return_hidden=True)
                if len(args.target)==1:# todo check this test
                
                    if args.batch_size==1:
                        prediction = prediction.unsqueeze(0)
#                     elif len(prediction.data.shape)==2:
#                         prediction = prediction.view(args.batch_size, -1, 2)

                    if args.modeltype not in ['cnn', 'prob_nn']:                        
                        #loss=criterion(prediction[:,:,1].squeeze(-1)[batch['obs_mask']==1], batch[main_target][batch['obs_mask']==1]).sum()
                        if args.batch_size==1:
                            loss = criterion(prediction[:,:,1].reshape((args.batch_size,-1))[batch['obs_mask']==1], batch[main_target[1]][batch['obs_mask']==1]).sum()
                        else:
#                             print('prediction ', prediction.shape)
#                             print(torch.tensor(batch['obs_mask'].data==1).shape)
#                             print((prediction[:, 1].view(-1)[batch['obs_mask'].data==1]).shape)
#                             print(batch['obs_mask'].data.shape)
#                             print(batch[main_target[1]].data.shape)
#                             print((batch[main_target[1]].data[batch['obs_mask'].data==1]).shape)
                            try:
                                loss = criterion(prediction[:, 1].view(-1)[batch['obs_mask'].data==1], batch[main_target[1]].data[batch['obs_mask'].data==1]).sum()
                            except:
                                
                                loss = criterion(prediction[:, 1].view(-1), batch[main_target[1]]).sum()
                        if l1_regularise:
                            l1_reg = None
                            for w in model.parameters():
                                if l1_reg is None:
                                    l1_reg = w.norm(1)
                                else:
                                    l1_reg = l1_reg + w.norm(1)
            
                            loss+= l1_constant*l1_reg
                        
                    else:
                        #for layer in weights.keys():
                            #weights[layer][weight_idx] = getattr(model['model'], layer).weight.data.numpy().flatten()
                            #pd.DataFrame(weights[layer]).to_csv(os.path.join(args.output_dir, layer + '_weights.csv'))
                        mus = prediction[0]
                        #print("mus: " + str(torch.min(torch.mean(mus))))
                        sigma = prediction[1]
                        #print("logstds: " + str(torch.min(torch.mean(logstds))))
                        x = batch['measurement'].float()[:, -1, :] 
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
                    prediction, hidden_states = model(batch['measurement_z'].float() if args.zscore else batch['measurement'].float(),
                                                  batch['measurement'].float(), 
                                                  batch['mask'].float(),
                                                  batch['time'].float(), 
                                                  pad_mask= batch['obs_mask'],
                                                  return_hidden=True)
 
                    if len(args.target)==1:# todo check this test

                        if args.batch_size==1:
                            prediction = prediction.unsqueeze(0)
#                         print(prediction.shape)
#                         print( prediction[:,:,1].reshape((valid_dataloader.batch_size, -1)).shape)
#                         print(batch[main_target[1]].shape)

                        if args.modeltype in ['cnn', 'prob_nn']: 
                            
                            mus = prediction[0]
                            sigma = prediction[1]
                            x = batch['measurement'].float()[:, -1, :]
                            log_lklhd = cnn_lklhd(mus, sigma, x)
                            scores.append(log_lklhd)
                            labels.append(batch[main_target][:, -1])
                            loss=-torch.mean(log_lklhd)
                            #loss=criterion(torch.sigmoid(log_lklhd), torch.ones(log_lklhd.shape[0]))
                        else:
                            if valid_dataloader.batch_size==1:
                                try:
                                    loss = criterion_valid( prediction[:,:,1].reshape((valid_dataloader.batch_size, -1))[batch['obs_mask']==1], batch[main_target[1]][batch['obs_mask']==1]).sum()
                                except:
                                    loss = criterion(prediction[:, 1].view(-1), batch[main_target[1]]).sum()
                            else:
                                try:
                                    loss = criterion(prediction[:, 1].view(-1)[batch['obs_mask'].data==1], batch[main_target[1]].data[batch['obs_mask'].data==1]).sum()
                                except:
                                    loss = criterion(prediction[:, 1].view(-1), batch[main_target[1]]).sum()
 
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
            if (np.mean(losses_epoch_valid) <= best_err) or epoch==start_epoch:
                best_err = np.mean(losses_epoch_valid)
                # If we save using the predefined names, we can load using `from_pretrained`
                print('val_err: ', best_err, ', epoch: ', epoch+1 )
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
                
            epochs.set_description(desc='')
            
            
    # ##################################  run through the validation again and write results
    
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
        assert test_dataloader.batch_size == 1, "batch size for test_dataloader must be one."
        participants.append(batch[-1][0])
        batch = batch[0][0]
        batch={k:v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            if args.modeltype in ['cnn', 'prob_nn']:
                predictions = []
                i = args.max_seq_len
                num_predictions = batch['measurement'].shape[1]

                while i < num_predictions:
                    X = batch['measurement'][:,(i-args.max_seq_len):i,:]
                    
                    prediction, hidden = model(X,
                                                      batch['measurement'].float(),
                                                      batch['mask'].float(),
                                                      batch['time'].float(),
                                                      pad_mask=batch['obs_mask'],
                                                      return_hidden=True)
                            
                    mus = prediction[0]
                    sigma = prediction[1]
                    x = batch['measurement'].float()[:, i, :]
                    log_lklhd = cnn_lklhd(mus, sigma, x)
                    predictions.append(log_lklhd) 
                    i += 1
                    score = torch.sigmoid(log_lklhd)
                    batch_size = batch['measurement'].shape[0]
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
                prediction, hidden_states = model(batch['measurement_z'].float() if args.zscore else batch['measurement'].float(),
                                          batch['measurement'].float(), 
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
                labels.append(batch[main_target[1]].detach().cpu().numpy())
    
    #TODO: remove this order check
    # write the order of participants seen in the test set to a file
    if isinstance(participants[0], tuple)|isinstance(participants[0], list):
        # check to see if it is a participant, date tuple.
        participants, _ = zip(*participants)
    pd.DataFrame(participants).to_csv(os.path.join(args.output_dir, 'valid_batch_order.csv'))
    
    write_results(args, scores, valid_dataloader, istestset=False)

    
            
                
    # ********************************** now test the model **********************************
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
                num_predictions = batch['measurement'].shape[1]

                while i < num_predictions:
                    X = batch['measurement'][:,(i-args.max_seq_len):i,:]
                    
                    prediction, hidden = model(X,
                                                      batch['measurement'].float(),
                                                      batch['mask'].float(),
                                                      batch['time'].float(),
                                                      pad_mask=batch['obs_mask'],
                                                      return_hidden=True)
                            
                    mus = prediction[0]
                    sigma = prediction[1]
                    x = batch['measurement'].float()[:, i, :]
                    log_lklhd = cnn_lklhd(mus, sigma, x)
                    predictions.append(log_lklhd) 
                    i += 1
                    score = torch.sigmoid(log_lklhd)
                    batch_size = batch['measurement'].shape[0]
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
                prediction, hidden_states = model(batch['measurement_z'].float() if args.zscore else batch['measurement'].float(),
                                          batch['measurement'].float(), 
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
                labels.append(batch[main_target[1]].detach().cpu().numpy())
    
    #TODO: remove this order check
    # write the order of participants seen in the test set to a file
    pd.DataFrame(participants).to_csv(os.path.join(args.output_dir, 'test_batch_order.csv'))

    if args.calculate_metrics:
        def get_metrics(labels, scores):
            labels=np.concatenate([l.ravel() for l in labels], axis=0)
            scores=np.concatenate([l.ravel() for l in scores], axis=0)
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
    
    
    
    write_results(args, scores, test_dataloader)
    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--target', type=str, default=('ili',), nargs='+', choices=['ili', 'ili_24', 'ili_48', 'covid', 'covid_24', 'covid_48', 'symptoms__fever__fever'], help='The target')
    parser.add_argument('--modeltype', type=str, default='grud', choices=['grud', 'linear', 'ar_mdl', 'lstm', 'gru', 'cnn', 'prob_nn', 'xgboost', 'lancet_rhr','fastgam'], help='The model to train.')

    parser.add_argument("--max_seq_len", type=int, default=48, help="maximum number of timepoints to feed into the model")
    parser.add_argument("--output_dir", type=str, required=True, help='save dir.')
    parser.add_argument('--data_dir', type=str, required=True, help='Explicit dataset path (else use rotation).')
    parser.add_argument('--home_dir', type=str, required=False, help='For prospective experiments')
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
    parser.add_argument("--l1_constant", type=float, default=0, help="l1 constant")
    parser.add_argument("--grud_hidden", type=int, default=67, help="learning rate for the model")
    parser.add_argument("--regular_sampling", action='store_true', help="Set this flag to have regularly sampled data rather than irregularly sampled data.")
    parser.add_argument("--override_splits", action='store_true', help="Use the same participants in irregular splits as there are in regular")
    
    parser.add_argument("--zscore", action='store_true', help="Set this flag to train a model using the z score (assume forward fill imputation for missing data)")
    parser.add_argument("--only_healthy", action='store_true', help='Set this flag to train the model on only healthy measurements before the first onset of the target illness')
    parser.add_argument("--all_survey", action='store_true', help='Train on all survey')
    parser.add_argument("--calculate_metrics", action='store_true', help='Calculate AUROC and other metrics using the scores calculated before writing them to a csv file.')
    parser.add_argument("--feat_subset", action='store_true', help='in the measurement and measurement_z dataframes only use the subset of features found to work better for xgboost and ridge regression') 
    parser.add_argument("--feat_regex", type=str, default="", help="A regex pattern to filter the features by. If a feature matches the pattern with a call to re.match then the feature will be used. If an empty string is given then all features will be used.")
    parser.add_argument("--num_feature", type=int, default=46, help="number of features passed to the model.") 
    parser.add_argument('--forecast_type', type=str, default='allevent', choices=[ 
        'allevent', 'timetoevent'], help='The target')
    parser.add_argument('--days_ago', type=int, default=7, help='Number of days in the past to include in feature set for XGBOOST')
    parser.add_argument("--unbalanced",  action='store_true', help='no task balancing in loss function') #todo add to all models.
    parser.add_argument("--weekofyear",  action='store_true', help='add week of year') #todo add to all models.
    parser.add_argument("--resample",  action='store_true', help='upsample the data to correct for the week-of-year and region bias. It is recommended to use in conjunction to correct for weekofyear feature.') #todo add to all models.
    parser.add_argument("--add_missingness",  action='store_true', help='add 50% corruption to testing data') #todo add to all models.
    parser.add_argument("--last_7",  action='store_true', help='Formulate dataset ask have you had this symptom in the last 7 days')
    
    
    
                                                                                                                      
    
    
    # add dataset splitting functions to here:
    parser.add_argument("--ignore_participants", action='store_true', help='Just do regular training without splitting participants into differnt test train groups (i.e. only split with time)') 
    parser.add_argument("--train_start_date",  type=str, default='', help='start date for training data (yyyy-mm-dd)')
    parser.add_argument("--train_end_date",  type=str, default='', help='The last day of training data (inclusive)')
    parser.add_argument("--validation_start_date",  type=str, default='', help='start date for training data')
    parser.add_argument("--validation_end_date",  type=str, default='', help='The last day of validation data (inclusive)')
    parser.add_argument("--validation_set_len",  type=int, help='Alternatively provide the len of the desired validation set')
    parser.add_argument("--test_start_date",  type=str, default='', help='start date for test data')
    parser.add_argument("--test_end_date",  type=str, default='', help='The last day of test data (inclusive)')
    

    args = parser.parse_args()
    
    if args.opt_level=='O0':
        amp=None
        
    if args.home_dir is None:
        args.home_dir=args.output_dir
        
    
    
    
        
        
        
        
        
    # todo reload args
    
    print(vars(args))
    if args.test:
        assert os.path.exists(os.path.join(args.output_dir, 'checkpoint_best.pth'))
        args.reload=True
    if not(os.path.exists(args.output_dir)):
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
    if args.train_end_date is not None:
        assert args.validation_start_date is not None, "If training set is defined, you must also define a validation set."
        
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

