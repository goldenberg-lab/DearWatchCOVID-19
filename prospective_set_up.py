import pandas as pd
import numpy as np

def get_retro(dfs, d):
    """
    Restrict the dataframes in dfs to correspond with the retrospective design as dictated by the dictionary d.
    """
    if 'activity' in dfs.keys(): new_act = dfs['activity'].copy()
    new_sur = dfs['survey'].copy()
    new_base = dfs['baseline'].copy()
    
    # Restrict time
    if 'activity' in dfs.keys(): new_act = new_act[new_act.index.get_level_values('date') < d['border_day']]
    new_sur = new_sur[new_sur.index.get_level_values('date') < d['border_day']]
    
    # Restrict time beginning for truncated setting
    if d['train_days'] > 0:
        if 'activity' in dfs.keys(): new_act = new_act[new_act.index.get_level_values('date') >= d['border_day'] - pd.to_timedelta(d['train_days'])]
        new_sur = new_sur[new_sur.index.get_level_values('date') >= d['border_day']] - pd.to_timedelta(d['train_days'])
    
    # Restrict participants
    if 'activity' in dfs.keys(): new_act = new_act[new_act.index.get_level_values('participant_id').isin(d['participant_ids'])]
    new_sur = new_sur[new_sur.index.get_level_values('participant_id').isin(d['participant_ids'])]
    new_base = new_base[new_base.index.get_level_values('participant_id').isin(d['participant_ids'])]
    if 'activity' in dfs.keys():
        new_dfs = {'activity': new_act, 'survey': new_sur, 'baseline': new_base}
    else:
        new_dfs = {'survey': new_sur, 'baseline': new_base}

    return new_dfs


def get_prosp(dfs, d):
    """
    Restrict the dataframes in dfs to correspond with the prospective design as 
    dictated by the dictionary d. This will only restrict to the set of 
    participants chosen for the prospective set, and the last day if there is 
    a number of test days in the dictionary. This is because different models 
    require different amounts of previous data and the dictionary should be 
    used by all models, so the unnecessary retrospective dates must be removed 
    afterwards.
    """
    if 'activity' in dfs.keys(): new_act = dfs['activity'].copy()
    new_sur = dfs['survey'].copy()
    new_base = dfs['baseline'].copy()
    
    if d['test_days'] > 0:

        if 'activity' in dfs.keys(): new_act = new_act[new_act.index.get_level_values('date') < d['border_day'] + pd.to_timedelta(d['test_days'], 'd')]
        new_sur = new_sur[new_sur.index.get_level_values('date') < d['border_day']+ pd.to_timedelta(d['test_days'], 'd')]


    # Restrict participants
    print('participants in pickle ', len(d['participant_ids']))
    if 'activity' in dfs.keys(): new_act = new_act[new_act.index.get_level_values('participant_id').isin(d['participant_ids'])]
    new_sur = new_sur[new_sur.index.get_level_values('participant_id').isin(d['participant_ids'])]
    new_base = new_base[new_base.index.get_level_values('participant_id').isin(d['participant_ids'])]
    
    if 'activity' in dfs.keys():
        new_dfs = {'activity': new_act, 'survey': new_sur, 'baseline': new_base}
    else:
        new_dfs = {'survey': new_sur, 'baseline': new_base}
    
    return new_dfs


