import os

import pandas as pd
import numpy as np
import pickle

#TODO: remove seed after debugging.
np.random.seed(7)

from run_model import *
from constants import *

import time


def psplit(df, idx, label):
    """
    Split the participants with a positive label in df into two sets, similarly for participants with a negative label. Return two numpy arrays of participant ids, each array are the chosen id's to be removed from two dataframes to ensure no overlap of participants between the two sets, and keeping half of all participants in df with the same prevelance of event positive participants.
    """
    pos = np.unique(df.loc[df[label] == 1].index.get_level_values(idx))
    all_id = np.unique(df.index.get_level_values(idx))
    neg = np.setdiff1d(all_id, pos)
    
    np.random.shuffle(pos)
    np.random.shuffle(neg)
    
    rmv_1 = np.concatenate((pos[:len(pos)//2], neg[:len(neg)//2]))
    rmv_2 = np.concatenate((pos[len(pos)//2:], neg[len(neg)//2:]))
    
    return rmv_1, rmv_2


def tsplit_df(df, border_day, first_day, last_day):
    """
    Split the data frame df on time.

    Will remove all dates after last_day and before first_day.
    """
    df_test = df.loc[(df.index.get_level_values('date') <= last_day)]
    
    # Split by time
    df_train = df_test.loc[(df_test.index.get_level_values('date') < border_day)]
    
    df_train = df_train.loc[(df_train.index.get_level_values('date') >= first_day)]
    df_test = df_test.loc[(df_test.index.get_level_values('date') >= first_day)]
    
    return df_train, df_test
    

def main(args):
    if args.data_dir == 'recent':
        with open(DATA_PATH_DICTIONARY_FILE, 'rb') as f:
            tmp = pickle.load(f)
        path = tmp[GET_PATH_DICT_KEY(args)]
        args.data_dir = os.path.dirname(path)
        dfs = load_data(args.data_dir,regular=args.regularly_sampled,fname=os.path.basename(path))
    else:
        dfs=load_data(args.data_dir, regular=args.regularly_sampled)
    
    if args.load_participants:
        #TODO: finish implementing this part
        
        # If no participants file in test subdirectory assumption for the flag is broken.
        assert os.exists(os.path.join(args.output_dir, 'test', 'participants.csv')), 'Missing file to load_participants from. Must be in directory {}, called participants.csv'.format(os.path.join(args.output_dir, 'test'))
        raise NotImplementedError

    # Check set of indices are the same for both activity and survey dataframe.
    assert dfs['survey'].index.equals(dfs['activity'].index)

    #Initialize border dates for the data
    earliest_day = dfs['activity'].index.get_level_values('date').min()
    final_day = dfs['activity'].index.get_level_values('date').max()
    
    #Border day is the first day in the prospective set. i.e. the data on the date of border day is included in the prospective set.
    # starts one period size in, so that the first test set is not empty.
    border_day = final_day - pd.to_timedelta(args.split_period, unit='d')
    
    # first_retro_day is an inclusive bound
    first_day = earliest_day if args.train_days < 1 else border_day - pd.to_timedelta(args.train_days, 'd')
    
    # last_prosp_day is an inclusive bound
    last_day = final_day if args.test_days < 1 else border_day + pd.to_timedelta(args.test_days - 1, 'd')
    
    test_p = None
    # Stopping while there is still at least one day of data in the retrospective set.
    while border_day > earliest_day + pd.to_timedelta(args.split_period, 'd'):
        print(border_day)
        
        # split the dataframes on time
        act_train, act_test = tsplit_df(dfs['activity'], border_day, first_day, last_day)
        sur_train, sur_test = tsplit_df(dfs['survey'], border_day, first_day, last_day)

        # Check there are enough positive cases in the prospective set, if args.test_days is
        # larger than 7, just check in the 7 most recent days.
        check_date = border_day + pd.to_timedelta(7, 'd')
        num_positive = sur_test[sur_test.index.get_level_values('date') < check_date][args.target].sum()
        if num_positive >= args.min_positive:

            # Split by participant
            prosp_df = sur_test[sur_test.index.get_level_values('date') >= border_day]

            if args.test_participants=='':
                train_p, test_p = psplit(prosp_df, 'participant_id', args.target)
            else:
                if test_p is None:
                    with open(args.test_participants, 'r') as f:
                        test_p = f.readlines()
                        test_p[-1]=test_p[-1].replace('\n','')
                    
                    train_p = list(set(dfs['activity'].index.get_level_values('participant_id'))-set(test_p))
        
            act_train = act_train[~act_train.index.get_level_values('participant_id').isin(train_p)]
            sur_train = sur_train[~sur_train.index.get_level_values('participant_id').isin(train_p)]
        
            act_test = act_test[~act_test.index.get_level_values('participant_id').isin(test_p)]
            sur_test = sur_test[~sur_test.index.get_level_values('participant_id').isin(test_p)]

            base_train = dfs['baseline'][dfs['baseline'].index.get_level_values('participant_id').isin(act_train.index.get_level_values('participant_id'))]
            base_test = dfs['baseline'][dfs['baseline'].index.get_level_values('participant_id').isin(act_test.index.get_level_values('participant_id'))]

            out_dir = os.path.join(args.output_dir, border_day.strftime('%Y_%m_%d'))
            os.makedirs(out_dir, exist_ok = True)
            os.makedirs(os.path.join(out_dir, 'test'), exist_ok=True)
            
            if args.less_space:
                fname = 'split_dict_' + args.regularly_sampled*'regular' + (not args.regularly_sampled)*'irregular' + '.pkl'

                train_part = np.unique(act_train.index.get_level_values('participant_id'))
                test_part = np.unique(act_test[act_test.index.get_level_values('date') >= border_day].index.get_level_values('participant_id'))
                
                train_dict = {'participant_ids': train_part, 'border_day': border_day, 'test_days': args.test_days, 'train_days': args.train_days}
                test_dict = {'participant_ids': test_part, 'border_day': border_day, 'test_days': args.test_days, 'train_days': args.train_days}

                with open(os.path.join(out_dir, fname), 'wb') as f:
                    pkl.dump(train_dict, f)

                with open(os.path.join(out_dir, 'test', fname), 'wb') as f:
                    pkl.dump(test_dict, f)

            else:
                fname='split_daily_data_'+'regular'*args.regularly_sampled+ \
                        'irregular'*(not(args.regularly_sampled))+'.hdf'
                act_train.to_hdf(os.path.join(out_dir, fname), 'activity')
                act_test.to_hdf(os.path.join(out_dir, 'test', fname), 'activity')

                sur_train.to_hdf(os.path.join(out_dir, fname), 'survey')
                sur_test.to_hdf(os.path.join(out_dir, 'test', fname), 'survey')

                base_train.to_hdf(os.path.join(out_dir, fname), 'baseline', format='table')
                base_test.to_hdf(os.path.join(out_dir, 'test', fname), 'baseline', format='table')
            

        border_day = border_day - pd.to_timedelta(args.split_period, unit='d')
        
        first_day = earliest_day if args.train_days < 1 else border_day - pd.to_timedelta(args.train_days, 'd')
    
        last_day = final_day if args.test_days < 1 else border_day + pd.to_timedelta(args.test_days - 1, 'd')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, required=True, help="Two options, first it's a path to the full dataset which the splits will be generated from, second it's the string recent, which then gets the path from the DATA_PATH_DICTIONARY_FILE from constants")
    parser.add_argument('--output_dir', type=str, required=True, help='Path to directory which will be filled with directories for each split.')
    parser.add_argument('--target', type=str, default='ili', help='Target to split the participants on for membership of test or train set.')
    parser.add_argument('--regularly_sampled', action='store_true', help="Set this flag to have regularly sampled data rather than irregularly sampled data.")
    parser.add_argument('--split_period', type=int, default=('7'), help='The difference in time between one split and the next, as measured by the number of days between the last day of each split.')
    parser.add_argument('--test_days', type=int, default=('7'), help='The number of days to be in the test set. If this is zero, then all future data will be kept in the prospective split. The default is the prospective set to be a single week.')
    parser.add_argument('--train_days', type=int, default=('0'), help='Number of days to retain for each train split, i.e. if train_days == 1 then there will be one day in the train split. If this is zero then all previous data will be kept in the train split. The default is to keep all previous days.')
    parser.add_argument('--min_positive', type=int, default=20, help='Minimum number of positive cases in the prospective test set required for the split to be saved.')
    parser.add_argument('--test_participants', type=str, default='', help='A filepath leading to the list of participants for the test set.')
    parser.add_argument('--load_participants', action='store_true', help='If this flag is set then it is assumed there is a file called participants in a sub directory of output_dir called test. Read the file and use those participants as the held out test set.')
    parser.add_argument('--less_space', action='store_true', help='Instead of saving the dataframes store a dictionary with all information required to get the appropriate partitions. The dictionary will include list of participants for each split, and arguments from this script. To load the data the program must load the full dataset then extract the border date from the output_dir path, other required info will be in this dictionary.')
    parser.add_argument('--fake_data', action='store_true', help='Get the fake data path from the dictionary of paths for reading in the dataframes.')

    args=parser.parse_args()
    
    print(vars(args))
    
#    with open(os.path.join(args.output_dir, 'split_df_args.json'), 'w') as f:
#        f.write(json.dumps(vars(args)))
    
    main(args)

