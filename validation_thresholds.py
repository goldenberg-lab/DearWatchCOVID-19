"""
evaluate.py
"""

import torch, torch.optim, torch.nn as nn, torch.nn.functional as F, torch.nn.init as init
from torch.utils.data import DataLoader, Dataset, RandomSampler, SubsetRandomSampler

from run_model import load_data

from radam import RAdam
try:
    from apex import amp
except ImportError:
    amp=None
from GRUD import GRUD
from models import *

import os
from tqdm import tqdm
import pickle
import json

import pandas as pd
import numpy as np
import random
import sklearn.metrics as skm
from copy import deepcopy


def tte_helper(x, event_times):
    diffs =  event_times - x
    absolute = abs(diffs)
    minimum = min(absolute)/np.timedelta64(1, 'D')
    return minimum


def time_to_event(df_small, col):
    """
    this is done for each participant_id
    """
    df_small.reset_index(inplace=True)
    event_times = df_small.loc[df_small[col]==1, 'date']
    f = (lambda x: tte_helper(x, event_times))
    df_small['time_to_event'] = df_small['date'].apply(f)
    return df_small

    
def build_tte(col):
    def tte(df_small):
        return time_to_event(df_small, col)
    return tte

def set_thresholds(results_df, ppv=0.2):
    """
    set_thresholds takes a set of true labels and many sets of predictions, and
    for each prediction set selects the best threshold in results_df for each 
    smallest acceptable ppv given.

    results_df: pandas.DataFrame, with one column labelled with the substring 
                'label', and a number of columns labelled with the substring
                'score'
    ppv: float, value between 0 and 1 not inclusive indicating the level of 
            precision the threshold must attain.
    return: dict of dicts, where each key is the score column name and each 
            value is precision, recall, thresh_choices and threshold. 
            Precision, is the set of all precisions for possible thresholds,
            similarly for recall, and thersh_choices is the selection of 
            possible thresholds. Lastly, threshold has an entry for each 
            given ppv, where the key is the smalles acceptable ppv used to 
            select the theshold which is the value.
    """
    label_col = [col for col in results_df.columns.tolist() if 'label' in col.lower()]
    assert len(label_col) == 1
    label_col = label_col[0]
    true_label=results_df[label_col].values

    thresholds={}

    cols=[col for col in results_df.columns.tolist() if 'score' in col.lower()]

    for score in cols:
        thresholds[score] = {}
        
        y_pred=results_df[score].values
        # calculate ppv at a particular threshold
        # p=precision_score(true_label, y_pred>0.2, average='binary', sample_weight=None)
        # print(p)
        precision, recall, thresh_choices = skm.precision_recall_curve(true_label, y_pred)
        thresholds[score]['precision'] = precision
        thresholds[score]['recall'] = recall
        thresholds[score]['thresh_choices'] = thresh_choices
        thresholds[score]['threshold'] = {}
        # find where the precision crosses our desired ppv
        indices = precision.reshape((len(precision), 1)) > np.asarray(ppv).reshape((1,len(ppv)))
        for i in range(indices.shape[1]):
            # for each ppv get the best threshold, add it to the dictionary of 
            # threhold for the given score type
            small_val_prec = np.min(precision[indices[:,0]])
            ppv_specific_thresh = thresh_choices[(precision == small_val_prec)[:-1]]
            thresholds[score]['threshold']['ppv: '+str(ppv[i])] = ppv_specific_thresh
        
        print(score + ' best threshold: ' + str(thresholds[score]['threshold']))
        # fig=plt.figure()
        # plt.plot(thresholds, precision[1:])
        # plt.title(score)
        # plt.show()

    return thresholds


def main(args):
    """
    """
    # even if args.dirs is a single element the parser still returns  a list, but
    # I am generally in favour of potentially redundant checks if not costly
    if isinstance(args.dirs, str):
        args.dirs=[args.dirs]
    
    for d in args.dirs:
        # if validation score exists open it
        # else:
        assert os.path.exists(os.path.join(d, 'validation_scores.csv')), 'Missing validation_scores.csv in dir: ' + d 
        
        val_res = pd.read_csv(os.path.join(d, 'validation_scores.csv'))
        
        thresholds = set_thresholds(val_res, ppv=args.precision_thresholds)

        pickle.dump(thresholds, open(os.path.join(d, 'chosen_threshold.pkl'), 'wb'))


if __name__=="__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dirs', type=str, nargs='+', help='The directories with the .csv files of scores')
    parser.add_argument('--dataset_dir', type=str, default='/datasets/evidationdata', help='The directories with the .csv files of scores')
    parser.add_argument('--precision_thresholds', type=float, nargs='+', help='Use the returned threshold to acheive this precision on the validation set.')    

    args = parser.parse_args()
    
    main(args)
