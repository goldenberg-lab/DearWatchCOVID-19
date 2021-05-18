"""
evaluate.py
"""

import numpy as np
import pandas as pd
import sklearn.metrics
from run_model import load_data
import gain_metric_utils

import os
import pdb
import glob


def tte_helper(x, event_times):
    if len(event_times) == 0:
        return 100
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
    df_small['tte_' + col] = df_small['date'].apply(f)
    return df_small

    
def build_tte(col):
    def tte(df_small):
        return time_to_event(df_small, col)
    return tte

def get_manuscript_metrics(df, col=None, threshold_98recall = 0.5):
    """
    create 
    """
    
    score_dicts = []
    constant_cols = ['participant_id', 'date']
    if col is None:
        # Assumes any column with a corresponding score column is a true label,
        # matched with the score from a prediction model.
        cols= [s for s in df.columns.to_list() if s + '_score' in df.columns.to_list()]

    for col in cols:
        tte=build_tte(col)
        
        score_dict={}
        # get the number of participants
        score_dict['num_participants'] = 0#TODO: not used right now
        # get the number of timestamps
        score_dict['num_prediction_opportunities'] = len(df)

        # get the prediction density
        score_dict['prediction_density'] = df[col].notna().sum()/len(df)
    
        labels = df[col].values.ravel()
        scores = df[col + '_score'].values.ravel()
        # get AUC   
        if len(set(labels[~np.isnan(scores)]))!=2:
            score_dict['AUROC']=np.nan
            score_dict['AUPR'] = np.nan
            score_dict['precision_at_98recall'] = np.nan
            score_dict['precision_at_98recall_TTE'] = np.nan
        else:
            score_dict['AUROC'] = sklearn.metrics.roc_auc_score(labels[~np.isnan(scores)], scores[~np.isnan(scores)])
            # get AUPR
            score_dict['AUPR'] = sklearn.metrics.average_precision_score(labels[~np.isnan(scores)], scores[~np.isnan(scores)])

            df = df.groupby('participant_id').apply(tte).drop(columns=['participant_id'])

            # get precision at 98\% recall
            y_pred = scores >= threshold_98recall
            score_dict['precision_at_98recall'] = sklearn.metrics.precision_score(labels, y_pred)
            #     get the time-to-event
            score_dict['precision_at_98recall_TTE'] = df.loc[df[col]>threshold_98recall, 'tte_' + col].median()   

    # fairness 
    # get the baseline data
        baseline_df = load_data(args.dataset_dir, load_activity=False, load_survey=False)['baseline']
        baseline_df = baseline_df.assign(gender=lambda x: x.gender_male + x.gender_other*2)
        df = df.join(baseline_df[['gender']])
    # pip install aif360
        
        # Not calculating fairness metrics to reduce computational cost.

        #from aif360.sklearn.metrics import disparate_impact_ratio, average_odds_error, generalized_fpr
        #from aif360.sklearn.metrics import generalized_fnr, difference
        #fairness_df = df.reset_index().set_index(['participant_id', 'date', 'gender'])
        #score_dict['disparate_impact_at_98recall_gender'] = disparate_impact_ratio(fairness_df, y_pred, prot_attr='gender')
        #score_dict['average_odds_at_98recall_gender'] = average_odds_error(fairness_df[[col]], y_pred, prot_attr='gender')
        
        score_dicts.append(score_dict)
    
    return score_dicts


def main(args):
    """
    """
    
    if isinstance(args.dirs, str):
        args.dirs=[args]

    score_dfs=[]
    for d in args.dirs:
        # run evaluation function
        print(d)
        # Check that only one file has the correct suffix
        path_matches = glob.glob(os.path.join(d, args.file_pattern))
        if len(path_matches) < 1:
            print('No testset_results in folder ' + str(d))
            continue
        elif len(path_matches) > 1:
            print('More than one testset_results file in folder ' + str(d))
            continue
                
        df = pd.read_csv(path_matches[0], parse_dates=['date'])
        df.set_index(['participant_id', 'date'], inplace=True)
        try:
            scores = get_manuscript_metrics(df, threshold_98recall=0.5) #TODO: we need to load the threshold from the validation scripts
        except:
            print("failed on directory: " + str(d))
            continue

        score_dfs.append(pd.DataFrame(scores, index=[d]))
        
    scores_df = pd.concat(score_dfs)
    scores_df.to_csv(args.output_path, index_label='result_directory')
    #for column in score_df.columns:
        #if 'score' not in columns:
            #continue
           
        #get_manuscript_metrics(df, col=col)
    


if __name__=="__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dirs', type=str, nargs='+', help='The directories with the .csv files of scores')
    parser.add_argument('--dataset_dir', type=str, default='/datasets/evidationdata', help='The directories with the .csv files of scores')
    parser.add_argument('--output_path', type=str, default='./evaluation_results.csv', help='Path to write all evaluation metrics to')
    parser.add_argument('--file_pattern', type=str, default='*_testset_results.csv', help='Pattern to be used to identify results files to be joined together by the date folder they are in. assumed each of the files is in a separate dir in dirs.')
    args = parser.parse_args()
    
    main(args)
