import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import sklearn.metrics as skm
import re
from constants import *

from run_model import *
from prospective_set_up import get_prosp

import pickle


def main(args):
    
    arg_suf = args.out_suf

    # Assumed fit model in sub dir out
    with open(os.path.join(args.home_dir, 'out_xgb' + arg_suf, 'xgboost_model.bin'), 'rb') as f:
        model = pickle.load(f) 
    
    if args.eval_all_data:
        dfs = load_data('/datasets/evidationdata/', regular=args.regularly_sampled, only_healthy=args.only_healthy)
    else:
        try:
            d_name = 'split_dict_' + args.regularly_sampled*'regular' + (not args.regularly_sampled)*'irregular' + '.pkl'
            with open(os.path.join(args.home_dir, 'test', d_name), 'rb') as f:
                prosp_specs = pickle.load(f)
            f_name = 'split_daily_data_' + args.regularly_sampled*'regular' + (not args.regularly_sampled)*'irregular' + '.hdf'
            assert not os.path.exists(os.path.join(args.home_dir, 'test', f_name)), 'Found a prospective dictionary file and full dataframe file in {}, uncertain which should be used.'.format(args.data_dir)
            
            with open(DATA_PATH_DICTIONARY_FILE, 'rb') as f:
                tmp = pickle.load(f)
            path = tmp[GET_PATH_DICT_KEY(args)]
            d = os.path.dirname(path)
            f = os.path.basename(path)
            dfs=load_data(d, regular=args.regularly_sampled, only_healthy=args.only_healthy, fname=f)
            
            del path, d, f, tmp

            dfs = get_prosp(dfs, prosp_specs)
            # Get the number of test_days, update the passed in argument.          
            if prosp_specs['test_days'] < 1:
                max_date = dfs['activity'].index.get_level_values('date').max()
                # Add one additional day to account for the border day being in the prosp set.
                args.test_size = (max_date - prosp_specs['border_day'] + pd.to_timedelta(1, 'd')).days
            else:
                args.test_size = prosp_specs['test_days']
            last_day_str = str(prosp_specs['border_day'])
        except OSError:
            # Assumed test part of split in sub dir test
            dfs=load_data(os.path.join(args.home_dir, 'test'), regular=args.regularly_sampled, only_healthy=args.only_healthy)
            last_day_str = X_test_flat.index.get_level_values('date').max()
    if args.weekofyear:
        idx=pd.IndexSlice
        dfs['activity']['weekday']=(dfs['activity'].index.get_level_values('date').weekday<5).astype(np.int32) # add week of year
        dfs['activity'].loc[:, idx['measurement', 'weekofyear']]=dfs['activity'].index.get_level_values('date').isocalendar().week.astype(np.int32).values
        dfs['activity'].loc[:, idx['measurement_z', 'weekofyear']]=dfs['activity'].index.get_level_values('date').isocalendar().week.astype(np.int32).values
        dfs['activity'].loc[:, idx['measurement_noimp', 'weekofyear']]=dfs['activity'].index.get_level_values('date').isocalendar().week.astype(np.int32).values
        dfs['activity'].loc[:, idx['mask', 'weekofyear']]=np.ones(len(dfs['activity']))
        dfs['activity'].loc[:, idx['time', 'weekofyear']]=np.ones(len(dfs['activity']))
    
    #Get the last day of the prospective week and subset the df down to only the data needed to speed up the process of getting the day lagged features. Only implemented when using all participants since loading the full dataset is about ten times larger than it needs to be.
    if args.eval_all_data:
        pattern = r'\d{4}_\d{2}_\d{2}'
        reg_search = re.search(pattern, args.home_dir)
        assert reg_search, "No date matched the pattern " + pattern + " in home_dir, unable to tell what the last day should be"
        last_day = pd.to_datetime(reg_search.group(0).replace('_', '-'))
        first_lag_day = last_day - pd.to_timedelta(args.max_seq_len + args.test_size, 'd')

        dfs['activity'] = dfs['activity'][(dfs['activity'].index.get_level_values('date') <= last_day) & (dfs['activity'].index.get_level_values('date') >= first_lag_day)]
        dfs['survey'] = dfs['survey'][(dfs['survey'].index.get_level_values('date') <= last_day) & (dfs['survey'].index.get_level_values('date') >= first_lag_day)]


    test_participants = np.unique(dfs['activity'].index.get_level_values('participant_id'))

    test_dfs = {k:v.loc[v.index.get_level_values('participant_id').isin(test_participants), :     ] for k, v in dfs.items()}
    
    # Assumed data scaler in sub dir out.
    if not(args.zscore):
        with open(os.path.join(args.home_dir, 'out_xgb' + arg_suf,'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
            test_dfs, _ = apply_standard_scaler(test_dfs, args, scaler=scaler)

    #test_dfs, _ = apply_standard_scaler(test_dfs, scaler=scaler)
    test_dataset=ILIDataset(test_dfs, args, full_sequence=True, feat_subset=args.feat_subset)
    test_dataloader=DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_dataloader_workers, collate_fn=id_collate)
    
    use_features = ['heart_rate_bpm', 'walk_steps', 'sleep_seconds',
             'steps__rolling_6_sum__max', 'steps__count', 'steps__sum',
             'steps__mvpa__sum', 'steps__dec_time__max',
             'heart_rate__perc_5th', 'heart_rate__perc_50th',
             'heart_rate__perc_95th', 'heart_rate__mean',
             'heart_rate__stddev', 'active_fitbit__sum',
             'active_fitbit__awake__sum', 'sleep__asleep__sum',
             'sleep__main_efficiency', 'sleep__nap_count',
             'sleep__really_awake__mean',
             'sleep__really_awake_regions__countDistinct', 'weekday', 
             'weekofyear']

    fake_features = [c[1] for c in test_dataloader.dataset.numerics.columns if 'fake' in c[1]]
    use_features = use_features + fake_features

    X_test, y_test, _ = create_lagged_data(test_dataloader, args, use_features=use_features, target=args.target, bound_events=False)
    
    # FYI, args.test_size handles the case when it is zero earlier... so don't panic...
    last_training = X_test.index.get_level_values('date').max() - pd.to_timedelta(args.test_size, 'd')
    
    X_test = X_test.loc[(X_test.index.get_level_values('date') > last_training)]
    # Got error when stepping through where the predict_proba requires the column order to be the same, this error didn't show up in the log when running but when pdb stepping afterwards, added this line to correct the order, if this breaks later this may be why.
    X_test_flat = X_test
    X_test_flat.columns = [' '.join(c).strip() for c in X_test_flat.columns.values]
    X_test_flat = X_test_flat[model.get_booster().feature_names]
    
    ys = y_test.loc[(y_test.index.get_level_values('date') > last_training)]
    y_tbl = ys.to_frame()
    
    if len(np.unique(dfs['survey'][args.target[0]])) == 2:
        yhats = model.predict_proba(X_test_flat)[:, 1]    
        y_tbl[('label', 'predicted')] = yhats    
    
    else:
        yhats = model.predict_proba(X_test_flat)
        y_tbl = multi_class_column_scores(yhats, X_test_flat, test_dataloader, args) 
        
    # reset index and drop multilevel
    y_tbl.columns = y_tbl.columns.droplevel(0) # todo check this line still works after merge
    y_tbl=y_tbl.reset_index()
    
    
    assert os.path.exists( os.path.join( args.home_dir,'out_xgb' + arg_suf, 'thresholds_activity.json')), "no file: "+ os.path.join( args.home_dir, 'out_xgb' + arg_suf, 'thresholds_activity.json')
    # load the thresholds if they exist:
    if os.path.exists( os.path.join( args.home_dir,'out_xgb' + arg_suf, 'thresholds_activity.json')):
        with open(os.path.join( args.home_dir,'out_xgb' + arg_suf, 'thresholds_activity.json'), 'r') as f:
            thresholds = json.load(f)
        # apply this to the data and get the scores.
        for k, v in thresholds.items():
            #y_tbl[args.target+'_pred_'+k] = np.asarray(yhats >= float(v) ).astype(np.int32)
            print(args.target[0])
            y_tbl[args.target[0]+'_pred_'+k] = np.asarray(yhats >= float(v) ).astype(np.int32)

                    
    
    if args.eval_all_data:
        fn = args.target[0] + '_xgb_prosp_allparticipants_results' + arg_suf + '.csv'
    else:
        fn = args.target[0] + '_xgb_prosp_testset_results' + arg_suf + '.csv'

    y_tbl.to_csv(os.path.join(args.home_dir, 'test', fn), index=False)
    
    if len(np.unique(dfs['survey'][args.target[0]])) > 2:
        return None    

#    auc = skm.roc_auc_score(y_tbl[('label', args.target[0])], y_tbl[('label', 'predicted')])
#    aps = skm.average_precision_score(y_tbl[('label', args.target[0])], y_tbl[('label','predicted')])

    
    try:
        auc = skm.roc_auc_score(y_tbl[('label', args.target[0])], y_tbl[('label', 'predicted')])
        aps = skm.average_precision_score(y_tbl[('label', args.target[0])], y_tbl[('label','predicted')])
    except:
        auc = skm.roc_auc_score(y_tbl[args.target[0]], y_tbl['predicted'])
        aps = skm.average_precision_score(y_tbl[args.target[0]], y_tbl['predicted'])
        
    # Write to a different file if a true test set or if just a prospective week.
    if args.eval_all_data:
        if not os.path.exists(os.path.join(args.home_dir, '..', 'all_participant_prosp_xgb' + arg_suf + '.csv')):
            with open(os.path.join(args.home_dir, '..', 'all_participant_prosp_xgb' + arg_suf + '.csv'), 'w') as f:
                f.write("Last_day, AUC, avg_precision_score \n")
        with open(os.path.join(args.home_dir, '..', 'all_participant_prosp_xgb' + arg_suf + '.csv'), 'a+') as f:
            f.write(last_day_str + ', ' + str(auc) + ', ' + str(aps) + '\n')
    else:
        with open(os.path.join(args.home_dir, '..', 'all_xgb_auc' + arg_suf + '.csv'), 'a+') as f:
            f.write(last_day_str + ', ' + str(auc) + ', ' +str(aps) + '\n')

    return None


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--target', type=str, default='ili', nargs='+', choices=['ili', 'ili_24', 'ili_48', 'covid', 'covid_24', 'covid_48', 'symptoms__fever__fever', 'flu_covid'], help='Set this flag to match how the model was trained. The target')
    parser.add_argument("--home_dir", type=str, required=True, help='Base directory for reading data and fit model. Assumed home_dir will contain an out folder, which contains the fit model on the training part of the split and the scaler used, and the test folder, which contains the test part of the split.')
    parser.add_argument('--days_ago', type=int, default=7, help='Set this flag to match how the model was trained. Number of days in the past t     o include in feature set for XGBOOST')
    parser.add_argument("--only_healthy", action='store_true', help='Set this flag to match how the model was trained. Set this flag to train the model on only healthy measurements before the first onset of the target illness')
    parser.add_argument("--regularly_sampled", action='store_true', help="Set this flag to match how the model was trained. Set this flag to have regularly sampled data rather than irregularly sampled data.")
    parser.add_argument("--feat_subset", action='store_true', help='Set this flag to match how the model was trained. in the measurement and measurement_z dataframes only use the subset of features found to work better for xgboost and ridge regression')
    parser.add_argument('--num_dataloader_workers', type=int, default=4, help='# dataloader workers.')
    parser.add_argument('--test_size', type=int, default=7, help='# of days in test set which were not in the training data')
    parser.add_argument('--max_seq_len', type=int, default=7, help='Set this to the same value as the model was trained. Maximum number of timepoints to feed into the model, required for load_data.')
    parser.add_argument('--zscore', action='store_true', help='Set this flag to match how the model was trained. Set this flag to train a model using the z score (assume forward fill imputation for missing data)')
    parser.add_argument('--weekofyear', action='store_true', help='Set this flag to match how the model was trained. Use the week_of_year feature when fitting testing the xgb model.')
    parser.add_argument('--resample', action='store_true', help='Set this flag to match how the model was trained. upsample the data to correct for the week-of-year and region bias. It is recommended to use in conjunction to correct for weekofyear feature.')
    parser.add_argument('--eval_all_data', action='store_true', default=False, help='Set this flag to evaluate the model on all participants not just a held out set.') 
    parser.add_argument('--out_suf', default='', type=str, help='Suffix to append (before the extension) to each output file name.')
    parser.add_argument('--no_imputation', action='store_true', help="If set use the data prior to imputation preserving the missing values as NA.")   
    parser.add_argument('--modeltype', default='xgboost', choices=['xgboost'], help="This script has been set up to evaluate xgboost results, this argument should not be used and is here to allow usage of functions which rely on arguments required for script run_model.py.")
    parser.add_argument('--fake_data', action='store_true', help='Set to true if using data with fake features generated, affects which hdf file is loaded from the dictionary of paths.')
    args = parser.parse_args()

    print(vars(args))
    with open(os.path.join(args.home_dir, 'test', 'args.json'), 'w') as f:
        f.write(json.dumps(vars(args)))


    main(args)


