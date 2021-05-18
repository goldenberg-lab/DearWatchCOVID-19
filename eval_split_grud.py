import os
import numpy as np
import pandas as pd
from GRUD import GRUD
import sklearn.metrics as skm
import re

from prospective_set_up import get_retro, get_prosp


from run_model import *

import pickle as pkl

def main(args):
    
    arg_suf = args.out_suf

    device='cuda' if torch.cuda.is_available() else 'cpu'
    
    # Store the set of suffixes attached to the model output folder in home_dir
    home_suffix = 'out_grud' + arg_suf

    assert os.path.exists(os.path.join(args.home_dir, home_suffix, 'train_means.pickle')), 'No train_means file to load in ' + str(args.home_dir) + '/' + home_suffix

    with open(os.path.join(args.home_dir, home_suffix, 'train_means.pickle'), 'rb') as f:
        train_means = pd.read_pickle(f)
        #train_means = pickle.load(f)

    input_size = args.num_feature + args.weekofyear
    cell_size = args.grud_hidden
    hidden_size = args.grud_hidden
    model=GRUD( input_size, cell_size, hidden_size, train_means, device, fp16=not(args.opt_level=='O0')) 

    
    if os.path.exists(os.path.join(args.model_dir, 'checkpoint_best.pth'))&(args.model_dir!=''):
        checkpoint = torch.load(os.path.join(args.model_dir, 'checkpoint_best.pth'))
    # Assumed fit model in sub dir out_grud with suffix if no model_dir is provided.
    else:
        assert os.path.exists(os.path.join(args.home_dir, home_suffix, 'checkpoint_best.pth')), "Missing trained model, expect file: " + str(os.path.join(args.home_dir, home_suffix, 'checkpoint_best.pth'))
        checkpoint = torch.load(os.path.join( args.home_dir, home_suffix, 'checkpoint_best.pth'), map_location=device)
    checkpoint['state_dict']= {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
    print(checkpoint['state_dict'].keys())
    try:
        model.load_state_dict(checkpoint['state_dict'])
    except:
        print(checkpoint['state_dict'].keys())
        model['model'].load_state_dict(checkpoint['state_dict'])
    model.to(device)

    if (not args.eval_train)|(args.data_dir is None):
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
            dfs = get_prosp(dfs, prosp_specs)

            # Clean up temporary objects
            del path, d, f, tmp
            # Get the number of test_days, update the passed in argument.          
            if prosp_specs['test_days'] < 1:
                max_date = dfs['activity'].index.get_level_values('date').max()
                args.test_size = (max_date - prosp_specs['border_day']).days
            else:
                args.test_size = prosp_specs['test_days']
            last_day_str = prosp_specs['border_day']
        except OSError: 
            dfs=load_data(os.path.join(args.home_dir, 'test'), regular=args.regularly_sampled, only_healthy=args.only_healthy)
            last_day_str = X_test_flat.index.get_level_values('date').max()  
    else:
        dfs=load_data(args.home_dir, regular=args.regularly_sampled, only_healthy=args.only_healthy)
    
    if args.weekofyear:
        idx=pd.IndexSlice
        dfs['activity']['weekday']=(dfs['activity'].index.get_level_values('date').weekday<5).astype(np.int32) # add week of year
        dfs['activity'].index
        dfs['activity'].index.get_level_values('date')
        dfs['activity'].index.get_level_values('date' ).isocalendar().week.astype(np.int32)
#         dfs['activity'].loc[:, idx['measurement', 'weekofyear']]=dfs['activity'].index.get_level_values('date').weekofyear
#         dfs['activity'].loc[:, idx['measurement_z', 'weekofyear']]=dfs['activity'].index.get_level_values('date').weekofyear
        dfs['activity'].loc[:, idx['measurement', 'weekofyear']]=dfs['activity'].index.get_level_values('date').isocalendar().week.astype(np.int32).values
        dfs['activity'].loc[:, idx['measurement_z', 'weekofyear']]=dfs['activity'].index.get_level_values('date').isocalendar().week.astype(np.int32).values
        dfs['activity'].loc[:, idx['mask', 'weekofyear']]=np.ones(len(dfs['activity']))
        dfs['activity'].loc[:, idx['time', 'weekofyear']]=np.ones(len(dfs['activity']))
        
        
    if not(args.eval_train):
        # remove participants who are not in the prospective set
        first_test_day = dfs['survey'].index.get_level_values('date').max() - pd.to_timedelta(args.test_size, 'd')
        # prospective participants were only necessarily removed from the last seven days of the dataframe.
        tmp = dfs['activity'][dfs['activity'].index.get_level_values('date') >= first_test_day]
        test_participants = np.unique(tmp.index.get_level_values('participant_id'))

        test_dfs = {k:v.loc[v.index.get_level_values('participant_id').isin(test_participants), :     ] for k, v in dfs.items()}

    else:
        # reload the training data (but we will save it as test_dfs)
        test_participants = pd.read_csv(os.path.join(args.home_dir, home_suffix, 'train_participants.csv')).values.ravel().tolist()
        test_dfs = {k:v.loc[v.index.get_level_values('participant_id').isin(test_participants), :     ] for k, v in dfs.items()}

    # Assumed data scaler in sub dir out.
    with open(os.path.join(args.home_dir, home_suffix, 'scaler.pkl'), 'rb') as f:
        scaler = pkl.load(f)
    test_dfs, _ = apply_standard_scaler(test_dfs, args, scaler=scaler)
    test_dataset=ILIDataset(test_dfs, args, full_sequence=True, feat_subset=args.feat_subset)
    test_dataloader=DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_dataloader_workers, collate_fn=id_collate)
    # Need this last_test_day for participants whose data ends before the final date to calculate how many days to keep. 
    last_prosp_day = test_dfs['survey'].index.get_level_values('date').max()
    proba_projection=torch.nn.Softmax(dim=1)
    labels=[]
    scores = []
    participants = []
    model.eval()
    batches=tqdm(test_dataloader, total=len(test_dataloader))
    for batch in batches:
        assert test_dataloader.batch_size == 1, "batch size for test_dataloader must be one."
        participant = batch[-1][0]
        batch = batch[0][0]
        batch={k:v.to(device) for k, v in batch.items()}
        #Some participants don't have data all the way till the last day of the prospective test set, so here calculate how many of those days there are take that many days out of the returned scores and labels.
        participant_last_day = test_dfs['survey'].loc[participant]['ili'].index.get_level_values('date').max()
        num_days = ( last_prosp_day - participant_last_day).days
        keep_days = args.test_size - num_days
        #if keep_days != 7:
        #    print('p: {0}; pld: {1}; k: {2}'.format(participant, participant_last_day, keep_days)) 
        with torch.no_grad():
            prediction, hidden_states = model(batch['measurement_z'].float() if args.zscore else batch['measurement'].float(), batch['measurement'].float(), batch['mask'].float(), batch['time'].float(), pad_mask= batch['obs_mask'], return_hidden=True)

            scores.append(proba_projection(prediction.view(-1, 2))[:, -1].detach().cpu().numpy()[-keep_days:])
#             print((batch[args.target[0]].detach().cpu().numpy()[:, -args.test_size:]).shape)
#             print(batch[args.target[0]].shape)
#             print(args.test_size)
#             raise
        assert len(batch[args.target[0]].detach().cpu().numpy()[:, -keep_days:].ravel())>0
        assert len(batch[args.target[0]].detach().cpu().numpy()[:, -keep_days:].shape)==2
        labels.append(batch[args.target[0]].detach().cpu().numpy()[:, -keep_days:].ravel())
        participants.append([participant]*scores[-1].shape[-1]) 
        
#         if len(labels)>10:
#             break

#     print(labels[0])
#     print(len(labels))
#     labels = np.concatenate(tuple(labels), axis=1).squeeze()
    try:
        labels = np.concatenate(tuple(labels), axis=0).squeeze()
    except:
        print(labels)
        print(tuple(labels))
        raise
    scores = np.concatenate(tuple(scores))
    
    last_day = dfs['activity'].index.get_level_values('date').max()
    prospective_days = [last_day - pd.to_timedelta(args.test_size-1,unit='d') + pd.to_timedelta(n, unit='d') for n in range(args.test_size)]

    dates = np.concatenate(tuple([prospective_days[:len(p)] for p in participants]))
    
    participants = np.concatenate(tuple(participants))

    auc = skm.roc_auc_score(labels, scores)
    aps = skm.average_precision_score(labels, scores)

    args.output_dir = args.home_dir

    out_df = pd.DataFrame({args.target[0] + '_score': scores, args.target[0]: labels, 'participant_id': participants, 'date': dates})
    
    assert os.path.exists( os.path.join( args.home_dir,'out_grud' + arg_suf, 'thresholds_activity.json')), print(os.path.join( args.home_dir,'out_grud' + arg_suf, 'thresholds_activity.json'))
    
    if os.path.exists( os.path.join( args.home_dir,'out_grud' + arg_suf, 'thresholds_activity.json')):
        with open(os.path.join( args.home_dir,'out_grud' + arg_suf, 'thresholds_activity.json'), 'r') as f:
            thresholds = json.load(f)
        # apply this to the data and get the scores.
        for k, v in thresholds.items():
            print(args.target[0])
            out_df[args.target[0]+'_pred_'+k] = np.asarray(scores >= float(v) ).astype(np.int32)
    
    # Check if an explicit path was passed for the model, if so get the date it was trained till.
    if args.model_dir!='':
        model_date = os.path.split(os.path.split(args.model_dir)[0])[1].replace('_', '-')
        fn = args.target[0] + '_grud_prosp_testset_results_' + arg_suf + model_date +'.csv'
        write_path = os.path.join(args.home_dir, 'test', fn)
    # Since model_dir wasn't passed the model must match the date of the home_dir.
    else:
        fn = args.target[0] + 'grud_prosp_testset_results' + arg_suf +'.csv'
        if args.eval_train:
            fn = args.target[0] + 'grud_prosp_trainset_results'+arg_suf+'.csv'
        write_path = os.path.join(args.home_dir, 'test', fn)
    
    if args.output_file is not None:
        write_path = args.output_file
    out_df.to_csv(os.path.join(args.home_dir, 'test', fn))
    
    if args.output_file is not None:
        # we don't need extra aggregation for models that aren't evaluated in a prospective manner so we can quit.
        return
    
    
    if args.model_dir!='':
        # last day is the last
        # model dir date
        model_date = os.path.split(args.model_dir)[1].replace('_', '-')
        test_date = os.path.split(os.path.split(args.home_dir)[0])[1].replace('_', '-')
        with open(os.path.join(args.home_dir, '..', 'all_grud_auc'+arg_suf+'.csv'), 'a+') as f:
            f.write(args.model_dir + ', ' + args.home_dir +', '+ str(last_day) + ', ' + str(auc) + ', ' + str(aps) + '\n')
    else:
        if not(args.eval_train):
            with open(os.path.join(args.home_dir, '..', 'all_grud_auc'+arg_suf+'.csv'), 'a+') as f:
                f.write(str(last_day) + ', ' + str(auc) + ', ' + str(aps) + '\n')
        else:
            with open(os.path.join(args.home_dir, '..', 'all_grud_train_auc'+arg_suf+'.csv'), 'a+') as f:
                f.write(str(last_day) + ', ' + str(auc) + ', ' + str(aps) + '\n')
                
    print('Successfully evaluated')


    return None


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--target', type=str, default=('ili',), nargs='+', choices=['ili', 'ili_24', 'ili_48', 'covid', 'covid_24', 'covid_48', 'symptoms__fever__fever'], help='The target')
    parser.add_argument("--home_dir", type=str, required=True, help='Base directory for reading data and fit model. Assumed home_dir will contain an out folder, which contains the fit model on the training part of the split and the scaler used, and the test folder, which contains the test part of the split.')
    parser.add_argument("--model_dir", type=str, default='', help='Add this if the model is in a different folder.')
    parser.add_argument("--data_dir", type=str, default=None, help='If evaluating on the dataset in home_dir/test is not wanted then an alternative directory can be supplied here to override that default behaviour.')
    parser.add_argument("--output_file", type=str, default=None, help='If writing the results to the file home_dir/test/args.target[0]grud_prosp_testset_results.csv is not wanted then an alternative directory can be supplied to override the output location.')
    parser.add_argument("--regularly_sampled", action='store_true', help="Set this flag to have regularly sampled data rather than irregularly sampled data.")
    parser.add_argument('--num_dataloader_workers', type=int, default=4, help='# dataloader workers.')
    parser.add_argument('--test_size', type=int, default=7, help='# of days in test set which were not in the training data')
    parser.add_argument('--max_seq_len', type=int, default=48, help='maximum number of timepoints to feed into the model, required for load_data.')
    parser.add_argument('--zscore', action='store_true', help='Set this flag to train a model using the z score (assume forward fill imputation for missing data)')
    parser.add_argument('--weekofyear', action='store_true', help='Use the week_of_year feature when fitting testing the xgb model.')
    parser.add_argument("--num_feature", type=int, default=48, help="number of features passed to the model.")
    parser.add_argument("--grud_hidden", type=int, default=67, help="learning rate for the model")
    parser.add_argument('--opt_level', type=str, default='O1', choices=['O0', 'O1'], help='The model to train.')
    parser.add_argument("--only_healthy", action='store_true', help='Set this flag to train the model on only healthy measurements before the first onset of the target illness')
    parser.add_argument("--feat_subset", action='store_true', help='in the measurement and measurement_z dataframes only use the subset of features found to work better for xgboost and ridge regression')
    parser.add_argument("--eval_train", action='store_true', help='Set this flag to get the performance on the training set.')
    parser.add_argument("--no_imputation", action='store_true', help='If set use the data prior to imputation preserving missing values as NA.')
    parser.add_argument("--out_suf", default='', type=str, help='Suffix to append (before the extension) to each output file name.')
    args = parser.parse_args()
    
    print(vars(args))
    with open(os.path.join(args.home_dir, 'test', 'args.json'), 'w') as f:
        f.write(json.dumps(vars(args)))

    main(args)


