import os
import numpy as np
import pandas as pd
from GRUD import GRUD
import sklearn.metrics as skm

from run_model import *

import pickle as pkl

def main(args):
    
    arg_suf = args.out_suf

    device='cuda' if torch.cuda.is_available() else 'cpu'
    
    # Store the set of suffixes attached to the model output folder in home_dir
    home_suffix = 'out_grud' + arg_suf

    assert os.path.exists(os.path.join(args.home_dir, 'out_grud', 'train_means.pickle')), 'No train_means file to load in ' + str(args.home_dir) + '/out_grud'

    with open(os.path.join(args.home_dir, 'out_grud', 'train_means.pickle'), 'rb') as f:
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

    # Assumed test part of split in sub dir test
    if os.path.exists(os.path.join(args.home_dir, 'test')):
        try:
            # Try to read in the dictionary detailing the retrospective data
            d_name = 'split_dict_' + args.regular_sampling*'regular' + (not args.regular_sampling)*'irregular' + '.pkl'
            with open(os.path.join(args.home_dir,'test', d_name), 'rb') as f:
                retro_specs = pickle.load(f)
            # If the load hasn't failed we know that the dictionary file exists, check for a full_dataframe file, if both exist raise an error because it's unclear what the program should have done.
            f_name = 'split_daily_data_' + args.regular_sampling*'regular' + (not args.regular_sampling)*'irregular' + '.hdf'
            assert not os.path.exists(os.path.join(args.home_dir, 'test',  f_name)), 'Found a retrospective dictionary file and full dataframe file in {}, uncertain which should be used.'.format(args.data_dir)
            dfs=load_data(args.data_dir, regular=args.regular_sampling,  load_activity=False, all_survey=True, fname='all_daily_data_allsurvey_irregular_merged_nov16.hdf')
            dfs = get_retro(dfs, retro_specs)
        except OSError:
            dfs=load_data(os.path.join(args.home_dir, 'test'), regular=args.regular_sampling)
    else:
        #assume that the test participants in grud are correct.
        dfs=dfs=load_data(args.data_dir, regular=args.regular_sampling, only_healthy=args.only_healthy)
        # subset to the test participants
        #todo use args to subset.
        participant_ili_tuples = dfs['activity'].index.tolist() #get_latest_event_onset(dfs)
            
        # if test participants are defined get them
        if args.test_start_date is None:
            raise Exception("No test dates specified")
        else:
            test_participants = [p for p, d in participant_ili_tuples if (d>=args.test_start_date)&(d<=args.test_end_date)]
        # if validation participants are defined
        valid_participants = [p for p, d in participant_ili_tuples if (d>=args.validation_start_date)&(d<=args.validation_end_date)]
        if args.train_start_date is None:
            train_participants = [p for p, d in participant_ili_tuples if (d<=args.train_end_date)]
        else:
            train_participants = [p for p, d in participant_ili_tuples if (d>=args.train_start_date)&(d<=args.train_end_date)]
        
    if args.weekofyear:
        idx=pd.IndexSlice
        dfs['activity']['weekday']=(dfs['activity'].index.get_level_values('date').weekday<5).astype(np.int32) # add week of year
        dfs['activity'].loc[:, idx['measurement', 'weekofyear']]=dfs['activity'].index.get_level_values('date').isocalendar().week.astype(np.int32)
        dfs['activity'].loc[:, idx['measurement_z', 'weekofyear']]=dfs['activity'].index.get_level_values('date').isocalendar().week.astype(np.int32)
        dfs['activity'].loc[:, idx['mask', 'weekofyear']]=np.ones(len(dfs['activity']))
        dfs['activity'].loc[:, idx['time', 'weekofyear']]=np.ones(len(dfs['activity']))
        
        
    # subset dfs to validation and test
    
    valid_dfs = {k:v.loc[v.index.get_level_values('participant_id').isin(valid_participants), :] for k, v in dfs.items()}
    test_dfs = {k:v.loc[v.index.get_level_values('participant_id').isin(test_participants), :] for k, v in dfs.items()}
    



    
        
    if not(args.zscore):
        # zscore is already normalised, so we do not need to apply the scalar
        # Assumed data scaler in sub dir out.
        with open(os.path.join(args.home_dir, 'out_grud', 'scaler.pkl'), 'rb') as f:
            scaler = pkl.load(f)
        print('apply scaler')
        test_dfs, _ = apply_standard_scaler(test_dfs, scaler=scaler)
    
    valid_dataset=ILIDataset(valid_dfs, args, full_sequence=True, feat_subset=args.feat_subset)
    valid_dataloader=DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=args.num_dataloader_workers, collate_fn=id_collate)
    
    test_dataset=ILIDataset(test_dfs, args, full_sequence=True, feat_subset=args.feat_subset)
    test_dataloader=DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_dataloader_workers, collate_fn=id_collate)
    
    
    # *********************** first do validation ****************************
    proba_projection=torch.nn.Softmax(dim=1)
    labels=[]
    scores = []
    participants = []
    model.eval()
    batches=tqdm(valid_dataloader, total=len(valid_dataloader))
    for batch in batches:
        assert valid_dataloader.batch_size == 1, "batch size for test_dataloader must be one."
        participant = batch[-1][0]
        batch = batch[0][0]
        batch={k:v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            prediction, hidden_states = model(batch['measurement_z'].float() if args.zscore else batch['measurement'].float(), batch['measurement'].float(), batch['mask'].float(), batch['time'].float(), pad_mask= batch['obs_mask'], return_hidden=True)

            scores.append(proba_projection(prediction.view(-1, 2))[:, -1].detach().cpu().numpy()[-args.test_size:])
            labels.append(batch[args.target[0]].detach().cpu().numpy()[:, -args.test_size:])
        participants.append([participant]*scores[-1].shape[-1]) 

    labels = np.concatenate(tuple(labels), axis=1).squeeze()
    scores = np.concatenate(tuple(scores))
    
    dates = valid_dataloader.Dataset.survey.index.get_level_values('date').tolist()
    
    participants2 = valid_dataloader.Dataset.survey.index.get_level_values('participant_id').tolist()
    
         
    participants = np.concatenate(tuple(participants))
    assert(all([item[0]==item[1] for item in zip(participants, participants2)]))
    
    # subset dates just to those in the validation week.
    out_df = pd.DataFrame({args.target[0] + '_score': scores, args.target[0]: labels, 'participant_id': participants, 'date': dates})
    out_df['date'] = pd.to_datetime(out_df['date'])
    
    out_df = out_df.loc[(out_df['date']>=args.validation_start_date)&(out_df['date']<=args.validation_end_date), :]
    
    
    assert os.path.exists( os.path.join( args.home_dir,'out_grud' + arg_suf, 'thresholds_activity.json')), print(os.path.join( args.home_dir,'out_grud' + arg_suf, 'thresholds_activity.json'))
    
    if os.path.exists( os.path.join( args.home_dir,'out_grud' + arg_suf, 'thresholds_activity.json')):
        with open(os.path.join( args.home_dir,'out_grud' + arg_suf, 'thresholds_activity.json'), 'r') as f:
            thresholds = json.load(f)
        # apply this to the data and get the scores.
        for k, v in thresholds.items():
            print(args.target[0])
            out_df[args.target[0]+'_pred_'+k] = np.asarray(scores >= float(v) ).astype(np.int32)

    auc = skm.roc_auc_score(out_df[args.target[0]].values, out_df[args.target[0]+'_score'].values)
    aps = skm.average_precision_score(out_df[args.target[0]].values, out_df[args.target[0]+'_score'].values)

    args.output_dir = args.home_dir

    
    fn = args.target[0] + 'grud_prosp_validset_results'+ arg_suf +'.csv'
    if not os.path.exists(os.path.join(args.home_dir, 'test')):
           os.mkdir(os.path.join(args.home_dir, 'test'))
    out_df.to_csv(os.path.join(args.home_dir, 'test', fn))

    with open(os.path.join(args.home_dir, '..', 'all_grud_auc.csv'), 'a+') as f:
        f.write(str(last_day) + ', ' + str(auc) + ', ' + str(aps) + '\n')
           
    # choose two thresholds now
    threshold={}
    # subset dates to just those in the prospective week
    
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(out_df[args.target[0]].values, out_df[args.target[0]+'_score'].values)
    
    # 98% specificity
    target_fpr = min(fpr[fpr>=0.98])
    # index of target fpr
    index = list(fpr).index(target_fpr)
    
    threshold['98_spec'] = thresholds[index]
    
    # 98% sensitivity
    target_tpr = min(tpr[tpr>=0.98])
    # index of target fpr
    index = list(tpr).index(target_tpr)
    
    threshold['98_sens'] = thresholds[index]
    
    # threshold to json
    with open(os.path.join(args.home_dir, 'test', 'thresholds.json'), 'w') as f:
        f.write(json.dumps(threshold))
    
    
    
    # *********************** Test set ***************************************
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
        with torch.no_grad():
            prediction, hidden_states = model(batch['measurement_z'].float() if args.zscore else batch['measurement'].float(), batch['measurement'].float(), batch['mask'].float(), batch['time'].float(), pad_mask= batch['obs_mask'], return_hidden=True)

            scores.append(proba_projection(prediction.view(-1, 2))[:, -1].detach().cpu().numpy()[-args.test_size:])
            labels.append(batch[args.target[0]].detach().cpu().numpy()[:, -args.test_size:])
        participants.append([participant]*scores[-1].shape[-1]) 

    labels = np.concatenate(tuple(labels), axis=1).squeeze()
    scores = np.concatenate(tuple(scores))
    
    last_day = dfs['activity'].index.get_level_values('date').max()
    prospective_days = [last_day - pd.to_timedelta(args.test_size-1,unit='d') + pd.to_timedelta(n, unit='d') for n in range(args.test_size)]
    
    dates = test_dataloader.Dataset.survey.index.get_level_values('date').tolist()
    
    participants2 = test_dataloader.Dataset.survey.index.get_level_values('participant_id').tolist()
    
         
    participants = np.concatenate(tuple(participants))
    assert(all([item[0]==item[1] for item in zip(participants, participants2)]))
    
    # subset dates just to those in the test week.
    out_df = pd.DataFrame({args.target[0] + '_score': scores, args.target[0]: labels, 'participant_id': participants, 'date': dates})
    out_df['date'] = pd.to_datetime(out_df['date'])
    
    out_df = out_df.loc[(out_df['date']>=args.test_start_date)&(out_df['date']<=args.test_end_date), :]
    
    for k, v in threshold.items():
        # apply a prediction threshold to the data
        out_df['prediction_at_'+k]=np.asarray(out_df[args.target[0] + '_score'] >= v)

    auc = skm.roc_auc_score(out_df[args.target[0]].values, out_df[args.target[0]+'_score'].values)
    aps = skm.average_precision_score(out_df[args.target[0]].values, out_df[args.target[0]+'_score'].values)
    
    
    # 98_spec TPR, FPR
    tn, fp, fn, tp  = sklearn.metrics.confusion_matrix(out_df[args.target[0]].values, out_df[args.target[0]+'_score'].values>threshold['98_spec']).ravel()
    prevalence=np.mean(out_df[args.target[0]].values)
    tpr_at_98_spec = tp/(tp+fn)
    fpr_at_98_spec = fp/(fp+tn)
    #98_sens TPR, FPR
    tn, fp, fn, tp  = sklearn.metrics.confusion_matrix(out_df[args.target[0]].values, out_df[args.target[0]+'_score'].values>threshold['98_sens']).ravel()
    prevalence=np.mean(out_df[args.target[0]].values)
    tpr_at_98_sens = tp/(tp+fn)
    fpr_at_98_sens = fp/(fp+tn)

    args.output_dir = args.home_dir

    
    fn = args.target[0] + 'grud_prosp_testset_results.csv'
    if not os.path.exists(os.path.join(args.home_dir, 'test')):
           os.mkdir(os.path.join(args.home_dir, 'test'))
    
    out_df.to_csv(os.path.join(args.home_dir, 'test', fn))

    with open(os.path.join(args.home_dir, '..', 'all_grud_auc.csv'), 'a+') as f:
        f.write(str(last_day) + ', ' + str(auc) + ', ' + str(aps) + ', ' + str(tpr_at_98_spec) + ', ' + str(fpr_at_98_spec) +', ' + str(tpr_at_98_sens) + ', ' + str(fpr_at_98_sens) + ', ' + str(prevalence) + ', ' + str(len(out_df[args.target[0]].values)) + '\n')
           

    return None


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--target', type=str, default=('ili',), nargs='+', choices=['ili', 'ili_24', 'ili_48', 'covid', 'covid_24', 'covid_48', 'symptoms__fever__fever'], help='The target')
    parser.add_argument("--home_dir", type=str, required=True, help='Base directory for reading data and fit model. Assumed home_dir will contain an out folder, which contains the fit model on the training part of the split and the scaler used, and the test folder, which contains the test part of the split.')
    parser.add_argument('--data_dir', type=str, required=False, help='Explicit dataset path (else use rotation).')
    parser.add_argument("--regular_sampling", action='store_true', help="Set this flag to have regularly sampled data rather than irregularly sampled data.")
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
    # add dataset splitting functions to here:
    parser.add_argument("--train_start_date",  type=str, default='', help='start date for training data (yyyy-mm-dd)')
    parser.add_argument("--train_end_date",  type=str, default='', help='The last day of training data (inclusive)')
    parser.add_argument("--validation_start_date",  type=str, default='', help='start date for training data')
    parser.add_argument("--validation_end_date",  type=str, default='', help='The last day of validation data (inclusive)')
    parser.add_argument("--validation_set_len",  type=int, help='Alternatively provide the len of the desired validation set')
    parser.add_argument("--test_start_date",  type=str, default='', help='start date for test data')
    parser.add_argument("--test_end_date",  type=str, default='', help='The last day of test data (inclusive)')
    
    args = parser.parse_args()
    
    with open(os.path.join(args.home_dir, 'test', 'args.json'), 'w') as f:
        f.write(json.dumps(vars(args)))
    
    
    
    args.train_start_date= pd.to_datetime(args.train_start_date) if args.train_start_date!='' else None
    args.train_end_date= pd.to_datetime(args.train_end_date) if args.train_end_date!='' else None
#     print(args.train_start_date)
#     print(not(args.train_start_date is None))
    if not(args.train_start_date is None) and not(args.train_end_date is None):
        assert args.train_end_date>=args.train_start_date, "The train end date must be after the train start date"
        
        
    # validation
    args.validation_start_date= pd.to_datetime(args.validation_start_date) if args.validation_start_date!='' else None
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
    
    print(vars(args))
    

    main(args)


