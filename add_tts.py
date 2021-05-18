import os
import pandas as pd
import numpy as np

from run_model import *


def helper(df): 
    #wrong_participants = ['39CcAJWhK1LDDnIu', '8dXL0bfaVtFGx2cX', 'fy1qZdfxyEvSi4tj', 'vGFHdB1ayjlv3afZ', '1EtZ1AgcoUAOvO5v', '3IO6GOSzXOeBmIb9', 'CpHtiKA0o7mH6gmw', 'h6QN6B1vPl0WBE3v', 'tL9G199wbLfhsB3D', 'ueuCKTU0gAgFA43z', 'TNyRY7Pi2qAkeLpr', 'uGGw2PsHOb1uHno7', '8IJD1b3l2Fwa9sYk', 'PzeWccMquVOY00N7', 'crshozeKvws2puc5', '3csbpZFoh3dEXowT', 'CSJawtUDmpFQptZM', 'HvNHcBmFijU2FhN7', 'ISbjGEvNI5XyqYzI', 'KE7q9nNvMczrQYff', 'Q3dKpjVedqr4pwJB', 'TMJdC9p1mCM3l2Fq', 'W2BTo2hbEemRPiQY', 'W2BTo2hbEemRPiQY', '318cUncDoixEISgY', '7gVnCVa11udZbSmO', 'LIw5vxCQ53o6kqrd', 'nZnclP1KbKbjLysA', 'uBOferp6vZilqh4L', 'IWWfYriuIXLTLEk6', 'MZUQV3XOriJPaUWC', 'YWuuYjBGKu5lRXJd', 'aBZg1bu4mT5qYKcD', 'ek0y7uvWe7tEzlR7', 'vEwrmojJN1KHwrMX', '3Ha8OemqTFHWk3av', '92gNEGhSIh9dDhEo', 'HOZR7YsPmeHibfbC', 'uFd1rcOMc6r87Mge', '8PMhJnNhraHnFEhN', 'Bb3HYj1UNfzrBzRq', 'XEUmMm9rnAVoMjEy', '0HlcBVhJywSdvvvg', 'OQaduBvx3JZ96fE7', 'QOkgB1GBLNRATBx1', '7cAyneMscf8UwLUt', 'EzcVN9UVgdSlMPoi', 'Xy2PeKiA1EILQSm0', 'ZgC9ENUzhnKqxK8i', 'xAlbQ915e8QKxPQq', 'PQ4aj49XYOpxYYI2', 'LpBZFkJ52RBkPkv6', 'S6wK4KOyoElCDzlz', 'YwySpMmBelV9lial', 'an1AkvjauWCPnzFN', 'HAcIsOQRe167qNYp', 'suVFQsb7s3ZqsTLz', 'tOyA11cFfZZ5JqaK', '7IjNducOYJzLOfMP', 'TxnMD2jVj0GtfYaw', 'hU1LKPHHCWtlmu1T']

#    import pdb
#    pdb.set_trace()

    helper.count += 1
    survey_dates = df['time_to_survey'][df['time_to_survey'] == 0].reset_index()
    
    survey_dates = (survey_dates['date'] + pd.to_timedelta(1, 'd'))
    
    tts = []
    for s_date in survey_dates:
        tmp = (df.index.get_level_values('date') - s_date)/pd.to_timedelta(1, 'd')
        tmp = np.where(tmp > 0, -np.inf, tmp)
        tts.append(tmp)
    
    if len(tts) == 0:
        helper.nosurvey_count += 1
        helper.nosurvey.append(df.index.get_level_values('participant_id')[0])
        return None
    tts = np.abs(np.stack(tts).max(axis=0))
    
    if helper.count % 1000 == 0:
        print(helper.count, flush=True)

    df['tts'] = tts

    return df
helper.count = 0
helper.nosurvey_count = 0
helper.nosurvey = []


def get_tts(dfs):
    """
    dfs: dictionary of dataframes as used in the project.
    """

    assert 'time_to_survey' in dfs['survey'].columns, 'time_to_survey not a column in the survey dataframe'
    
    out_df = dfs['survey'][['time_to_survey']].groupby('participant_id').apply(helper)

    return out_df[['tts']]


def main(args):

    dfs=load_data(args.data_dir, regular=args.regular_sampling)
    
    out_df = get_tts(dfs)
    
    fname = 'all_daily_data_' + 'regular'*args.regular_sampling +'irregular'*(not(args.regular_sampling)) + '.hdf'

    out_df.to_hdf(os.path.join(args.output_dir, fname), 'survey')
    

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--output_dir", type=str, required=True, help='save dir.')
    parser.add_argument('--data_dir', type=str, required=True, help='Explicit dataset path (else use rotation).')
    parser.add_argument('--seed', type=int, default=0)
    
    parser.add_argument("--regular_sampling", action='store_true', help="Set this flag to have regularly sampled data rather than irregularly sampled data.")
    parser.add_argument("--zscore", action='store_true', help="Set this flag to train a model using the z score (assume forward fill imputation for missing data)")
    
    args = parser.parse_args()
    
    print(vars(args))
    if not (os.path.exists(args.output_dir)):
        os.mkdir(args.output_dir)
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
              f.write(json.dumps(vars(args)))
    
    main(args)
    
