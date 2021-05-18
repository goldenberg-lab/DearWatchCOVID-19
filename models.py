# import dependecies
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
try:
    from glmnet_python import glmnet
except:
    glmnet=None

import torch.utils.data as utils
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.distributions as distributions
import math

import pickle
import os
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import Ridge
import sklearn.metrics as skm
import pandas as pd
import time

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

import pdb

from sklearn.preprocessing import StandardScaler

class processor():
    def __init__(self):
        self.scaler = StandardScaler()
    def transform(self,X): 
        self.p = X.shape[1]
        # Add square and log to data
        print('Adding non-linear transforms')
        xx = np.hstack([X, X**2])
        print('Transforming data')
        if hasattr(self.scaler, 'mean_'):
            return self.scaler.transform(xx)
        else:
            return self.scaler.fit_transform(xx)


class LogisticRegression(nn.Module):
    """
    Prediction head for ICD10
    """
    def __init__(self, input_dim, dropout_prob=0.1):
        super(LogisticRegression, self).__init__()
        #predict next outputs
        self.fc1=nn.Linear(input_dim, 2)
        self.drop=nn.Dropout(p=dropout_prob)
    def forward(self,  X, X_last_obsv, Mask, Delta, pad_mask=None, return_hidden=False):
        if len(X_last_obsv.shape)==3:
            batch_size, seq_size, inp_dim= X_last_obsv.shape
            if return_hidden:
                return self.drop(self.fc1(X_last_obsv.view(-1, inp_dim))).view(batch_size, seq_size, -1), None
            return self.drop(self.fc1(X_last_obsv.view(-1, inp_dim))).view(batch_size, seq_size, -1)
        else:
#             batch_size, inp_dim= X_last_obsv.shape
            if return_hidden:
                return self.drop(self.fc1(X_last_obsv)), None
            else:
                return self.drop(self.fc1(X_last_obsv))


class prob_NN(nn.Module):
    """
    Convolve along time.
    """
    def __init__(self, input_dim, seq_len, num_feature=48, dropout_prob=0.1):
        super(prob_NN, self).__init__()
        #predict latest output, input_dim is all the data in the batch, so consider the final
        # set of observations as the labels so drop it right away.
        self.num_feature = num_feature
       
        self.sigma_size = (num_feature*(num_feature+1))//2
        self.fc1 = nn.Linear(num_feature*seq_len, self.sigma_size * 8)
        self.fc2 = nn.Linear(self.sigma_size * 8, self.sigma_size * 4)
        self.fc3 = nn.Linear(self.sigma_size * 4, self.sigma_size * 2)
        self.fc4 = nn.Linear(self.sigma_size *2, num_feature + self.sigma_size)
        self.drop= nn.Dropout(p=dropout_prob)
        

    def forward(self,  X, X_last_obsv, Mask, Delta, pad_mask=None, return_hidden=False):
        batch_size, seq_size, inp_dim= X.shape 
        # Assumes data input is of the form [patient, time, feature]
        # swap the time and feature dimension to convolve over time for each feature
        # Also drop the latest features since they will be predicted.
        X_in = X.permute([0,2,1])[:,:,:-1]

        H_fc1 = F.relu(self.fc1(X_in.reshape(-1, self.num_feature*(seq_size-1))))
        H_fc2 = self.drop(F.relu(self.fc2(H_fc1)))
        H_fc3 = self.drop(F.relu(self.fc3(H_fc2)))
        H_fc4 = self.fc4(H_fc3)
        #dist = distributions.multivariate_normal.MultivariateNormal(H_fc3[:, :48], H_fc3[:, 48:])
        mus = H_fc4[:, :self.num_feature]

        tril_indices = torch.tril_indices(row=self.num_feature, col=self.num_feature, offset=0)
        sigma = torch.zeros(batch_size, self.num_feature, self.num_feature)
        Var = torch.exp(H_fc4[:, self.num_feature:])**2
        sigma[:, tril_indices[0], tril_indices[1]] = Var
        
        if return_hidden:
            return [mus, sigma], H_fc4 #Is there a way I can use the returned hidden layer to check they are updating?
        
        return [mus, sigma]


class CNN1D(nn.Module):
    """
    Convolve along time.
    """
    def __init__(self, input_dim, seq_len, num_feature=48, dropout_prob=0.1):
        super(CNN1D, self).__init__()
        #predict latest output, input_dim is all the data in the batch, so consider the final
        # set of observations as the labels so drop it right away.
        self.ks_conv = 5
        self.ks_pool = 3
        self.stride_pool = 1
        self.num_feature = num_feature
        self.sigma_size = (num_feature*(num_feature + 1))//2

        self.conv_1 = nn.Conv1d(input_dim, input_dim, kernel_size = self.ks_conv, groups=num_feature)
        self.l_out_conv1 = math.floor(seq_len - self.ks_conv + 1)
        self.pool = nn.AvgPool1d(kernel_size=self.ks_pool, stride = self.stride_pool)
        self.l_out_pool = math.floor((self.l_out_conv1 - self.ks_pool)/self.stride_pool + 1)
        self.conv_2 = nn.Conv1d(input_dim, input_dim, kernel_size = self.ks_conv, groups=input_dim)
        self.l_out_conv2 = math.floor(self.l_out_pool - self.ks_conv + 1)
        self.fc1=nn.Linear(num_feature*self.l_out_conv2, self.sigma_size * 4)
        self.fc2 = nn.Linear(self.sigma_size * 4, self.sigma_size * 2)
        self.fc3 = nn.Linear(self.sigma_size * 2, num_feature + self.sigma_size)
        self.drop=nn.Dropout(p=dropout_prob)

    def forward(self,  X, X_last_obsv, Mask, Delta, pad_mask=None, return_hidden=False):
        batch_size, seq_size, inp_dim= X.shape
        
        # Assumes data input is of the form [patient, time, feature]
        # swap the time and feature dimension to convolve over time for each feature
        # Also drop the latest features since they will be predicted.
        X_in = X.permute([0,2,1])[:,:,:-1]

        H_conv_1 = self.pool(self.conv_1(X_in))
        H_conv_2 = self.conv_2(H_conv_1)
        H_fc1 = F.relu(self.fc1(H_conv_2.view(-1,self.num_feature*self.l_out_conv2)))
        H_fc2 = F.relu(self.fc2(H_fc1))
        H_fc3 = self.fc3(H_fc2)
        #dist = distributions.multivariate_normal.MultivariateNormal(H_fc3[:, :48], H_fc3[:, 48:])
        mus = H_fc3[:, :self.num_feature]

        tril_indices = torch.tril_indices(row=self.num_feature, col=self.num_feature, offset=0)
        sigma = torch.zeros(batch_size, self.num_feature, self.num_feature)
        Var = torch.exp(H_fc3[:, self.num_feature:])**2
        sigma[:, tril_indices[0], tril_indices[1]] = Var
        #H_drop = self.drop(sample)
        
        if return_hidden:
            return [mus, sigma], H_fc3 #Is there a way I can use the returned hidden layer to check they are updating?
        
        return [mus, sigma]
            
    
class LSTMmodel(nn.Module):
    """
    Prediction head for ICD10
    """
    def __init__(self, input_dim, hidden_dim, dropout_prob=0.1, bidirectional=False, num_layers=2):
        super(LSTMmodel, self).__init__()
        #predict next outputs
        
        self.lstm=nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, bidirectional=False, num_layers=num_layers,  batch_first=True, dropout=dropout_prob)
        
#         modules = [nn.LSTMCell(input_size=input_dim, hidden_size=hidden_dim),
#                                   nn.ReLU(),
#                                   nn.Dropout(p=dropout_prob)] +
#                     [nn.LSTMCell(input_size=hidden_dim, hidden_size=hidden_dim),
#                                   nn.ReLU(),
#                                   nn.Dropout(p=dropout_prob)]*(num_layers-1)
        
#         self.lstm = nn.ModuleList(*models)
        

        self.fc1=nn.Linear(hidden_dim, 2)
        
        self.num_layers=num_layers
        self.bidirectional = bidirectional
        self.hidden_dim=hidden_dim
    def forward(self,  X, X_last_obsv, Mask, Delta, pad_mask=None, return_hidden=False):
#         batch_size, seq_size, inp_dim= X_last_obsv.data.shape
        
#         if pad_mask is not None:
#             X_lengths= pad_mask.sum(dim=-1)
            
#             print(X_lengths)
            # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
#             X_last_obsv = torch.nn.utils.rnn.pack_padded_sequence(X_last_obsv, X_lengths, batch_first=True, enforce_sorted=False)

        # now run through LSTM
        outputs, h_c =  self.lstm(X_last_obsv)
        
        
#         if pad_mask is not None:
#             # undo the packing operation
#             outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        if return_hidden:
            return self.fc1(outputs.data), outputs
        else:
            return self.fc1(outputs.data)
            
#         if return_hidden:
#             return self.fc1(outputs.view(-1, self.hidden_dim).contiguous()).view(batch_size, seq_size, -1), outputs
        
#         return self.fc1(outputs.view(-1, self.hidden_dim).contiguous()).view(batch_size, seq_size, -1)

class GRUSIMPLEmodel(nn.Module):
    """
    Prediction head for ICD10
    """
    def __init__(self, input_dim, hidden_dim, dropout_prob=0.1, bidirectional=False, num_layers=2):
        super(GRUSIMPLEmodel, self).__init__()
        #predict next outputs
        
        self.gru=nn.GRU(input_size=input_dim*3, hidden_size=hidden_dim, bidirectional=False, num_layers=num_layers,  batch_first=True, dropout=dropout_prob)

        self.fc1=nn.Linear(hidden_dim, 2)
        
        self.num_layers=num_layers
        self.bidirectional = bidirectional
        self.hidden_dim=hidden_dim
    def forward(self,  X, X_last_obsv, Mask, Delta, pad_mask=None, return_hidden=False):
        
        print(X_last_obsv.data.shape)
        
#         input_to_model = X_last_obsv.clone()        
        X_last_obsv.data = torch.cat((X_last_obsv.data, Mask.data, Delta.data), dim=-1)
        print('input_shape: ', X_last_obsv.data.shape)
        print()
        outputs, h_c =  self.gru(X_last_obsv)
        

        if return_hidden:
            return self.fc1(outputs.data), outputs
        else:
            return self.fc1(outputs.data)
        
        
class GRUmodel(nn.Module):
    """
    Prediction head for ICD10
    """
    def __init__(self, input_dim, hidden_dim, dropout_prob=0.1, bidirectional=False, num_layers=2):
        super(GRUmodel, self).__init__()
        #predict next outputs
        
        self.gru=nn.GRU(input_size=input_dim, hidden_size=hidden_dim, bidirectional=False, num_layers=num_layers,  batch_first=True, dropout=dropout_prob)

        self.fc1=nn.Linear(hidden_dim, 2)
        
        self.num_layers=num_layers
        self.bidirectional = bidirectional
        self.hidden_dim=hidden_dim
    def forward(self,  X, X_last_obsv, Mask, Delta, pad_mask=None, return_hidden=False):
#         self.gru.flatten_parameters()
        outputs, h_c =  self.gru(X_last_obsv)
        

        if return_hidden:
            return self.fc1(outputs.data), outputs
        else:
            return self.fc1(outputs.data)

    
    
# class GRU(nn.Module):
#     """
#     """
#     def __init__(self, input_dim, hidden_dim, dropout_prob=0.1, bidirectional=False, num_layers=2):
#         super(GRU, self).__init__()
#         #predict next outputs
        
#         self.gru=nn.GRU(input_size=input_dim, hidden_size=hidden_dim, bidirectional=False, num_layers=num_layers,  batch_first=True, dropout=dropout_prob)
        

        

#         self.fc1=nn.Linear(hidden_dim, 2)
        
#         self.num_layers=num_layers
#         self.bidirectional = bidirectional
#         self.hidden_dim=hidden_dim
#     def forward(self,  X, X_last_obsv, Mask, Delta, pad_mask=None, return_hidden=False):
#         batch_size, seq_size, inp_dim= X_last_obsv.shape
        
#         if pad_mask is not None:
#             X_lengths= pad_mask.sum(dim=-1)

#             # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
#             X_last_obsv = torch.nn.utils.rnn.pack_padded_sequence(X_last_obsv, X_lengths, batch_first=True, enforce_sorted=False)

#         # now run through LSTM
#         outputs, h_c =  self.gru(X_last_obsv)
        
#         if pad_mask is not None:
#             # undo the packing operation
#             outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            
#         if return_hidden:
#             return self.fc1(outputs.view(-1, self.hidden_dim).contiguous()).view(batch_size, seq_size, -1), outputs
        
#         return self.fc1(outputs.view(-1, self.hidden_dim).contiguous()).view(batch_size, seq_size, -1)
    


class fastgam_mdl():
    def __init__(self, args):
        self.args = args
        self.isfit = False
        self.istuned = False
        self.enc = False
        # features Raghu chose as a subset of all features.
        self.use_features = ['heart_rate_bpm', 'walk_steps', 'sleep_seconds',
                             'steps__rolling_6_sum__max', 'steps__count', 
                             'steps__sum', 'steps__mvpa__sum', 
                             'steps__dec_time__max', 'heart_rate__perc_5th', 
                             'heart_rate__perc_50th', 'heart_rate__perc_95th', 
                             'heart_rate__mean', 'heart_rate__stddev', 
                             'active_fitbit__sum', 'active_fitbit__awake__sum',
                             'sleep__asleep__sum', 'sleep__main_efficiency', 
                             'sleep__nap_count', 'sleep__really_awake__mean',
                             'sleep__really_awake_regions__countDistinct', 
                             'weekday']

    def predict(self, data):
        assert self.isfit
        print('Getting X/y')
        X, y, cn, xidx = self.transform(data)
        eta = self.mdl.predict(X)
        return y, eta, xidx
    
    def save(self, folder=None):
        """
        pickle's object for later
        """
        if folder is None:
            folder = self.args.output_dir
        file_pckl = os.path.join(folder, 'GAM_mdl.pickle')
        with open(file_pckl, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)
        pickle_file.close()
        
    
    def load(self, folder=None):
        """
        path should point to pickle object
        """
        if folder is None:
            folder = self.args.output_dir
        file_pckl = os.path.join(folder, 'GAM_mdl.pickle')
        pickle_in = open(file_pckl,"rb")
        tmp = pickle.load(pickle_in)
        self.mdl = tmp.mdl
        self.normalizer = tmp.normalizer
        self.enc = tmp.enc
        self.discretizer = tmp.discretizer
        self.isfit = tmp.isfit
        
    def fit(self, data):
        print('---- FITTING MODEL ----')
        print('Getting X/y')
        X, y, cn, xidx = self.transform(data)

        print('# Fit the Ridge model')
        stime = time.time()
        self.mdl = Ridge(alpha=0.1).fit(X=X,y=y)
        etime = time.time()
        print('Took %i seconds to train Ridge model' % (etime - stime))
        self.isfit = True
    
    def transform(self, data):
        """
        Process_data for the ar_mdl, should store the dataframes in the 
        dictionary attributes separated as activity and survey
        Method will be called in fit, tune, and predict to process dataframes
        returns: X, y in a dataframe format
        """
        import warnings
        warnings.filterwarnings('ignore')
        # NUMBER OF BINS AND ORDER, RESPECTIVELY
        nb, nk = 10, 2
        idx = pd.IndexSlice
        survey, activity = data['survey'], data['activity']
        activity = activity.loc[:,idx['measurement',self.use_features]]
        activity.insert(0, ('idx','cidx'), activity.groupby('participant_id').cumcount().values+1)
        # Get the lags by ID...
        stime = time.time()
        print('Creating fast lags')
        out = fast_lags(activity, self.args.days_ago, diff=False)
        
        assert out.groupby('participant_id').tail(1).notnull().all().all()
        print('Create target dataframes')
        target = pd.DataFrame(survey.loc[:,self.args.target[0]].astype(int))
        target.columns = pd.MultiIndex.from_tuples([['label',self.args.target[0]]],names=['df_type', 'value'])
        print('Merging')
        out = out.join(target)
    
        if self.args.forecast_type == 'timetoevent':
            print('Subetting for time2event')
            out = out.join(out.assign(cidx=out.groupby('participant_id').cumcount()).loc[out[('label',self.args.target[0])]==1].groupby('participant_id').head(1)[['cidx']].droplevel(1))
            out = out[out.groupby('participant_id').cumcount() <= out.cidx].drop(columns=['cidx'])
            check = np.all( out[('label',self.args.target[0])].groupby('participant_id').sum() == 1 )
            print('t2e subset worked: %s' % check)
            assert check
        
        # Remove any person that does have at least lags+1
        nids = out.index.get_level_values(0).value_counts()
        out = out[out.index.get_level_values(0).isin(idx[nids[nids >= self.args.days_ago + 1].index])]
        # Fill missing (because last row has no missing values, groupby is not necessay)
        out = out.fillna(0)
        assert out.groupby('participant_id').tail(1).notnull().all().all()
        # Extract the label
        y = out[('label',self.args.target[0])].values
        X = out.drop(columns=[('label',self.args.target[0])])
        cn = X.columns
        xidx = X.index
        X = X.values
        n, p = X.shape
        if not self.enc:
            print('Training discretizer for the first time')
            # Fit encoder for every column of X
            self.discretizer = [KBinsDiscretizer(n_bins=nb).fit(X[:,[j]]) for j in range(p)]
            Xgam = [self.discretizer[j].transform(X[:,[j]]) for j in range(p)] # encode knots
            Xgam = [np.tile(X[:,[j]],[1,Xgam[j].shape[1]]) * Xgam[j].toarray() for j in range(p)] # zero if outside of knots
            # Add intercept
            Xgam = [np.c_[np.ones(n),Xgam[j]] for j in range(p)]
            # Combine order 1 and order 2
            Xgam = np.c_[np.hstack(Xgam), np.hstack([Xgam[j][:,1:]**2 for j in range(p)])]
            print('There are %i features after %i-order expansion' % (Xgam.shape[1], nk))
            print('Training standard scaler for the first time')
            self.normalizer = StandardScaler().fit(Xgam)
            self.enc = True
        else:
            print('discretizer and scaler already trained')
            Xgam = [self.discretizer[j].transform(X[:,[j]]) for j in range(p)] # encode knots
            Xgam = [np.tile(X[:,[j]],[1,Xgam[j].shape[1]]) * Xgam[j].toarray() for j in range(p)] # zero if outside of knots
            Xgam = [np.c_[np.ones(n),Xgam[j]] for j in range(p)]
            Xgam = np.c_[np.hstack(Xgam), np.hstack([Xgam[j][:,1:]**2 for j in range(p)])]
        print('Normalize features')
        Xgam = self.normalizer.transform(Xgam)
        
        return Xgam, y, cn, xidx        
    
class ar_mdl():
    def __init__(self, args, nlambda):
        self.args = args
        self.isfit = False
        self.glmnet_obj = None
        self.nlambda = nlambda
        self.bhat_star = None
        self.ahat_star = None
        # features Raghu chose as a subset of all features.
        self.use_features = ['heart_rate_bpm', 'walk_steps', 'sleep_seconds',
                             'steps__rolling_6_sum__max', 'steps__count', 
                             'steps__sum', 'steps__mvpa__sum', 
                             'steps__dec_time__max', 'heart_rate__perc_5th', 
                             'heart_rate__perc_50th', 'heart_rate__perc_95th', 
                             'heart_rate__mean', 'heart_rate__stddev', 
                             'active_fitbit__sum', 'active_fitbit__awake__sum',
                             'sleep__asleep__sum', 'sleep__main_efficiency', 
                             'sleep__nap_count', 'sleep__really_awake__mean',
                             'sleep__really_awake_regions__countDistinct', 
                             'weekday']

    def train(self, X_train, y_train, X_valid, y_valid):
        self.glmnet_obj = glmnet(x=X_train, y=np.atleast_2d(y_train.astype(float)).T, family='gaussian', standardize=False, nlambda=self.nlambda)
        all_val_auc = []
        for jj in range(self.nlambda):
            eta_jj = X_valid.dot(self.glmnet_obj['beta'][:, jj]) + self.glmnet_obj['a0'][jj]
            auc_jj = skm.roc_auc_score(y_valid, eta_jj)
            all_val_auc.append(auc_jj)
        lam_results = pd.DataFrame({'jj':range(self.nlambda),'lam':self.glmnet_obj['lambdau'], 'auc':all_val_auc, 'dof':self.glmnet_obj['df']})
        lam_results = lam_results[lam_results['auc'] > 0.5].sort_values('lam').reset_index(None, True)
        lam_results = lam_results.assign(dauc=lambda x: x['auc'] - x['auc'].max(), ddof=lambda x: x['dof'] - x['dof'].max())
        lam_results.to_csv(os.path.join(self.args.output_dir, 'df_lam.csv'), index=False)
        self.lam_results = lam_results

        # Pick the lambda within 0.5% of the max AUC
        jj_star = lam_results.loc[(lam_results['dauc'] > -0.005)].sort_values('ddof')['jj'].values[0]
        self.bhat_star = self.glmnet_obj['beta'][:, jj_star]
        self.ahat_star = self.glmnet_obj['a0'][jj_star]
        self.isfit = True


    def predict(self, X):
        assert self.isfit, 'The ar_mdl has not been fit yet. Train the model before predicting on new data.'
        return X.dot(self.bhat_star) + self.ahat_star






