"""
GRUD.py
utilities for GRU-D on MIMIC-III
"""


# import dependecies
import pandas as pd
import numpy as np
import os
import glob
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch.utils.data as utils
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math
import time



class FilterLinear(nn.Module):
    """
    As seen in https://github.com/zhiyongc/GRU-D/
    """
    def __init__(self, in_features, out_features, filter_square_matrix, bias=True):
        '''
        filter_square_matrix : filter square matrix, whose each elements is 0 or 1.
        '''
        super(FilterLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        use_gpu = torch.cuda.is_available()
        self.filter_square_matrix = None
        if use_gpu:
            self.filter_square_matrix = Variable(filter_square_matrix.cuda(), requires_grad=False)
        else:
            self.filter_square_matrix = Variable(filter_square_matrix, requires_grad=False)

        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
#         print(self.weight.data)
#         print(self.bias.data)

    def forward(self, input):
#         print(self.filter_square_matrix.mul(self.weight))
        return F.linear(input, self.filter_square_matrix.mul(self.weight), self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) + ')'

class GRUD(nn.Module):
    def __init__(self, input_size, cell_size, hidden_size, X_mean, device, output_last = False, fp16=False):
        """
        With minor modifications from https://github.com/zhiyongc/GRU-D/
        Recurrent Neural Networks for Multivariate Times Series with Missing Values
        GRU-D: GRU exploit two representations of informative missingness patterns, i.e., masking and time interval.
        cell_size is the size of cell_state.
        Implemented based on the paper:
        @article{che2018recurrent,
          title={Recurrent neural networks for multivariate time series with missing values},
          author={Che, Zhengping and Purushotham, Sanjay and Cho, Kyunghyun and Sontag, David and Liu, Yan},
          journal={Scientific reports},
          volume={8},
          number={1},
          pages={6085},
          year={2018},
          publisher={Nature Publishing Group}
        }
        GRU-D:
            input_size: variable dimension of each time
            hidden_size: dimension of hidden_state
            mask_size: dimension of masking vector
            X_mean: the mean of the historical input data
        """

        super(GRUD, self).__init__()

        self.hidden_size = hidden_size
        self.delta_size = input_size
        self.mask_size = input_size

        # use_gpu = torch.cuda.is_available()
        # if use_gpu:
        #     self.identity = torch.eye(input_size).cuda()
        #     self.zeros = Variable(torch.zeros(input_size).cuda())
        #     self.zeros_h = Variable(torch.zeros(self.hidden_size).cuda())
        #     self.X_mean = Variable(torch.Tensor(X_mean).cuda())
        # else:
        
        self.identity = torch.eye(input_size).to(device)
        self.zeros = Variable(torch.zeros(input_size)).to(device)
        self.zeros_h = Variable(torch.zeros(self.hidden_size)).to(device)
        self.X_mean = Variable(torch.Tensor(X_mean)).to(device)
        
        if fp16=='True':
            self.identity.half()
            self.zeros.half()
            self.zeros_h.half()
            self.X_mean = self.X_mean.half()
            

        self.zl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size) # Wz, Uz are part of the same network. the bias is bz
        self.rl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size) # Wr, Ur are part of the same network. the bias is br
        self.hl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size) # W, U are part of the same network. the bias is b

        self.gamma_x_l = FilterLinear(self.delta_size, self.delta_size, self.identity)

        self.gamma_h_l = nn.Linear(self.delta_size, self.hidden_size) # this was wrong in available version. remember to raise the issue

        self.output_last = output_last

        self.fc = nn.Linear(self.hidden_size, 2)

    def step(self, x, x_last_obsv, x_mean, h, mask, delta):
        """
        Inputs:
            x: input tensor
            x_last_obsv: input tensor with forward fill applied
            x_mean: the mean of each feature
            h: the hidden state of the network
            mask: the mask of whether or not the current value is observed
            delta: the tensor indicating the number of steps since the last time a feature was observed.
        Returns:
            h: the updated hidden state of the network
        """

        batch_size = x.shape[0]
        dim_size = x.shape[1]
        
#         print(self.zeros.dtype, delta.dtype)
#         print(self.gamma_x_l(delta).dtype)

        delta_x = torch.exp(-torch.max(self.zeros, self.gamma_x_l(delta))) #exponentiated negative rectifier
        delta_h = torch.exp(-torch.max(self.zeros_h, self.gamma_h_l(delta))) #self.zeros became self.zeros_h to accomodate hidden size != input size

        # print(x.shape) # 1, 533
        # print(x_mean.shape)
        # print(delta_x.shape)
        # print(x_last_obsv.shape)

        x = mask * x + (1 - mask) * (delta_x * x_last_obsv + (1 - delta_x) * x_mean)
        h = delta_h * h

        # print(x.shape) #534, 533
        # print(h.shape) # 1,67
        # print(mask.shape) # 1,533

        combined = torch.cat((x, h, mask), 1)
        z = torch.sigmoid(self.zl(combined)) #sigmoid(W_z*x_t + U_z*h_{t-1} + V_z*m_t + bz)
        r = torch.sigmoid(self.rl(combined)) #sigmoid(W_r*x_t + U_r*h_{t-1} + V_r*m_t + br)

        # print(x.shape, (r*h).shape, mask.shape)
        new_combined=torch.cat((x, r*h, mask), 1)
        h_tilde = torch.tanh(self.hl(new_combined)) #tanh(W*x_t +U(r_t*h_{t-1}) + V*m_t) + b
        # h_tilde = torch.tanh(self.hl(combined)) #tanh(W*x_t +U(r_t*h_{t-1}) + V*m_t) + b
        h = (1 - z) * h + z * h_tilde

        return h

    def forward(self, X, X_last_obsv, Mask, Delta, pad_mask=None, return_hidden=False):
        batch_size = X.size(0)
#         type_size = input.size(1)
        step_size = X.size(1) # num timepoints
        spatial_size = X.size(2) # num features

        Hidden_State = self.initHidden(batch_size)
#         X = torch.squeeze(input[:,0,:,:])
#         X_last_obsv = torch.squeeze(input[:,1,:,:])
#         Mask = torch.squeeze(input[:,2,:,:])
#         Delta = torch.squeeze(input[:,3,:,:])

        if pad_mask is not None:
            pass

        outputs = None
        for i in range(step_size):
            Hidden_State = self.step(torch.squeeze(X[:,i:i+1,:], 1)\
                                     , torch.squeeze(X_last_obsv[:,i:i+1,:], 1)\
                                     , torch.unsqueeze(self.X_mean, 0)\
                                     , Hidden_State\
                                     , torch.squeeze(Mask[:,i:i+1,:], 1)\
                                     , torch.squeeze(Delta[:,i:i+1,:], 1))
            if outputs is None:
                outputs = Hidden_State.unsqueeze(1)
            else:
                # # this makes the sequence reversed
                # outputs = torch.cat((Hidden_State.unsqueeze(1), outputs), 1)
                #this preserves the order
                outputs = torch.cat((outputs, Hidden_State.unsqueeze(1)), 1)
        # print(outputs.shape)
        if True:
            #binary outcomes for all states
            if return_hidden:
                # print(self.fc(torch.squeeze(outputs, 0)).shape)
                return self.fc(torch.squeeze(outputs, 0)), outputs
            else:
                # print(self.fc(torch.squeeze(outputs, 0)).shape)
                return self.fc(torch.squeeze(outputs, 0))
        # we want to predict a binary outcome
        else:
            # binary outcome for last state
            return self.fc(Hidden_State)

#         if self.output_last:
#             return outputs[:,-1,:]
#         else:
#             return outputs

    def initHidden(self, batch_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Parameter(Variable(torch.zeros(batch_size, self.hidden_size).cuda()))
            return Hidden_State
        else:
            Hidden_State = Parameter(Variable(torch.zeros(batch_size, self.hidden_size)))
            return Hidden_State

class next_measurement(nn.Module):
    """
    predict the next value
    """
    def __init__(self, feat_dim, hidden_dim):
        super(next_measurement, self).__init__()
        #predict next outputs
        self.fc1=nn.Linear(hidden_dim, 2048)
        self.fc2=nn.Linear(2048, 2048)
        self.fc3=nn.Linear(2048, feat_dim)

        self.drop=nn.Dropout(p=0.1)
        self.rl=nn.ReLU()

    def forward(self, x):

        x=self.drop(x)
        x=self.rl(self.fc1(x))
        x=self.rl(self.fc2(x))
        x=self.rl(self.fc3(x))

        return x


class icd10_head(nn.Module):
    """
    Prediction head for ICD10
    """
    def __init__(self, icd10_dim, hidden_dim):
        super(icd10_head, self).__init__()
        #predict next outputs
        self.fc1=nn.Linear(hidden_dim, 2048)
        self.fc2=nn.Linear(2048, 2048)
        self.fc3=nn.Linear(2048, icd10_dim)

        self.drop=nn.Dropout(p=0.1)
        self.rl=nn.ReLU()
    def forward(self, x):
        x=self.drop(x)
        x=self.rl(self.fc1(x))
        x=self.rl(self.fc2(x))
        x=self.rl(self.fc3(x))

        return x




def train_GRUD(model, train_dataloader, val_dataloader, num_epochs = 300, patience = 3, min_delta = 0.00001, weight=None, multitask=False, multitask_weights=None,):
    """
    """
    from tqdm import tqdm
    from sklearn.metrics import roc_auc_score

    print('Model Structure: ', model)
    print('Start Training ... ')

    device = torch.device("cuda" if torch.cuda.is_available() and not False else "cpu")
    n_gpu = torch.cuda.device_count()

    if multitask:
        _, _, _, _, _, icd10, sepsis, resp=next(iter(train_dataloader))
        try:
            hidden_dim=model.hidden_size
        except:
            hidden_dim=model.module.hidden_size
        icd_code_head=icd10_head(icd10.shape[-1], hidden_dim)
        icd_code_head.to(device)
        if n_gpu>1:
            icd_code_head = torch.nn.DataParallel(icd_code_head)

        sepsis_head=icd10_head(2, hidden_dim)
        sepsis_head.to(device)
        if n_gpu>1:
            sepsis_head = torch.nn.DataParallel(sepsis_head)

        resp_head=icd10_head(2, hidden_dim)
        resp_head.to(device)
        if n_gpu>1:
            resp_head = torch.nn.DataParallel(resp_head)

        # soft=nn.Sigmoid()



    # if (type(model) == nn.modules.container.Sequential):
    #     output_last = model[-1].output_last
    #     print('Output type dermined by the last layer')
    # else:
    #     output_last = model.output_last
    #     print('Output type dermined by the model')
    try:
        output_last = model.output_last
    except:
        #data parallel
        output_last = model.module.output_last

    if weight is not None:
        weight=weight.float().to(device)

    # loss_MSE = torch.nn.MSELoss()
    # loss_nll=torch.nn.NLLLoss()
    loss_CEL=torch.nn.CrossEntropyLoss(reduction='mean', weight=weight)
    if multitask:
        loss_icd=torch.nn.MultiLabelSoftMarginLoss(reduction ='mean', weight=multitask_weights[0])
        loss_sepsis=torch.nn.CrossEntropyLoss(reduction='mean', weight=multitask_weights[1])
        loss_resp=torch.nn.CrossEntropyLoss(reduction='mean', weight=multitask_weights[2])
    # loss_L1 = torch.nn.L1Loss()

    learning_rate = 0.0001
    optimizer = torch.optim.RMSprop(model.parameters(), lr = learning_rate, alpha=0.99)
    if multitask:
        optimizer = torch.optim.RMSprop(list(model.parameters())+list(icd_code_head.parameters())+list(sepsis_head.parameters())+list(resp_head.parameters()), lr = learning_rate, alpha=0.99)
        # base_params=[]
        # for name, param in model.named_parameters():
        #     if param.requires_grad & ('fc' not in name):
        #         #not the fully connected head
        #         base_params.append(param)
        # multitask_optimizer = torch.optim.RMSprop(base_params+list(icd_code_head.parameters()), lr = learning_rate, alpha=0.99)

    use_gpu = torch.cuda.is_available()

    interval = 100


    cur_time = time.time()
    pre_time = time.time()

    proba_projection=torch.nn.Softmax(dim=1)

    # Variables for Early Stopping
    is_best_model = 0
    patient_epoch = 0

    

    for epoch in range(num_epochs):

        losses_epoch_train = []
        losses_epoch_valid = []

        ### TRAIN ---------------------------------------------------------


        model.train()
        if multitask:
            icd_code_head.train()
            sepsis_head.train()
            resp_head.train()
        batches=tqdm(train_dataloader, desc='step', total=len(train_dataloader))
        for batch in batches:
            batch = tuple(t.to(device) for t in batch)
            measurement, measurement_last_obsv, mask, time_, labels, icd10, sepsis_labels, resp_labels = batch

            for item in batch:
                if np.sum(np.sum(np.sum(np.isnan(item.cpu().data.numpy()))))>0:
                    print("Nans")

            optimizer.zero_grad()
            # if multitask:
            #     multitask_optimizer.zero_grad()

            prediction, hidden_states=model(measurement.float(), measurement_last_obsv.float(), mask.float(), time_.float(), return_hidden=True)

            # get hidden_state for additional losses
            if multitask:
                icd10_prediction=icd_code_head(hidden_states)
                sepsis_prediction=sepsis_head(hidden_states)
                resp_prediction=resp_head(hidden_states)
                # add next sequence prediction
                # add next measurement prediction

            if output_last:
                loss_main = loss_CEL(prediction.view(measurement.shape[1],2), labels.long().squeeze(0))
                loss_train = loss_main
            else:
                full_labels = torch.cat((inputs[:,1:,:], labels.long()), dim = 1)
                loss_train = loss_MSE(outputs, full_labels)

            if multitask:
                # print('icd10_prediction: ', icd10_prediction.shape, icd10.shape)
                # print('icd10_prediction queezed: ', torch.squeeze(icd10_prediction, 0).shape, torch.unsqueeze(torch.squeeze(icd10), 0).expand_as(torch.squeeze(icd10_prediction, 0)).float().shape)
                loss_from_icd=loss_icd(torch.squeeze(icd10_prediction, 0), torch.unsqueeze(torch.squeeze(icd10), 0).expand_as(torch.squeeze(icd10_prediction, 0)).float())
                loss_train+=loss_from_icd
                loss_from_sepsis=loss_sepsis(sepsis_prediction.view(measurement.shape[1],2), torch.squeeze(sepsis_labels.long(), 0))
                loss_train+=loss_from_sepsis
                loss_from_resp=loss_resp(resp_prediction.view(measurement.shape[1],2), torch.squeeze(resp_labels.long(), 0))
                loss_train+=loss_from_resp
                # loss_train.backward(retain_graph=True)
                # loss_train_multitask.backward()
            else:
                pass
            loss_train.backward()

            optimizer.step()
            # if multitask:
            #     multitask_optimizer.step()

            losses_epoch_train.append((loss_main.cpu().data.numpy(),
                                        loss_from_icd.cpu().data.numpy(),
                                        loss_from_sepsis.cpu().data.numpy(),
                                        loss_from_resp.cpu().data.numpy()))





            batches.set_description("{:02f}".format(loss_train.cpu().data.numpy()))


        ### VALIDATION ---------------------------------------------------------

        model.eval()
        icd_code_head.eval()
        sepsis_head.eval()
        resp_head.eval()
        # batches=tqdm(enumerate(val_dataloader), desc='step', total=len(val_dataloader))
        labels=[]
        scores=[]
        for i, batch in enumerate(val_dataloader):
            batch = tuple(t.to(device) for t in batch)
            measurement_val, measurement_last_obsv_val, mask_val, time_val, labels_val, icd10, sepsis_labels, resp_labels = batch

            # print(measurement_val.shape, measurement_last_obsv_val.shape, mask_val.shape, time_val.shape)
            # print(measurement.shape, measurement_last_obsv.shape, mask.shape, time_.shape)

            with torch.no_grad():
                prediction_val, hidden_states = model(measurement_val.float(), measurement_last_obsv_val.float(), mask_val.float(), time_val.float(), return_hidden=True)
                scores.append(proba_projection(prediction_val).detach().cpu().numpy()) # I just learned how to spell detach
                labels.append(labels_val.detach().cpu().numpy())





            if output_last:
                loss_valid =loss_CEL(prediction_val.view(measurement_val.shape[1],2), labels_val.long().squeeze(0))
            else:
                full_labels_val = torch.cat((inputs_val[:,1:,:], labels_val.long()), dim = 1)
                loss_valid = loss_MSE(outputs_val, full_labels_val.long())




            if multitask:
                icd10_prediction=icd_code_head(hidden_states)
                sepsis_prediction=sepsis_head(hidden_states)
                resp_prediction=resp_head(hidden_states)
                loss_from_icd=loss_icd(torch.squeeze(icd10_prediction, 0), torch.unsqueeze(torch.squeeze(icd10), 0).expand_as(torch.squeeze(icd10_prediction, 0)).float())
                loss_valid+=loss_from_icd
                loss_from_sepsis=loss_sepsis(sepsis_prediction.view(measurement_val.shape[1],2), torch.squeeze(sepsis_labels.long(), 0))
                loss_valid+=loss_from_sepsis
                loss_from_resp=loss_resp(resp_prediction.view(measurement_val.shape[1],2), torch.squeeze(resp_labels.long(), 0))
                loss_valid+=loss_from_resp


                losses_epoch_valid.append((loss_main.cpu().data.numpy(),
                                        loss_from_icd.cpu().data.numpy(),
                                        loss_from_sepsis.cpu().data.numpy(),
                                        loss_from_resp.cpu().data.numpy()))
            else:
                losses_epoch_valid.append(loss_valid.cpu().data.numpy())



            # # compute the loss
            # labels.append(label.view(-1,).squeeze().cpu().data.numpy())
            # scores.append(m(classfication).view(-1).squeeze().cpu().data.numpy())

#             print(sklearn.metrics.roc_auc_score(labels_val.detach().cpu().numpy(), prediction_val.detach().cpu().numpy()[:,1]))

        try:
            # print("garbage")
            # print(np.asarray(losses_epoch_valid).shape)
            # print("success")
            avg_losses_epoch_valid=np.mean(np.asarray(losses_epoch_valid), axis=0)
            avg_losses_epoch_train=np.mean(np.asarray(losses_epoch_train), axis=0)
            # print(avg_losses_epoch_valid.shape)
            # print(avg_losses_epoch_valid.shape)
        except:
            avg_losses_epoch_valid=np.mean(np.concatenate(losses_epoch_valid), axis=0)
            avg_losses_epoch_train=np.mean(np.concatenate(losses_epoch_train), axis=0)

        # Early Stopping
        if epoch == 0:
            is_best_model = 1
            best_model = model
            min_loss_epoch_valid = 10000.0
            if avg_losses_epoch_valid[0] < min_loss_epoch_valid:
                min_loss_epoch_valid = avg_losses_epoch_valid[0]
        else:
            if min_loss_epoch_valid - avg_losses_epoch_valid[0] > min_delta:
                is_best_model = 1
                best_model = model
                min_loss_epoch_valid = avg_losses_epoch_valid [0]
                patient_epoch = 0
            else:
                is_best_model = 0
                patient_epoch += 1
                if patient_epoch >= patience:
                    print('Early Stopped at Epoch:', epoch)
                    break

        labels=np.concatenate([l.ravel() for l in labels], axis=0)
        scores=np.concatenate(scores, axis=0)
        AUC=roc_auc_score(labels, scores[:,1].ravel())

        cur_time = time.time()
        train_loss = np.around(avg_losses_epoch_train[0], decimals=8)
        valid_loss = np.around(avg_losses_epoch_valid[0], decimals=8)
        valid_auc =  np.around(AUC, decimals=2)
        time_elapsed = np.around([cur_time - pre_time] , decimals=2)
        if len(avg_losses_epoch_valid)>1:
            train_ICD10 = np.around(avg_losses_epoch_train[1], decimals=4)
            valid_ICD10 = np.around(avg_losses_epoch_valid[1], decimals=4)
            train_sepsis = np.around(avg_losses_epoch_train[2], decimals=4)
            valid_sepsis = np.around(avg_losses_epoch_valid[2], decimals=4)
            train_resp = np.around(avg_losses_epoch_train[3], decimals=4)
            valid_resp = np.around(avg_losses_epoch_valid[3], decimals=4)

            print("""Epoch: {},
                  train_loss: {}, valid_loss: {},
                  valid_auc: {}, time: {},
                  best model: {} \n
                  train_ICD10: {} valid_ICD10: {},
                  train_sepsis: {}, valid_sepsis: {},
                  train_resp: {}, valid_resp:""".format(
                        epoch, \
                        train_loss, valid_loss,\
                        valid_auc,\
                        time_elapsed,\
                        is_best_model,\
                        train_ICD10, valid_ICD10,\
                        train_sepsis, valid_sepsis,
                        train_resp, valid_sepsis))
        else:
            print("""Epoch: {}, train_loss: {}, valid_loss: {},
                  valid_auc: {}, time: {},
                  best model: {}""".format(epoch, train_loss, valid_loss,\
                                           valid_auc, time_elapsed, is_best_model))
        pre_time = cur_time


    return best_model, [avg_losses_epoch_train, avg_losses_epoch_valid]




def predict_GRUD(model, test_dataloader):
    """
    Input:
        model: GRU-D model
        test_dataloader: containing batches of measurement, measurement_last_obsv, mask, time_, labels
    Returns:
        predictions: size[num_samples, 2]
        labels: size[num_samples]
    """
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() and not False else "cpu")
    proba_projection=torch.nn.Softmax(dim=1)

    predictions=[]
    labels=[]
    ethnicities=[]
    genders=[]
    for data in test_dataloader:
        data = tuple(t.to(device) for t in data)
        measurement, measurement_last_obsv, mask, time_, label, icd10, sepsis_labels, resp_labels  = data #, ethnicity, gender = data


#         if use_gpu:
#             convert_to_cuda=lambda x: Variable(x.cuda())
#             X, X_last_obsv, Mask, Delta, label = map(convert_to_cuda, [measurement, measurement_last_obsv, mask, time_, label])
#         else:
# #                 inputs, labels = Variable(inputs), Variable(labels)
#             convert_to_tensor=lambda x: Variable(x)
#             X, X_last_obsv, Mask, Delta, label  = map(convert_to_tensor, [measurement, measurement_last_obsv, mask, time_, label])


        prediction=model(measurement.float(), measurement_last_obsv.float(), mask.float(), time_.float())
        pred=proba_projection(prediction)

        predictions.append(pred.detach().cpu().numpy()) # I just learned how to spell detach
        labels.append(label.detach().cpu().numpy())
        # ethnicities.append(ethnicity.numpy())
        # genders.append(gender.numpy())

    return predictions, labels #, ethnicities, genders