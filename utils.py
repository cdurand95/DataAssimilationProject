import os
import time
import copy

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl

from sklearn import decomposition
from sklearn.feature_extraction import image

import scipy
from scipy.integrate import solve_ivp

##### CNN model

class CNN(pl.LightningModule):

    def __init__(self, data, n_layers=4, dW=1, dimCNN=10, batch_size=128, lr=5e-3):
        super(CNN, self).__init__()

        self.data = data
        self.data_shape = data[0]['Truth'].shape[1:]
        self.nlayers = n_layers
        self.dW = dW
        self.dimCNN = dimCNN
        self.batch_size = batch_size
        self.lr = lr

        self.layers_list = nn.ModuleList(
            [torch.nn.Conv1d(
            self.data_shape[1],
            self.data_shape[1]*dimCNN,
            2*self.dW+1,
            padding=dW)])
        self.layers_list.extend(
            [torch.nn.Conv1d(
            self.data_shape[1]*dimCNN,
            self.data_shape[1]*dimCNN,
            2*self.dW+1,
            padding=dW) for _ in range(1, self.nlayers)])
        self.layers_list.append(
            torch.nn.Conv1d(
            self.data_shape[1]*dimCNN,
            self.data_shape[1],
            1,
            padding=0,
            bias=False))

        self.tot_loss = []
        self.tot_val_loss = []

        self.best_loss = 1e10

    def forward(self, xinp):

        xinp = xinp.view(-1, self.data_shape[1], self.data_shape[0])

        x = self.layers_list[0](xinp)
        for layer in self.layers_list[1:]:
            x = layer(F.relu(x))
        x = x.view(-1, self.data_shape[0], self.data_shape[1])

        return x

    def setup(self, stage='None'):

        training_dataset = torch.utils.data.TensorDataset(
                               torch.Tensor(self.data[0]['Init']),
                               torch.Tensor(self.data[0]['Obs']),
                               torch.Tensor(self.data[0]['Mask']),
                               torch.Tensor(self.data[0]['Truth']))
        val_dataset      = torch.utils.data.TensorDataset(
                               torch.Tensor(self.data[1]['Init']),
                               torch.Tensor(self.data[1]['Obs']),
                               torch.Tensor(self.data[1]['Mask']),
                               torch.Tensor(self.data[1]['Truth']))
        test_dataset     = torch.utils.data.TensorDataset(
                               torch.Tensor(self.data[2]['Init']),
                               torch.Tensor(self.data[2]['Obs']),
                               torch.Tensor(self.data[2]['Mask']),
                               torch.Tensor(self.data[2]['Truth']))

        self.dataloaders = {
            'train': torch.utils.data.DataLoader(training_dataset,
                         batch_size=self.batch_size,
                         shuffle=True, num_workers=0),
            'val':   torch.utils.data.DataLoader(val_dataset,
                         batch_size=self.batch_size,
                         shuffle=False, num_workers=0),
            'test':  torch.utils.data.DataLoader(test_dataset,
                         batch_size=self.batch_size,
                         shuffle=False, num_workers=0)
        }

    def loss(self, x, y):
         return torch.mean((x - y)**2)

    def training_step(self, train_batch, batch_idx):

        inputs_init, inputs_missing, masks, targets_GT = train_batch
        inputs_init = torch.autograd.Variable(inputs_init, requires_grad=True)

        num_loss = 0
        running_loss = 0.0

        outputs = self(inputs_init)
        loss = torch.mean((outputs - targets_GT)**2)

        running_loss += loss.item() * inputs_missing.size(0)
        num_loss += inputs_missing.size(0)
        epoch_loss = running_loss / num_loss

        self.tot_loss.append(epoch_loss)
        self.log('train_loss', epoch_loss)

        return loss

    def validation_step(self, val_batch, batch_idx):

        inputs_init, inputs_missing, masks, targets_GT = val_batch

        num_val_loss = 0
        running_val_loss = 0.0

        outputs = self(inputs_init)
        loss = torch.mean((outputs - targets_GT)**2)

        running_val_loss += loss.item() * inputs_missing.size(0)
        num_val_loss += inputs_missing.size(0)
        epoch_val_loss = running_val_loss / num_val_loss

        self.tot_val_loss.append(epoch_val_loss)
        self.log('val_loss', loss, prog_bar=False, logger=False)

        if epoch_val_loss < self.best_loss:
            self.best_loss = epoch_val_loss
            self.best_model_wts = copy.deepcopy(self.state_dict())

        return loss

    def configure_optimizers(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return self.optimizer

    def train_dataloader(self):
        return self.dataloaders['train']

    def val_dataloader(self):
        return self.dataloaders['val']

    def test_dataloader(self):
        return self.dataloaders['test']

##### Data

class time_series:
    values = 0.
    time   = 0.

### L63

def AnDA_Lorenz_63(S, t, sigma, rho, beta):
    """ Lorenz-63 dynamical model. """
    x_1 = sigma*(S[1]-S[0]);
    x_2 = S[0]*(rho-S[2])-S[1];
    x_3 = S[0]*S[1] - beta*S[2];
    dS  = np.array([x_1,x_2,x_3]);
    return dS

def L63_sparse_noisy_data(
    y0 = np.array([8.,0.,30.]),
    sigma = 10.,
    rho = 28.,
    beta = 8./3.,
    dt_integration = 0.01,
    final_time = 5.,
    freq_obs = 2,
    seed_noise = 1234,
    sigma_noise = np.sqrt(2.),
    var_mask = np.array([1,1,1]),
    seed_sparsity = 4321,
    seed_arange = 1111,
    sparsity = 1,
    masked_value=0.,
    num_variables=3):

    """
    Returns data mask, noisy and sparse observation data, noisy data and true
    data generated by the L63 model with given initial condition y0.

    y0 :             array of shape (3,), initial condition
    sigma :          float, L63 parameter
    rho :            float, L63 parameter
    beta :           float, L63 parameter
    dt_integration : float, integration time using RK4 scheme
    final_time :     float, final simulation time
    freq_obs :       int, number of timesteps between each observation of
                     true state
    seed_noise :     int, random seed of noise generation
    sigma_noise :    float, standard deviation of observation noise
    var_mask :       array of shape (3,), tag for observed variables
    seed_sparsity :  int, random seed for data sparsity
    sparsity :       float in [0, 1], percentage of observationnal data for
                     each observed variable
    masked_value :   value retained for masked data
    """

    t_eval = np.linspace(0, final_time, num=int(final_time/dt_integration))
    t_obs = t_eval[::freq_obs]

    y_true = solve_ivp(
        fun=lambda t,y: AnDA_Lorenz_63(y, t, sigma, rho, beta),
        t_span=(0, final_time),
        y0=y0,
        t_eval=t_eval,
        method='RK45').y.transpose()

    np.random.seed(seed_noise)
    y_noise = y_true + sigma_noise * np.random.randn(*y_true.shape)

    np.random.seed(seed_sparsity)
    mask = np.zeros(y_true.shape)
    mask_obs = np.zeros(y_true[::freq_obs].shape)

    s = int(sparsity*mask_obs.shape[0])
    for i in range(mask_obs.shape[1]):
        mask_obs[np.random.choice(mask_obs.shape[0], size=s, replace=False), i] = np.ones(s)

    np.random.seed(seed_arange)
    mask[::freq_obs] = mask_obs
    if num_variables==3 :
        var_mask = np.array([1,1,1])
    elif num_variables==2:
        var_mask = np.array([1,1,0])
        np.random.shuffle(var_mask)
    elif num_variables==1:
        var_mask = np.array([1,0,0])
        np.random.shuffle(var_mask)
    else :
        return RaiseError
    mask = mask*var_mask

    y_obs = np.copy(y_noise)
    np.putmask(y_obs, mask==0, masked_value)

    y_missing = np.copy(y_true)
    np.putmask(y_missing, mask==0, masked_value)

    return mask, y_obs, y_true, y_missing

def L63PatchDataExtraction(sparsity=1, sigma_noise=np.sqrt(2.), num_variables=3, var_mask=np.array([1, 1, 1])):
    """
    Returns Training, Validation and Testing Dataset using L63_sparse_noisy_data function for L63 Model
    Inputs :
    sparsity :       float in [0, 1], percentage of observationnal data for
                     each observed variable
    sigma_noise :    float, standard deviation of observation noise
    num_variables :  int in [1, 2, 3], choice of the number of variables observed

    Outputs : 3 dictionnaries : Training_dataset, Val_dataset and Test_dataset with the array 'Truth', 'Missing', 'Obs', 'Init', and 'Mask'
    'Truth' : ground-truth trajectory simulated
    'Missing' : ground-truth trajectory on observed data
    'Obs' : Observed data : masks and noise applied
    'Mask' : Mask array : value 1 in the point is observed, 0 otherwise
    'Init' : Interpolated trajectory between the observed data points.

    """
    NbTraining = 128*100
    NbVal      = 128*20
    NbTest     = 128*20
    begin_time = 10
    final_time =( NbTraining+2*NbVal +begin_time)*2

    mask, y_obs, y_true, y_missing = L63_sparse_noisy_data(sparsity = sparsity, sigma_noise = sigma_noise,final_time =final_time,num_variables=num_variables, var_mask_var_mask)

    X_train          = y_true[begin_time*100:begin_time*100+NbTraining*200].reshape((NbTraining,200,3))
    X_train_missing  = y_missing[begin_time*100:begin_time*100+NbTraining*200].reshape((NbTraining,200,3))
    X_train_obs      = y_obs[begin_time*100:begin_time*100+NbTraining*200].reshape((NbTraining,200,3))
    mask_train       = mask[begin_time*100:begin_time*100+NbTraining*200].reshape((NbTraining,200,3))

    X_val          = y_true[begin_time*100+NbTraining*200:begin_time*100+NbTraining*200+NbVal*200].reshape((NbVal,200,3))
    X_val_missing  = y_missing[begin_time*100+NbTraining*200:begin_time*100+NbTraining*200+NbVal*200].reshape((NbVal,200,3))
    X_val_obs      = y_obs[begin_time*100+NbTraining*200:begin_time*100+NbTraining*200+NbVal*200].reshape((NbVal,200,3))
    mask_val       = mask[begin_time*100+NbTraining*200:begin_time*100+NbTraining*200+NbVal*200].reshape((NbVal,200,3))

    X_test          = y_true[begin_time*100+NbTraining*200+NbVal*200:begin_time*100+NbTraining*200+NbVal*200+NbTest*200].reshape((NbTest,200,3))
    X_test_missing  = y_missing[begin_time*100+NbTraining*200+NbVal*200:begin_time*100+NbTraining*200+NbVal*200+NbTest*200].reshape((NbTest,200,3))
    X_test_obs      = y_obs[begin_time*100+NbTraining*200+NbVal*200:begin_time*100+NbTraining*200+NbVal*200+NbTest*200].reshape((NbTest,200,3))
    mask_test       = mask[begin_time*100+NbTraining*200+NbVal*200:begin_time*100+NbTraining*200+NbVal*200+NbTest*200].reshape((NbTest,200,3))

    meanTr          = np.mean(X_train_missing[:]) / np.mean(mask_train)
    meanTe          = np.mean(X_test_missing[:]) / np.mean(mask_test)
    meanV           = np.mean(X_val_missing[:]) / np.mean(mask_val)
    # print(meanTr)

    X_train_Init = np.zeros(X_train.shape)
    for ii in range(0,X_train.shape[0]):
    # Initial linear interpolation for each component
        XInit = np.zeros((X_train.shape[1],X_train.shape[2]))

        for kk in range(0,3):
            indt  = np.where( mask_train[ii,:,kk] == 1.0 )[0]

            indt_ = np.where( mask_train[ii,:,kk] == 0.0 )[0]
            if len(indt) > 1:
                indt_[ np.where( indt_ < np.min(indt)) ] = np.min(indt)
                indt_[ np.where( indt_ > np.max(indt)) ] = np.max(indt)
                fkk = scipy.interpolate.interp1d(indt, X_train_obs[ii,indt,kk],axis=-1)
                XInit[indt,kk]  = X_train_obs[ii,indt,kk]
                XInit[indt_,kk] = fkk(indt_)

            else:
                XInit = XInit + meanTr

        X_train_Init[ii,:,:] = XInit

    X_test_Init = np.zeros(X_test.shape)
    for ii in range(0,X_test.shape[0]):
    # Initial linear interpolation for each component
        XInit = np.zeros((X_test.shape[1],X_test.shape[2]))
        for kk in range(0,3):
            indt  = np.where( mask_test[ii,:,kk] == 1.0 )[0]

            indt_ = np.where( mask_test[ii,:,kk] == 0.0 )[0]

            if len(indt) > 1:

                indt_[ np.where( indt_ < np.min(indt)) ] = np.min(indt)
                indt_[ np.where( indt_ > np.max(indt)) ] = np.max(indt)

                fkk = scipy.interpolate.interp1d(indt, X_test_obs[ii,indt,kk],axis=-1)
                XInit[indt,kk]  = X_test_obs[ii,indt,kk]
                XInit[indt_,kk] = fkk(indt_)

            else:
                XInit = XInit + meanTe

        X_test_Init[ii,:,:] = XInit

    X_val_Init = np.zeros(X_val.shape)
    for ii in range(0,X_val.shape[0]):
    # Initial linear interpolation for each component
        XInit = np.zeros((X_val.shape[1],X_val.shape[2]))

        for kk in range(0,3):
            indt  = np.where( mask_val[ii,:,kk] == 1.0 )[0]
            indt_ = np.where( mask_val[ii,:,kk] == 0.0 )[0]

            if len(indt) > 1:
                indt_[ np.where( indt_ < np.min(indt)) ] = np.min(indt)
                indt_[ np.where( indt_ > np.max(indt)) ] = np.max(indt)
                fkk = scipy.interpolate.interp1d(indt, X_val_obs[ii,indt,kk],axis=-1)
                XInit[indt,kk]  = X_val_obs[ii,indt,kk]
                XInit[indt_,kk] = fkk(indt_)
            else:
                XInit = XInit + meanV

        X_val_Init[ii,:,:] = XInit

    Training_dataset = {}
    Training_dataset['Truth']=X_train
    Training_dataset['Obs']=X_train_obs
    Training_dataset['Missing']=X_train_missing
    Training_dataset['Init']=X_train_Init
    Training_dataset['Mask']=mask_train

    Val_dataset = {}
    Val_dataset['Truth']=X_val
    Val_dataset['Obs']=X_val_obs
    Val_dataset['Missing']=X_val_missing
    Val_dataset['Init']=X_val_Init
    Val_dataset['Mask']=mask_val

    Test_dataset = {}
    Test_dataset['Truth']=X_test
    Test_dataset['Obs']=X_test_obs
    Test_dataset['Missing']=X_test_missing
    Test_dataset['Init']=X_test_Init
    Test_dataset['Mask']=mask_test

    return Training_dataset, Val_dataset, Test_dataset

### L96

def AnDA_Lorenz_96(S, t ,F, J):
    """ Lorenz-96 dynamical model. """
    x = np.zeros(J);
    x[0] = (S[1]-S[J-2])*S[J-1]-S[0];
    x[1] = (S[2]-S[J-1])*S[0]-S[1];
    x[J-1] = (S[0]-S[J-3])*S[J-2]-S[J-1];
    for j in range(2,J-1):
        x[j] = (S[j+1]-S[j-2])*S[j-1]-S[j];
    dS = x.T + F;
    return dS

def L96_sparse_noisy_data(
    F = 8,
    dt_integration = 0.05,
    final_time = 10.,
    freq_obs = 2,
    seed_noise = 1234,
    seed_init  = 1357,
    sigma_noise = np.sqrt(2.),
    var_mask = np.ones(40),
    seed_sparsity = 4321,
    seed_arange  = 1111,
    sparsity = 1,
    masked_value=0.,
    num_variables=40):

    """
    Returns data mask, noisy and sparse observation data, noisy data and true
    data generated by the L63 model with given initial condition y0.

    y0 :             array of shape (3,), initial condition
    F  :             float, L96 term
    dt_integration : float, integration time using RK4 scheme
    final_time :     float, final simulation time
    freq_obs :       int, number of timesteps between each observation of
                     true state
    seed_noise :     int, random seed of noise generation
    sigma_noise :    float, standard deviation of observation noise
    var_mask :       array of shape (40,), tag for observed variables
    seed_sparsity :  int, random seed for data sparsity
    sparsity :       float in [0, 1], percentage of observationnal data for
                     each observed variable
    masked_value :   value retained for masked data
    num_variable :   int between 1 and 40 : number of observed variables
    """

    np.random.seed(seed_init)
    y0 = np.random.randn(40)
    t_eval = np.linspace(0, final_time, num=int(final_time/dt_integration))
    t_obs = t_eval[::freq_obs]

    y_true = solve_ivp(
        fun=lambda t,y: AnDA_Lorenz_96(y,t,F=F,J=40),
        t_span=(0, final_time),
        y0=y0,
        t_eval=t_eval,
        method='RK45').y.transpose()

    np.random.seed(seed_noise)
    y_noise = y_true + sigma_noise * np.random.randn(*y_true.shape)

    np.random.seed(seed_sparsity)
    mask = np.zeros(y_true.shape)
    mask_obs = np.zeros(y_true[::freq_obs].shape)

    s = int(sparsity*mask_obs.shape[0])
    for i in range(mask_obs.shape[1]):
        mask_obs[np.random.choice(mask_obs.shape[0], size=s, replace=False), i] = np.ones(s)

    mask[::freq_obs] = mask_obs
    np.random.seed(seed_arange)
    if num_variables != 40 :
        var_mask = np.concatenate((np.ones(num_variables), np.zeros(40-num_variables)), axis=0)
        np.random.shuffle(var_mask)
    mask = mask*var_mask

    y_obs = np.copy(y_noise)
    np.putmask(y_obs, mask==0, masked_value)

    y_missing = np.copy(y_true)
    np.putmask(y_missing, mask==0, masked_value)

    return mask, y_obs, y_true, y_missing

def L96PatchDataExtraction(sparsity=1,sigma_noise=np.sqrt(2.),num_variables=40):
    """
    Returns Training, Validation and Testing Dataset using L63_sparse_noisy_data function for L63 Model
    Inputs :
    sparsity :       float in [0, 1], percentage of observationnal data for
                     each observed variable
    sigma_noise :    float, standard deviation of observation noise
    num_variables :  int in [1:40], choice of the number of variables observed

    Outputs : 3 dictionnaries : Training_dataset, Val_dataset and Test_dataset with the array 'Truth', 'Missing', 'Obs', 'Init', and 'Mask'
    'Truth' : ground-truth trajectory simulated
    'Missing' : ground-truth trajectory on observed data
    'Obs' : Observed data : masks and noise applied
    'Mask' : Mask array : value 1 in the point is observed, 0 otherwise
    'Init' : Interpolated trajectory between the observed data points.
    """

    NbTraining = 1280
    NbVal      = 256
    NbTest     = 256
    begin_time = 2
    final_time = 2512*10 + begin_time*100

    mask, y_obs, y_true, y_missing = L96_sparse_noisy_data(sparsity = sparsity, sigma_noise = sigma_noise,final_time =final_time,num_variables=num_variables)

    X_train          = y_true[begin_time*100:begin_time*100+NbTraining*200].reshape((NbTraining,200,40))
    X_train_missing  = y_missing[begin_time*100:begin_time*100+NbTraining*200].reshape((NbTraining,200,40))
    X_train_obs      = y_obs[begin_time*100:begin_time*100+NbTraining*200].reshape((NbTraining,200,40))
    mask_train       = mask[begin_time*100:begin_time*100+NbTraining*200].reshape((NbTraining,200,40))

    X_val          = y_true[begin_time*100+NbTraining*200:begin_time*100+NbTraining*200+NbVal*200].reshape((NbVal,200,40))
    X_val_missing  = y_missing[begin_time*100+NbTraining*200:begin_time*100+NbTraining*200+NbVal*200].reshape((NbVal,200,40))
    X_val_obs      = y_obs[begin_time*100+NbTraining*200:begin_time*100+NbTraining*200+NbVal*200].reshape((NbVal,200,40))
    mask_val       = mask[begin_time*100+NbTraining*200:begin_time*100+NbTraining*200+NbVal*200].reshape((NbVal,200,40))

    X_test          = y_true[begin_time*100+NbTraining*200+NbVal*200:begin_time*100+NbTraining*200+NbVal*200+NbTest*200].reshape((NbTest,200,40))
    X_test_missing  = y_missing[begin_time*100+NbTraining*200+NbVal*200:begin_time*100+NbTraining*200+NbVal*200+NbTest*200].reshape((NbTest,200,40))
    X_test_obs      = y_obs[begin_time*100+NbTraining*200+NbVal*200:begin_time*100+NbTraining*200+NbVal*200+NbTest*200].reshape((NbTest,200,40))
    mask_test       = mask[begin_time*100+NbTraining*200+NbVal*200:begin_time*100+NbTraining*200+NbVal*200+NbTest*200].reshape((NbTest,200,40))

    meanTr          = np.mean(X_train_missing[:]) / np.mean(mask_train)
    meanTe          = np.mean(X_test_missing[:]) / np.mean(mask_test)
    meanV           = np.mean(X_val_missing[:]) / np.mean(mask_val)

    X_train_Init = np.zeros(X_train.shape)
    for ii in range(0,X_train.shape[0]):
    # Initial linear interpolation for each component
        XInit = np.zeros((X_train.shape[1],X_train.shape[2]))

        for kk in range(0,40):
            indt  = np.where( mask_train[ii,:,kk] == 1.0 )[0]

            indt_ = np.where( mask_train[ii,:,kk] == 0.0 )[0]
            if len(indt) > 1:
                indt_[ np.where( indt_ < np.min(indt)) ] = np.min(indt)
                indt_[ np.where( indt_ > np.max(indt)) ] = np.max(indt)
                fkk = scipy.interpolate.interp1d(indt, X_train_obs[ii,indt,kk],axis=-1)
                XInit[indt,kk]  = X_train_obs[ii,indt,kk]
                XInit[indt_,kk] = fkk(indt_)

            else:
                XInit = XInit + meanTr

        X_train_Init[ii,:,:] = XInit

    X_test_Init = np.zeros(X_test.shape)
    for ii in range(0,X_test.shape[0]):
    # Initial linear interpolation for each component
        XInit = np.zeros((X_test.shape[1],X_test.shape[2]))
        for kk in range(0,40):
            indt  = np.where( mask_test[ii,:,kk] == 1.0 )[0]

            indt_ = np.where( mask_test[ii,:,kk] == 0.0 )[0]

            if len(indt) > 1:

                indt_[ np.where( indt_ < np.min(indt)) ] = np.min(indt)
                indt_[ np.where( indt_ > np.max(indt)) ] = np.max(indt)

                fkk = scipy.interpolate.interp1d(indt, X_test_obs[ii,indt,kk],axis=-1)
                XInit[indt,kk]  = X_test_obs[ii,indt,kk]
                XInit[indt_,kk] = fkk(indt_)

            else:
                XInit = XInit + meanTe

        X_test_Init[ii,:,:] = XInit

    X_val_Init = np.zeros(X_val.shape)
    for ii in range(0,X_val.shape[0]):
    # Initial linear interpolation for each component
        XInit = np.zeros((X_val.shape[1],X_val.shape[2]))

        for kk in range(0,40):
            indt  = np.where( mask_val[ii,:,kk] == 1.0 )[0]
            indt_ = np.where( mask_val[ii,:,kk] == 0.0 )[0]

            if len(indt) > 1:
                indt_[ np.where( indt_ < np.min(indt)) ] = np.min(indt)
                indt_[ np.where( indt_ > np.max(indt)) ] = np.max(indt)
                fkk = scipy.interpolate.interp1d(indt, X_val_obs[ii,indt,kk],axis=-1)
                XInit[indt,kk]  = X_val_obs[ii,indt,kk]
                XInit[indt_,kk] = fkk(indt_)
            else:
                XInit = XInit + meanV

        X_val_Init[ii,:,:] = XInit

    Training_dataset = {}
    Training_dataset['Truth']=X_train
    Training_dataset['Obs']=X_train_obs
    Training_dataset['Missing']=X_train_missing
    Training_dataset['Init']=X_train_Init
    Training_dataset['Mask']=mask_train

    Val_dataset = {}
    Val_dataset['Truth']=X_val
    Val_dataset['Obs']=X_val_obs
    Val_dataset['Missing']=X_val_missing
    Val_dataset['Init']=X_val_Init
    Val_dataset['Mask']=mask_val

    Test_dataset = {}
    Test_dataset['Truth']=X_test
    Test_dataset['Obs']=X_test_obs
    Test_dataset['Missing']=X_test_missing
    Test_dataset['Init']=X_test_Init
    Test_dataset['Mask']=mask_test

    return Training_dataset, Val_dataset, Test_dataset

##### Metrics

def R_score(model, dataset):
    R_score = 0
    test= next(iter(dataset))
    x_truth=test[3].detach().numpy()
    x_pred=model(test[0])
    x_pred=x_pred.detach().numpy()
    R_score = np.sqrt(((x_pred-x_truth)**2).mean(axis=1)).mean(axis = 0)
    print('Variables reconstruction score : {}'.format(R_score))
    print('Global reconstruction score : {}'.format(R_score.mean()))
    return R_score

def reconstruction_error_4DVar(GT, pred):
    R_score = 0
    x_truth=GT.detach().numpy()
    x_pred=pred.detach().numpy()
    R_score = np.sqrt(((x_pred-x_truth)**2).mean(axis=2)).mean()
    return R_score

##### Plots

def visualisation_data(X_train, X_train_obs, X_train_Init, idx):

    plt.figure(figsize=(10,5))
    for jj in range(0,3):
        indjj = 131+jj
        plt.subplot(indjj)
        plt.plot(X_train_obs[idx,:,jj],'k.',label='Observations')
        plt.plot(X_train[idx,:,jj],'b-',label='Simulated trajectory')
        plt.plot(X_train_Init[idx,:,jj],label='Interpolated trajectory')

        plt.legend()
        plt.xlabel('Timestep')
    plt.savefig('Figures/visualisation_dataL63_2D.pdf')

def visualisation_data96(X_train, X_train_obs, X_train_Init, idx):

    plt.figure(figsize=(5,10))
    label = ['Truth','Observations','Interpolations']

    plt.subplot(311)
    plt.imshow(X_train[idx].transpose())
    plt.title('Truth')
    plt.colorbar()
    plt.xlabel('Timestep')

    plt.subplot(312)
    plt.imshow(X_train_obs[idx].transpose())
    plt.title('Observations')
    plt.colorbar()
    plt.xlabel('Timestep')

    plt.subplot(313)
    plt.imshow(X_train_Init[idx].transpose())
    plt.title('Interpolations')
    plt.colorbar()
    plt.xlabel('Timestep')

    plt.savefig('Figures/visualisation_dataL96_2D.pdf')

def plot_loss(model, max_epoch):

    tot_loss=torch.FloatTensor(model.tot_loss)
    tot_val_loss=torch.FloatTensor(model.tot_val_loss)
    n=np.shape(tot_loss)[0]//max_epoch
    m=np.shape(tot_val_loss)[0]//max_epoch
    j,k=0,0
    mean_loss=[]
    mean_val_loss=[]
    for i in range(max_epoch):
        mean_loss.append(torch.mean(tot_loss[j:j+n]))
        mean_val_loss.append(torch.mean(tot_val_loss[k:k+m]))
        k+=m
        j+=n

    plt.semilogy(np.arange(1,max_epoch+1,1),mean_loss ,'-',label='Train')
    plt.semilogy(np.arange(1,max_epoch+1,1),mean_val_loss ,'-',label='Validation')
    plt.xlabel('steps')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()

def plot_prediction(model, idx, dataset, name='prediction'):
    test = next(iter(dataset))

    x_pred=model(test[0])

    x_obs=test[1][idx].detach().numpy()
    x_pred=x_pred[idx].detach().numpy()
    x_truth=test[3][idx].detach().numpy()

    time_=np.arange(0,2,0.01)

    plt.figure(figsize=(15,6))
    for j in range(3):
        plt.subplot(1,3,j+1)
        plt.plot(time_,x_obs[j],'b.',alpha=0.2,label='obs')
        plt.plot(time_,x_pred[j],alpha=1,label='Prediction')
        plt.plot(time_,x_truth[j],alpha=0.7,label='Truth')
        plt.xlabel('Time')
        plt.ylabel('Position')
        plt.title('Variable {}'.format(j))
        plt.legend()
    plt.savefig('Figures/'+name+'.pdf',transparent = True)

def plot_loss(model, max_epoch):
    tot_loss=torch.FloatTensor(model.tot_loss)
    tot_val_loss=torch.FloatTensor(model.tot_val_loss)
    n=np.shape(tot_loss)[0]//max_epoch
    m=np.shape(tot_val_loss)[0]//max_epoch
    j,k=0,0
    mean_loss=[]
    mean_val_loss=[]
    for i in range(max_epoch):
        mean_loss.append(torch.mean(tot_loss[j:j+n]))
        mean_val_loss.append(torch.mean(tot_val_loss[k:k+m]))
        k+=m
        j+=n

    plt.semilogy(np.arange(1,max_epoch+1,1),mean_loss ,'-',label='Train')
    plt.semilogy(np.arange(1,max_epoch+1,1),mean_val_loss ,'-',label='Validation')
    plt.xlabel('steps')
    plt.ylabel('MSE')
    plt.legend()
    
def evaluation_model(path,max_epoch,model_name = 'L63',idx = 25,stage = 'Test',savepath='Blabla',sparsity = 1):
    '''
    model_name = 'L63' or 'L96' 
    idx = int (under 2000 for L63 and under 256 for L96 
    stage : 'Val' or 'Test' 
    path : path where the models are located '''
    
    
    n_layers_list = [2, 4, 6, 8]
    dW_list = [1, 2, 4, 8]
    i=1
    j=1
    fig, axs = plt.subplots(4,4, sharey=True,figsize=(15,15))
    
    #Loss plot
    for n_layer in n_layers_list : 
        for w in dW_list :
            model = torch.load(path + '/model_n{}_dW{}_epoch{}.pth'.format(n_layer,w, max_epoch))
            plt.subplot(4,4,4*(i-1)+j)
            tot_loss=torch.FloatTensor(model.tot_loss)
            tot_val_loss=torch.FloatTensor(model.tot_val_loss)
            len_tot_loss =np.shape(model.tot_loss)[0]//100
            len_tot_val_loss =np.shape(model.tot_val_loss)[0]//20
            tot_loss=torch.FloatTensor(model.tot_loss).numpy()
            tot_val_loss=torch.FloatTensor(model.tot_val_loss).numpy()
            plt.semilogy(np.arange(1,len_tot_loss+1,1),tot_loss[::100] ,'-',label='Train')
            plt.semilogy(np.arange(1,len_tot_val_loss+1,1),tot_val_loss[20::20] ,'-',label='Validation')
            plt.xlabel('epoch',fontsize=8)
            plt.ylabel('MSE',fontsize=8)
            plt.legend(fontsize=8)
            plt.title('Padding : {},  Layers Numbers : {}'.format(w,n_layer),fontsize=8)
            
            j+=1
        i+=1
        j=1
        
    plt.subplots_adjust( wspace=0.5, hspace=0.5)
    plt.savefig(savepath+'losses.pdf')
    i=1
    j=1
    if model_name == 'L63':
        data = utils.L63PatchDataExtraction(sparsity=sparsity)
    else : 
        data = utils.L96PatchDataExtraction(sparsity=sparsity)
    if stage == 'Val' : 
        dataset = data[1]
    elif stage == 'Test' : 
        dataset = data[2]   
                      
    x_obs=dataset[1][idx]

    x_truth=dataset[3][idx]

    time_=np.arange(0,2,0.01)                 
    fig, axs = plt.subplots(4,4, sharex=True, sharey=True,figsize=(15,15))
                
    for n_layer in n_layers_list : 
        for w in dW_list :
            model = torch.load(path + '/model_n{}_dW{}_epoch{}.pth'.format(n_layer,w, max_epoch))
            plt.subplot(4,4,4*(i-1)+j)
            x_pred=model(dataset[0][idx])
            x_pred=x_pred.detach().numpy()
            plt.plot(time_,x_obs[:,0],'b.',alpha=0.2,label='Obs')
            plt.plot(time_,x_pred[0,:,0],alpha=1,label='Prediction')
            plt.plot(time_,x_truth[:,0],alpha=0.7,label='Truth')
            plt.xlabel('Time')
            plt.ylabel('Position')
            plt.legend(fontsize = 8)
            plt.title('Padding : {},  Layers Numbers : {}'.format(w,n_layer),fontsize = 8)
            j+=1
        
        i+=1
        j=1
    plt.subplots_adjust( wspace=0.5, hspace=0.5)
    plt.savefig(savepath+'reconstructions.pdf',transparent = True)
    
    
def plot_prediction(model, idx, dataset, name='prediction'):
    test= next(iter(dataset))

    x_pred=model(test[0])

    x_obs=test[1][idx].detach().numpy()

    x_pred=x_pred[idx].detach().numpy()

    x_truth=test[3][idx].detach().numpy()

    time_=np.arange(0,2,0.01)

    plt.figure(figsize=(15,6))
    for j in range(3):
        plt.subplot(1,3,j+1)
        plt.plot(time_,x_obs[:,j],'b.',alpha=0.2,label='obs')
        plt.plot(time_,x_pred[:,j],alpha=1,label='Prediction')
        plt.plot(time_,x_truth[:,j],alpha=0.7,label='Truth')
        plt.xlabel('Time')
        plt.ylabel('Position')
        plt.title('Variable {}'.format(j))
        plt.legend()
    plt.savefig(name+'.pdf',transparent = True)

def visualisation4DVar(idx, x_obs, x_GT, xhat):
    plt.figure(figsize = (10,5))
    for kk in range(0,3):
        plt.subplot(1,3,kk+1)
        plt.plot(x_obs[idx,:,kk].detach().numpy(),'.',ms=3,alpha=0.3,label='Observations')
        plt.plot(x_GT[idx,:,kk].detach().numpy(),label='Simulated trajectory',alpha=0.8)
        plt.plot(xhat[idx,:,kk].detach().numpy(),label='4DVar Prediction',alpha=0.7)

        plt.legend()
    plt.suptitle('4DVar Reconstruction')
    plt.savefig('4DVar.pdf',transparent = True)

def plot_prediction96(model,idx,dataset,name='prediction'):
    test= next(iter(dataset))

    x_pred=model(test[0])

    x_obs=test[1][idx].detach().numpy()
    x_pred=x_pred[idx].detach().numpy()
    x_truth=test[3][idx].detach().numpy()


    time_=np.arange(0,2,0.01)

    plt.figure(figsize = (10,10))
    plt.subplot(3,1,1)
    plt.imshow(x_truth.transpose())
    plt.colorbar()
    plt.title('Ground truth')

    plt.subplot(3,1,2)
    plt.imshow(x_pred.transpose())
    plt.colorbar()
    plt.title('Prediction')

    plt.subplot(3,1,3)
    plt.imshow(x_truth.transpose()-x_pred.transpose())
    plt.colorbar()
    plt.title('Difference')

    plt.savefig('Figures/'+name+'.pdf',transparent = True)
