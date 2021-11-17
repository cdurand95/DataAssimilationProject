import numpy as np
import matplotlib.pyplot as plt 
import os

import time
import copy
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F

from sklearn import decomposition
import scipy
from scipy.integrate import solve_ivp

from sklearn.feature_extraction import image

from scipy.integrate import solve_ivp

def AnDA_Lorenz_63(S,t,sigma,rho,beta):
    """ Lorenz-63 dynamical model. """

    x_1 = sigma*(S[1]-S[0]);
    x_2 = S[0]*(rho-S[2])-S[1];
    x_3 = S[0]*S[1] - beta*S[2];
    dS  = np.array([x_1,x_2,x_3]);
    return dS

class Simulation_data:
    model = 'Lorenz_63'
    class parameters:
        sigma = 10.0
        rho = 28.0
        beta = 8.0/3
    dt_integration = 0.01 # integration time
    dt_states = 1 # number of integeration times between consecutive states (for xt and catalog)
    dt_obs = 8 # number of integration times between consecutive observations (for yo)
    var_obs = np.array([0,1,2]) # indices of the observed variables
    nb_loop_train = 10**2 # size of the catalog
    nb_loop_test = 20000 # size of the true state and noisy observations
    sigma2_catalog = 0.0 # variance of the model error to generate the catalog
    sigma2_obs = 2.0 # variance of the observation error to generate observation

class time_series:
  values = 0.
  time   = 0.

## data generation: L63 series
SD = Simulation_data()    
y0 = np.array([8.0,0.0,30.0])
tt = np.arange(SD.dt_integration,SD.nb_loop_test*SD.dt_integration+0.000001,SD.dt_integration)
S = solve_ivp(fun=lambda t,y: AnDA_Lorenz_63(y,t,SD.parameters.sigma,SD.parameters.rho,SD.parameters.beta),t_span=[0.,5+0.000001],y0=y0,first_step=SD.dt_integration,t_eval=np.arange(0,5+0.000001,SD.dt_integration),method='RK45')

y0 = S.y[:,-1];
S = solve_ivp(fun=lambda t,y: AnDA_Lorenz_63(y,t,SD.parameters.sigma,SD.parameters.rho,SD.parameters.beta),t_span=[SD.dt_integration,SD.nb_loop_test+0.000001],y0=y0,first_step=SD.dt_integration,t_eval=tt,method='RK45')
S = S.y.transpose()


class time_series:
  values = 0.
  time   = 0.
  
xt = time_series()
xt.values = S
xt.time   = tt


def L63PatchDataExtraction(xt,RMD):

  NbTraining = 10000
  NbVal      = 2000
  NbTest     = 2000
  time_step = 1
  dT        = 200
  sigNoise  = np.sqrt(2.0)
  rateMissingData = RMD#0.75#0.95

  # extract subsequences
  dataTrainingNoNaN = image.extract_patches_2d(image=xt.values[0:12000:time_step,:],patch_size=(dT,3),max_patches=NbTraining)
  dataValNoNaN     = image.extract_patches_2d(image=xt.values[15000::time_step,:],patch_size=(dT,3),max_patches=NbVal)
  dataTestNoNaN     = image.extract_patches_2d(image=xt.values[17500::time_step,:],patch_size=(dT,3),max_patches=NbTest)


  flagTypeMissData = 1
  if flagTypeMissData == 0:
      indRand         = np.random.permutation(dataTrainingNoNaN.shape[0]*dataTrainingNoNaN.shape[1]*dataTrainingNoNaN.shape[2])
      indRand         = indRand[0:int(rateMissingData*len(indRand))]

      dataTraining    = np.copy(dataTrainingNoNaN).reshape((dataTrainingNoNaN.shape[0]*dataTrainingNoNaN.shape[1]*dataTrainingNoNaN.shape[2],1))
      dataTraining[indRand] = float('nan')
      dataTraining    = np.reshape(dataTraining,(dataTrainingNoNaN.shape[0],dataTrainingNoNaN.shape[1],dataTrainingNoNaN.shape[2]))
    

      indRand         = np.random.permutation(dataTestNoNaN.shape[0]*dataTestNoNaN.shape[1]*dataTestNoNaN.shape[2])
      indRand         = indRand[0:int(rateMissingData*len(indRand))]
      dataTest        = np.copy(dataTestNoNaN).reshape((dataTestNoNaN.shape[0]*dataTestNoNaN.shape[1]*dataTestNoNaN.shape[2],1))
      dataTest[indRand] = float('nan')
      dataTest          = np.reshape(dataTest,(dataTestNoNaN.shape[0],dataTestNoNaN.shape[1],dataTestNoNaN.shape[2]))

      indRand         = np.random.permutation(dataValNoNaN.shape[0]*dataValNoNaN.shape[1]*dataValNoNaN.shape[2])
      indRand         = indRand[0:int(rateMissingData*len(indRand))]
      dataVal         = np.copy(dataValNoNaN).reshape((dataValNoNaN.shape[0]*dataValNoNaN.shape[1]*dataValNoNaN.shape[2],1))
      dataVal[indRand] = float('nan')
      dataVal          = np.reshape(dataVal,(dataValNoNaN.shape[0],dataValNoNaN.shape[1],dataValNoNaN.shape[2]))


      genSuffixObs    = '_ObsRnd_%02d_%02d'%(100*rateMissingData,10*sigNoise**2)

  else:
      time_step_obs   = int(1./(1.-rateMissingData))
      dataTraining    = np.zeros((dataTrainingNoNaN.shape))
      dataTraining[:] = float('nan')
      dataTraining[:,::time_step_obs,:] = dataTrainingNoNaN[:,::time_step_obs,:]
    
      dataTest    = np.zeros((dataTestNoNaN.shape))
      dataTest[:] = float('nan')
      dataTest[:,::time_step_obs,:] = dataTestNoNaN[:,::time_step_obs,:]

      dataVal    = np.zeros((dataValNoNaN.shape))
      dataVal[:] = float('nan')
      dataVal[:,::time_step_obs,:] = dataValNoNaN[:,::time_step_obs,:]


      genSuffixObs    = '_ObsSub_%02d_%02d'%(100*rateMissingData,10*sigNoise**2)
      
  # set to NaN patch boundaries
  dataTraining[:,0:10,:] =  float('nan')
  dataVal[:,0:10,:] =  float('nan')
  dataTest[:,0:10,:]     =  float('nan')
  dataTraining[:,dT-10:dT,:] =  float('nan')
  dataTest[:,dT-10:dT,:]     =  float('nan')
  dataVal[:,dT-10:dT,:]     =  float('nan')

  # mask for NaN
  maskTraining = (dataTraining == dataTraining).astype('float')
  maskTest     = ( dataTest    ==  dataTest   ).astype('float')
  maskVal     = ( dataVal    ==  dataVal   ).astype('float')


  dataTraining = np.nan_to_num(dataTraining)
  dataVal = np.nan_to_num(dataVal)
  dataTest     = np.nan_to_num(dataTest)

  # Permutation to have channel as #1 component
  dataTraining      = np.moveaxis(dataTraining,-1,1)
  maskTraining      = np.moveaxis(maskTraining,-1,1)
  dataTrainingNoNaN = np.moveaxis(dataTrainingNoNaN,-1,1)

  dataTest      = np.moveaxis(dataTest,-1,1)
  maskTest      = np.moveaxis(maskTest,-1,1)
  dataTestNoNaN = np.moveaxis(dataTestNoNaN,-1,1)

  dataVal     = np.moveaxis(dataVal,-1,1)
  maskVal     = np.moveaxis(maskVal,-1,1)
  dataValNoNaN = np.moveaxis(dataValNoNaN,-1,1)


  ## raw data
  X_train         = dataTrainingNoNaN
  X_train_missing = dataTraining
  mask_train      = maskTraining

  X_test         = dataTestNoNaN
  X_test_missing = dataTest
  mask_test      = maskTest

  X_val        = dataValNoNaN
  X_val_missing = dataVal
  mask_val      = maskVal


  ## normalized data
  meanTr          = np.mean(X_train_missing[:]) / np.mean(mask_train) 

  x_train_missing = X_train_missing - meanTr
  x_test_missing  = X_test_missing - meanTr
  x_val_missing  = X_val_missing - meanTr

  # scale wrt std
  stdTr           = np.sqrt( np.mean( X_train_missing**2 ) / np.mean(mask_train) )
  x_train_missing = x_train_missing / stdTr
  x_test_missing  = x_test_missing / stdTr
  x_val_missing  = x_val_missing / stdTr

  x_train = (X_train - meanTr) / stdTr
  x_test  = (X_test - meanTr) / stdTr
  x_val  = (X_val - meanTr) / stdTr


  # Generate noisy observsation
  X_train_obs = X_train_missing + sigNoise * maskTraining * np.random.randn(X_train_missing.shape[0],X_train_missing.shape[1],X_train_missing.shape[2])
  X_test_obs  = X_test_missing  + sigNoise * maskTest * np.random.randn(X_test_missing.shape[0],X_test_missing.shape[1],X_test_missing.shape[2])
  X_val_obs  = X_val_missing  + sigNoise * maskVal * np.random.randn(X_val_missing.shape[0],X_val_missing.shape[1],X_val_missing.shape[2])
  
  x_train_obs = (X_train_obs - meanTr) / stdTr
  x_test_obs  = (X_test_obs - meanTr) / stdTr
  x_val_obs  = (X_val_obs - meanTr) / stdTr

  return x_train, x_val, x_test, x_train_obs, x_val_obs,x_test_obs, x_train_missing, x_val_missing,x_test_missing, mask_train,mask_val, mask_test

def visualisation_data(x_train,x_train_obs,idx):

  plt.figure(figsize=(10,5))
  for jj in range(0,3):
    indjj = 131+jj
    plt.subplot(indjj)
    plt.plot(x_train_obs[idx,jj,:],'k.',label='Observations')
    plt.plot(x_train[idx,jj,:],'b-',label='Simulated trajectory')
    plt.legend()
    plt.xlabel('Timestep')
  plt.savefig('visualisation_dataL63_2D.pdf')

  plt.figure(figsize=(8,8))
  ax = plt.axes(projection='3d')

  ax.plot3D(x_train_obs[idx,0,:],x_train_obs[idx,1,:],x_train_obs[idx,2,:],'k.',linewidth=0.7,alpha=0.8,label='Observations')
  ax.plot3D(x_train[idx,0,:],x_train[idx,1,:],x_train[idx,2,:],'b-',linewidth=0.7,alpha=1,label='Simulated trajectory')
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')
  plt.legend()
  plt.savefig('visualisation_dataL63_3D.pdf')




























