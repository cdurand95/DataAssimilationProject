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

##### Metrics

def R_score(model, dataset):
    ''' Compute the reconstruction score for the model given in input on the complete dataset given in input
    Output : return an array with the reconstruction score on each variable, print this R-score and its mean over all the variables.'''
    
    
    x_truth=dataset['Truth']
    
    x_pred=model(torch.Tensor(dataset['Init']))
    x_pred=x_pred.detach().numpy()
    
    R_score = np.sqrt(((x_pred-x_truth)**2).mean(axis=1)).mean(axis = 0)
    print('Variables reconstruction score : {}'.format(R_score))
    print('Global reconstruction score : {}'.format(R_score.mean()))
    return R_score

def reconstruction_error_4DVar(GT, pred):
    '''Returns reconstruction score for 4D Var training '''
    R_score = 0
    x_truth=GT.detach().numpy()
    x_pred=pred.detach().numpy()
    R_score = np.sqrt(((x_pred-x_truth)**2).mean(axis=2)).mean()
    return R_score

##### Plots

def visualisation_data(X_truth, X_obs, X_mask, X_init, idx, path, type_LM = 'L63'):
    ''' Visualisation of simulated data 
    Input : 
    - X_truth : Ground Truth trajectory
    - X_obs   : Observed Points
    - X_mask  : Mask for unobserved points
    - X_init  : interpolated trajectory
    - idx     : index of example plotted
    - path    : directory path to save the figure
    - type_LM : str 'L63' or 'L96' depending on Lorenz 63 model or Lorenz 96

    Output :
    - Matplotlib figure in path directory'''


    if type_LM == 'L63':
        plt.figure(figsize=(10,5))
        for jj in range(0,3):
            indjj = 131+jj
            plt.subplot(indjj)
            plt.plot(X_obs[idx,:,jj]*X_mask[idx,:,jj],'k.',label='Observations')
            plt.plot(X_truth[idx,:,jj],'b-',label='Simulated trajectory')
            plt.plot(X_init[idx,:,jj],label='Interpolated trajectory')

            plt.legend()
            plt.xlabel('Timestep')
        plt.savefig(path+'/visualisation_dataL63.pdf',transparent=True)


    elif type_LM == 'L96':
        plt.figure(figsize=(5,10))
        

        plt.subplot(311)
        plt.imshow(X_truth[idx].transpose())
        plt.title('Truth')
        plt.colorbar()
        plt.xlabel('Timestep')

        plt.subplot(312)
        plt.imshow(X_obs[idx].transpose())
        plt.title('Observations')
        plt.colorbar()
        plt.xlabel('Timestep')

        plt.subplot(313)
        plt.imshow(X_init[idx].transpose())
        plt.title('Interpolations')
        plt.colorbar()
        plt.xlabel('Timestep')

        plt.savefig(path+'/visualisation_dataL96.pdf',transparent=True)

    else :
        print ("Please select 'L63' or 'L96' as type_LM ")

def plot_loss(model, max_epoch,path):

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
    plt.savefig(path+'/loss'+'.pdf',transparent=True)

def plot_prediction(model, idx, dataset, path, type_LM = 'L63',name='prediction'):
    
    x_pred=model(torch.Tensor(dataset['Init']))
    x_obs=dataset['Obs'][idx]
    
    
    x_pred=x_pred[idx].detach().numpy()
    x_truth=dataset['Truth'][idx]

    if type_LM == 'L63' : 
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
        plt.savefig(path+'/'+type_LM+name+'.pdf',transparent = True)

    elif type_LM == 'L96' : 
    
        time_=np.arange(0,10,0.05)

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

        plt.savefig(path+'/'+type_LM+name+'.pdf',transparent = True)
    else :
        print ("Please select 'L63' or 'L96' as type_LM ")

def evaluation_model(path,max_epoch,model_name = 'L63',idx = 25,stage = 'Test',savepath='Figures/',sparsity = 1):
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
    plt.savefig(savepath+model_name+'losses.pdf',transparent = True)
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
                      
    x_obs=dataset['Obs'][idx]

    x_truth=dataset['Truth'][idx]

    

    if model_name == 'L63':

        time_=np.arange(0,2,0.01)                 
        fig, axs = plt.subplots(4,4, sharex=True, sharey=True,figsize=(15,15))
                
        for n_layer in n_layers_list : 
            for w in dW_list :
                model = torch.load(path + '/model_n{}_dW{}_epoch{}.pth'.format(n_layer,w, max_epoch))
                x_pred=model(dataset['Init'][idx])
                x_pred=x_pred.detach().numpy()
                plt.subplot(4,4,4*(i-1)+j)
            
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
        plt.savefig(savepath+model_name+'reconstructions.pdf',transparent = True)

    if model_name == 'L96':

        time_=np.arange(0,10,0.05)                 
        fig, axs = plt.subplots(4,4, sharex=True, sharey=True,figsize=(15,15))
                
        for n_layer in n_layers_list : 
            for w in dW_list :
                model = torch.load(path + '/model_n{}_dW{}_epoch{}.pth'.format(n_layer,w, max_epoch))
                x_pred=model(dataset['Init'][idx])
                x_pred=x_pred.detach().numpy()
                plt.subplot(4,4,4*(i-1)+j)
            
                
                plt.imshow(time_,x_pred.transpose()-x_truth.transpose())
                
                plt.xlabel('Time')
                plt.ylabel('Position')
                
                plt.title('Padding : {},  Layers Numbers : {}'.format(w,n_layer),fontsize = 8)
                j+=1
        
            i+=1
            j=1
        plt.suptitle('Difference between grounf truth and prediction',fontsize = 10)
        plt.subplots_adjust( wspace=0.5, hspace=0.5)
        plt.savefig(savepath+model_name+'reconstructions.pdf',transparent = True)



def visualisation4DVar(idx, x_obs, x_GT, xhat, model_type='L63'):
    if model_type == 'L63' :
        plt.figure(figsize = (6,12))
        for kk in range(0,3):
            plt.subplot(3,1,kk+1)
            plt.plot(x_obs[idx,:,kk].detach().numpy(),'.',ms=3,alpha=0.3,label='Observations')
            plt.plot(x_GT[idx,:,kk].detach().numpy(),label='Simulated trajectory',alpha=0.8)
            plt.plot(xhat[idx,:,kk].detach().numpy(),label='4DVar Prediction',alpha=0.7)

            plt.legend()
        plt.suptitle('4DVar Reconstruction')
        plt.savefig(model_type + '4DVar.pdf',transparent = True)

    if model_type == 'L96' :
        plt.figure(figsize = (6,12))
        
        plt.subplot(3,1,1)
        plt.imshow(x_obs[idx].detach().numpy().transpose())
        plt.title('Observations')
        plt.colorbar()

        plt.subplot(3,1,2)
        plt.imshow(x_GT[idx].detach().numpy().transpose())
        plt.title('Ground truth')
        plt.colorbar()
        
        plt.subplot(3,1,3)
        plt.imshow(xhat[idx].detach().numpy().transpose())
        plt.title('4D Var Prediction')
        plt.colorbar()

        plt.suptitle('4DVar Reconstruction')
        plt.savefig(model_type + '4DVar.pdf',transparent = True)


