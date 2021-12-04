import os
import time
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import utils

### Data generation

def L63_data(sparsity):

    print('Generating L63 data')
    ts = time.time()
    data = utils.L63PatchDataExtraction(sparsity=sparsity)
    te = time.time()

    print('Data generated in {:.0f} s'.format(te-ts))
    print('Data shape')
    print('Training   : ' + str(data[0]['Truth'].shape))
    print('Validation : ' + str(data[1]['Truth'].shape))
    print('Test       : ' + str(data[2]['Truth'].shape))

    return data

### Define model for a set of hyperparameters

def define_model(data, n_layers, dW, dimCNN, lr):
    model = utils.CNN(data, n_layers=n_layers, dW=dW, dimCNN=dimCNN, lr=lr)
    return model

### Train model

def train_model(model, max_epoch, patience=5, device=torch.device('cpu')):

    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=patience, verbose=False, mode='min')

    if device.type == 'cuda':
        trainer = pl.Trainer(max_epochs=max_epoch, callbacks=[early_stop_callback], gpus=1, auto_select_gpus=True)
    else :
        trainer = pl.Trainer(max_epochs=max_epoch, callbacks=[early_stop_callback])

    trainer.fit(model)

    return None

### Loop over each hyperparameter combination

def gridsearch(data, n_layers_list, dW_list, dimCNN, lr, max_epoch, patience, device):

    path = 'L63gridsearch/sparsity{}'.format(sparsity)
    if not(os.path.isdir(path)):
        os.makedirs(path)

    hp_cart_product = itertools.product(n_layers_list, dW_list)

    hp_perf = []

    for n_layers, dW in hp_cart_product:

        print('Building CNN model with {} layers and a convolution kernel of half-width {}'.format(n_layers, dW))

        model = define_model(data, n_layers, dW, dimCNN, lr)

        print(model)
        print('Number of trainable parameters = %d'%(sum(p.numel() for p in model.parameters() if p.requires_grad)))

        train_model(model, max_epoch, patience=patience, device=device)

        torch.save(model, path + '/model_n{}_dW{}_epoch{}.pth'.format(n_layers, dW, max_epoch))
        # model = torch.load('model.pth')

        hp_perf.append((model.tot_loss[-1], n_layers, dW))

    hp_perf.sort()

    with open(path + '/perf_{}epochs.txt'.format(max_epoch), 'w') as file:
        for loss, n_layers, dW in hp_perf[::-1] :
            print('Loss : {:.2f}, n_layers : {}, dw : {}'.format(loss, n_layers, dW))
            file.write('Loss : {:.2f}, n_layers : {}, dw : {}\n'.format(loss, n_layers, dW))

    return None

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument('-m', '--model', type=str, choices=['L63', 'L96'], default='L63')
    parser.add_argument('-s', '--sparsity', type=float, default=1)
    parser.add_argument('-e', '--max_epoch', type=int, default=100)
    parser.add_argument('-l', '--learning_rate', type=float, default=5e-3)
    parser.add_argument('-p', '--patience', type=int, default=5)
    parser.add_argument('-d', '--dimCNN', type=int, default=10)

    args = parser.parse_args()

    # Hyperparameters to loop over
    n_layers_list = [2, 4, 6, 8]
    dW_list = [1, 2, 4, 8]

    # Parsed arguments
    sparsity = args.sparsity
    max_epoch = args.max_epoch
    lr = args.learning_rate
    patience = args.patience
    dimCNN = args.dimCNN

    # Use gpu if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device :', device)

    # Set pytorch random seed for reproducibility
    torch.manual_seed(4765)

    data = L63_data(sparsity)

    gridsearch(data, n_layers_list, dW_list, dimCNN, lr, max_epoch, patience, device)
