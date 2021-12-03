import os
import time
import itertools
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import utils

# Use gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device :', device)

### Hyperparameters to loop over

n_layers_list = [2, 4, 6, 8]
dW_list = [1, 2, 4, 8]

dimCNN = 10

sparsity = 1
max_epochs = 100

lr = 5e-3

### Data generation

print('Generating L63 data')
ts = time.time()
data = utils.L63PatchDataExtraction(sparsity=sparsity)
te = time.time()

print('Data generated in {:.0f} s'.format(te-ts))
print('Data shape')
print('Training   : ' + str(data[0]['Truth'].shape))
print('Validation : ' + str(data[1]['Truth'].shape))
print('Test       : ' + str(data[2]['Truth'].shape))

### Define model for a set of hyperparameters

def define_model(data, n_layers, dW, dimCNN, lr):
    model = utils.CNN(data, n_layers=n_layers, dW=dW, dimCNN=dimCNN, lr=lr)
    return model

### Train model

def train_model(model, max_epochs):

    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=5, verbose=False, mode='min')

    if device.type == 'cuda':
        trainer = pl.Trainer(max_epochs=max_epochs, callbacks=[early_stop_callback], gpus=1, auto_select_gpus=True)
    else :
        trainer = pl.Trainer(max_epochs=max_epochs, callbacks=[early_stop_callback])

    trainer.fit(model)

    return None

### Save training data

### Loop over each hyperparameter combination

def gridsearch():

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

        train_model(model, max_epochs)

        torch.save(model, path + '/model_n{}_dW{}_epoch{}.pth'.format(n_layers, dW, max_epochs))
        # model = torch.load('model.pth')

        hp_perf.append((model.tot_loss[-1], n_layers, dW))

    hp_perf.sort()

    with open(path + '/perf_{}epochs.txt'.format(max_epochs), 'w') as file:
        for loss, n_layers, dW in hp_perf[::-1] :
            print('Loss : {:.2f}, n_layers : {}, dw : {}'.format(loss, n_layers, dW))
            file.write('Loss : {:.2f}, n_layers : {}, dw : {}\n'.format(loss, n_layers, dW))

    return None

if __name__ == "__main__":
    gridsearch()
