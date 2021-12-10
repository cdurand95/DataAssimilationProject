import os
import time
import copy
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

def L63_data(path, sparsity, var_mask):

    print('Generating L63 data')
    ts = time.time()
    data = utils.L63PatchDataExtraction(sparsity=sparsity, var_mask=var_mask)
    te = time.time()

    print('Data generated in {:.0f} s'.format(te-ts))
    print('Data shape')
    print('Training   : ' + str(data[0]['Truth'].shape))
    print('Validation : ' + str(data[1]['Truth'].shape))
    print('Test       : ' + str(data[2]['Truth'].shape))

    # Add datasave
    np.save(path + '/data0.npy', data[0])
    np.save(path + '/data1.npy', data[1])
    np.save(path + '/data2.npy', data[2])

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

    return model

### Learning rate finder

def learning_rate_finder(path, data, n_layers, dW, dimCNN, lr_list, max_epoch, patience, device):

    best_loss = np.inf
    lr_loss = []

    for lr in lr_list :

        model = define_model(data, n_layers, dW, dimCNN, lr)

        if lr==lr_list[0]:
            print('Number of trainable parameters = %d'%(sum(p.numel() for p in model.parameters() if p.requires_grad)))

        print('Currently testing learning rate', lr)

        model = train_model(model, max_epoch, patience=patience, device=device)

        loss = model.tot_loss[-1]

        lr_loss.append(loss)

        if loss < best_loss:
            best_lr = lr
            best_loss = loss
            best_model = copy.deepcopy(model)

    with open(path + '/lr_perf_n{}_dW{}_epoch{}.txt'.format(n_layers, dW, max_epoch), 'w') as file:
        file.write('Learning rate | Loss\n')
        for lr, loss in zip(lr_list, lr_loss):
            file.write('{:e} {:e}\n'.format(lr, loss))

    with open(path + 'lr_list.txt', 'a') as file:
        file.write('{}\n'.format(best_lr))

    print('Best learning rate, attained loss :', best_lr, best_model.tot_loss[-1])

    return best_model

### Loop over each hyperparameter combination

def gridsearch(path, data, n_layers_list, dW_list, dimCNN, lr_list, max_epoch, patience, device, use_lr_finder=False):

    hp_cart_product = itertools.product(n_layers_list, dW_list)

    hp_perf = []

    for i, (n_layers, dW) in enumerate(hp_cart_product):

        print('Building CNN model with {} layers and a convolution kernel of half-width {}'.format(n_layers, dW))

        if use_lr_finder:
            model = learning_rate_finder(path, data, n_layers, dW, dimCNN, lr_list, max_epoch, patience, device)
        else :
            model = define_model(data, n_layers, dW, dimCNN, lr_list[i])
            print('Number of trainable parameters = %d'%(sum(p.numel() for p in model.parameters() if p.requires_grad)))
            train_model(model, max_epoch, patience=patience, device=device)

        torch.save(model, path + '/model_n{}_dW{}_epoch{}.pth'.format(n_layers, dW, max_epoch))

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
    parser.add_argument('-l', '--n_layers', nargs='+', type=int, default=[2, 4, 6, 8],
                        help='int : number of layers')
    parser.add_argument('-w', '--dW', nargs='+', type=int, default=[1, 2, 4, 8],
                        help='int : convolution kernal half-width')
    parser.add_argument('-d', '--dimCNN', type=int, default=10,
                        help='int : number of nodes in each layer per input variable')
    parser.add_argument('-s', '--sparsity', type=float, default=1,
                        help='float [0-1] : random sparsity in observed data')
    parser.add_argument('-v', '--var_mask', nargs='+', type=int, default=[1, 1, 1],
                        help='int {0, 1} : variable mask, len must be equal to number of variable in used model')
    parser.add_argument('-p', '--patience', type=int, default=5,
                        help='int : number of epoch after which learning is stopped if validation loss does not increase')
    parser.add_argument('-e', '--max_epoch', type=int, default=250,
                        help='int : maximum number of epochs if early stopping callbacks is not triggered')
    parser.add_argument('--lr', type=float, default=None,
                        help='float : learning rate used for all models')
    parser.add_argument('--lr_max', type=float, default=1e-2,
                        help='float : maximum learning rate')
    parser.add_argument('--lr_min', type=float, default=1e-4,
                        help='float : minimum learning rate')
    parser.add_argument('-n', '--n_lr', type=int, default=17,
                        help='int : number of learning rate to iterate through')
    parser.add_argument('--lr_path', type=str, default='',
                        help='str : path to file containing learning rate for each model')

    args = parser.parse_args()

    # Parsed arguments

    n_layers_list = args.n_layers
    dW_list = args.dW
    dimCNN = args.dimCNN
    sparsity = args.sparsity
    var_mask = np.array(args.var_mask)
    patience = args.patience
    max_epoch = args.max_epoch

    n_hparams = len(n_layers_list)*len(dW_list)

    # Learning rate

    if args.lr_path != '':
        use_lr_finder = False
        with open(args.lr_path, 'r') as file:
            lr_list = [float(lr) for lr in file.read().splitlines()]
        if len(lr_list) != n_hparams:
            raise ValueError('ValueError not enough learning rates provided')
    elif args.lr is not None:
        use_lr_finder = False
        lr_list = [args.lr] * n_hparams
    else:
        use_lr_finder = True
        lr_list = np.logspace(args.lr_max, args.lr_min, num=args.n_lr).tolist()

    # Use gpu if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device :', device)

    # Set pytorch random seed for reproducibility
    torch.manual_seed(4765)

    # Define savepath
    path = 'L63gridsearch/sparsity{}'.format(sparsity)
    if not(os.path.isdir(path)):
        os.makedirs(path)

    data = L63_data(path, sparsity, var_mask)

    gridsearch(path, data, n_layers_list, dW_list, dimCNN, lr_list, max_epoch, patience, device, use_lr_finder=use_lr_finder)
