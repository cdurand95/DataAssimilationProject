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



