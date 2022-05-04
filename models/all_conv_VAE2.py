# -*- coding: utf-8 -*-

# ALL-CONVOLUTIONAL VAE inspired by "Striving for Simplicity: The All Convolutional Net, Springenberg et al. 2014"
# all pooling layers are replaced by simple convolutional layers with stride > 1

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision import transforms

par = {    'batch'             : 60,
           'lr'                : 0.0005,
           'w_decay'           : 0.001,
           'filter1'           : 8,
           'filter2'           : 32,
           'node1'             : 200,
           'node2'             : 200,
           'kernel1'           : 3,
           'kernel2'           : 6,
           'kernel_pool'       : 4,
           'drop1'             :0.15,#trial.suggest_categorical("drop1", [0, 0.1, 0.2, 0.3]),
           'drop2'             :0.15,#trial.suggest_categorical("drop2", [0, 0.1, 0.2, 0.3]),
           'drop3'             :0, #trial.suggest_categorical("drop3", [0, 0.1, 0.2, 0.3]),
           'drop4'             :0,   #trial.suggest_categorical("drop4", [0, 0.1, 0.2, 0.3]),
           'drop5'             :0,   #trial.suggest_categorical("drop5", [0, 0.1, 0.2, 0.3]),
           'drop6'             :0.15,#trial.suggest_categorical("drop6", [0, 0.1, 0.2, 0.3]),
           'drop7'             :0.15} #trial.suggest_categorical("drop7", '[0, 0.1, 0.2', 0.3])}

'''
def pad_generator(kernel):
        if kernel == 3:
            padding = 1
        elif kernel == 5:
            padding = 2
        elif kernel == 7:
            padding = 3
        return padding
    
padding1 = pad_generator(par['kernel1'])
padding2 = pad_generator(par['kernel2'])
padding3 = pad_generator(par['kernel3'])
'''

class Encoder(nn.Module):
    def __init__(self, latent, par=par, dim = 5):
        super().__init__()
        
        self.cnn1 = nn.Sequential(nn.Conv2d(1, par['filter1'], par['kernel1'], 1, padding = 1),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm2d(par["filter1"]),
                                   nn.Dropout2d(par['drop1']))
        
        self.fakepool1 = nn.Conv2d(par['filter1'], par['filter1'], par['kernel_pool'], par['kernel_pool'])
                                   
        self.cnn2 = nn.Sequential(nn.Conv2d(par['filter1'], par['filter2'], par['kernel2'], 1),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm2d(par["filter2"]),
                                   nn.Dropout2d(par['drop2']))
        
        self.fakepool2 = nn.Conv2d(par['filter2'], par['filter2'], par['kernel_pool'], par['kernel_pool'])
        
        self.flatten = nn.Flatten(start_dim=1)

        self.enc_lin = nn.Sequential(nn.Linear(par["filter2"]*dim*dim , par["node1"]),
                                         nn.LeakyReLU(),
                                         nn.BatchNorm1d(par["node1"]),
                                         nn.Dropout(par['drop3']))
        
        self.mu = nn.Sequential(nn.Linear(par["node1"], latent))
        self.log_var = nn.Sequential(nn.Linear(par["node1"], latent))
        
    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5*log_var)
            eps = torch.randn_like(std)
            return mu + eps*std  #secondo il paper
        else:
            return mu
            
        
    def forward(self, x):
        x = self.fakepool1(self.cnn1(x))
        x = self.fakepool2(self.cnn2(x))
        x = self.flatten(x)
        x = self.enc_lin(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var
    

class Decoder(nn.Module):
    def __init__(self, latent, par=par, dim = 5):
        super().__init__()
        
        self.dec_lin1 = nn.Sequential(nn.Linear(latent, par["node2"]),
                                         nn.LeakyReLU(),
                                         nn.BatchNorm1d(par["node2"]),
                                         nn.Dropout(par["drop4"]))
        
        self.dec_lin2 = nn.Sequential(nn.Linear(par["node2"], par["filter2"]*dim*dim),
                                         nn.LeakyReLU(),
                                         nn.BatchNorm1d(par["filter2"]*dim*dim),
                                         nn.Dropout(par["drop5"]))
        
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(par["filter2"],dim,dim))
        
        self.fakedepool1 = nn.ConvTranspose2d(par["filter2"], par["filter2"], par['kernel_pool'], par['kernel_pool'])
        
        self.decnn1 = nn.Sequential(nn.ConvTranspose2d(par["filter2"], par["filter1"], par['kernel2'], 1),
                                    nn.LeakyReLU(),
                                    nn.BatchNorm2d(par["filter1"]),
                                    nn.Dropout2d(par['drop6']))
        
        self.fakedepool2 = nn.ConvTranspose2d(par["filter1"], par["filter1"], par['kernel_pool'], par['kernel_pool'])
        
        self.decnn2 = nn.Sequential(nn.ConvTranspose2d(par["filter1"], 1, par['kernel1'], 1, padding = 1))

                                         
    def forward(self, z):
    #forward del decoder
        x = self.unflatten(self.dec_lin2(self.dec_lin1(z)))
        x = self.decnn1(self.fakedepool1(x))
        x = self.decnn2(self.fakedepool2(x))
        reconstruction = torch.sigmoid(x)
        return reconstruction

