# -*- coding: utf-8 -*-

### 2 convolutional layers beta-VAE ###

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision import transforms

par = {    'batch'             : 20,
           'lr'                : 0.0001,
           'w_decay'           : 0.00001,
           'filter1'           : 32,
           'filter2'           : 32,
           'node1'             : 800,
           'drop1'             : 0.3,#trial.suggest_categorical("drop1", [0, 0.1, 0.2, 0.3]),
           'drop2'             : 0,#trial.suggest_categorical("drop2", [0, 0.1, 0.2, 0.3]),
           'drop3'             : 0, #trial.suggest_categorical("drop3", [0, 0.1, 0.2, 0.3]),
           'drop4'             : 0,   #trial.suggest_categorical("drop4", [0, 0.1, 0.2, 0.3]),
           'drop5'             : 0.3,
           'drop6'             : 0}   #trial.suggest_categorical("drop5", [0, 0.1, 0.2, 0.3]),} #trial.suggest_categorical("drop7", '[0, 0.1, 0.2', 0.3])}


class Encoder(nn.Module):
    def __init__(self, latent, par=par, dim = 12):
        super().__init__()
        
        self.cnn1 = nn.Sequential(nn.Conv2d(1, par['filter1'], 4, 2, padding = 1),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm2d(par["filter1"]),
                                   nn.Dropout2d(par['drop1']))
                                           
        self.cnn2 = nn.Sequential(nn.Conv2d(par['filter1'], par['filter2'], 7,4, padding=1),
                                   nn.LeakyReLU(),
                                   nn.BatchNorm2d(par["filter2"]),
                                   nn.Dropout2d(par['drop2']))

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
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.flatten(x)
        x = self.enc_lin(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var
    

class Decoder(nn.Module):
    def __init__(self, latent, par=par, dim = 12):
        super().__init__()
        
        self.dec_lin1 = nn.Sequential(nn.Linear(latent, par["node1"]),
                                         nn.LeakyReLU(),
                                         nn.BatchNorm1d(par["node1"]),
                                         nn.Dropout(par["drop4"]))
        
        self.dec_lin2 = nn.Sequential(nn.Linear(par["node1"], par["filter2"]*dim*dim),
                                         nn.LeakyReLU(),
                                         nn.BatchNorm1d(par["filter2"]*dim*dim),
                                         nn.Dropout(par["drop5"]))
        
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(par["filter2"],dim,dim))
                
        self.decnn1 = nn.Sequential(nn.ConvTranspose2d(par["filter2"], par["filter1"], 7,4, padding = 1),
                                    nn.LeakyReLU(),
                                    nn.BatchNorm2d(par["filter1"]),
                                    nn.Dropout2d(par['drop6']))
                
        self.decnn2 = nn.Sequential(nn.ConvTranspose2d(par["filter1"], 1, 4,2))

                                         
    def forward(self, z):
    #forward del decoder
        x = self.unflatten(self.dec_lin2(self.dec_lin1(z)))
        x = self.decnn1(x)
        x = self.decnn2(x)
        reconstruction = torch.sigmoid(x)
        return reconstruction



