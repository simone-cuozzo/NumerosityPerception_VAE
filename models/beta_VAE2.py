# -*- coding: utf-8 -*-

### Symmetrical beta-VAE as described in "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework, Higgins et al. 2016" ###

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision import transforms

par = {"nodes" : [450, 450], "filters" : [16,8,8,16,16,16], "kernels": [2,4,8,8,4,2], "strides": [2,2,4,4,2,2]} #original from thesis

class Encoder(nn.Module):
    def __init__(self, latent, dim=5):
        super().__init__()

        self.encoder_cnn1 = nn.Sequential(nn.Conv2d(1, par["filters"][0], par["kernels"][0], par["strides"][0]),
                                         nn.LeakyReLU(),
                                         nn.BatchNorm2d(par["filters"][0]))
        
        self.encoder_cnn2 = nn.Sequential(nn.Conv2d(par["filters"][0], par["filters"][1], par["kernels"][1], par["strides"][1]),
                                         nn.LeakyReLU(),
                                         nn.BatchNorm2d(par["filters"][1]),
                                         nn.Dropout2d(0.2))

        self.encoder_cnn3 = nn.Sequential(nn.Conv2d(par["filters"][1], par["filters"][2], par["kernels"][2], par["strides"][2]),
                                         nn.LeakyReLU(),
                                         nn.BatchNorm2d(par["filters"][2]))

        self.flatten = nn.Flatten(start_dim=1)

        self.encoder_lin = nn.Sequential(nn.Linear(par["filters"][2]*dim*dim , par["nodes"][0]),
                                         nn.LeakyReLU(),
                                         nn.BatchNorm1d(par["nodes"][0]))

        self.mu = nn.Sequential(nn.Linear(par["nodes"][0], latent))
        self.log_var = nn.Sequential(nn.Linear(par["nodes"][0], latent))


    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5*log_var)
            eps = torch.randn_like(std)
            return mu + eps*std  #secondo il paper
        else:
            return mu

    def forward(self, x):
        #forward del encoder
        x = self.encoder_cnn3(self.encoder_cnn2(self.encoder_cnn1(x)))
        x = self.flatten(x)
        x = self.encoder_lin(x)
        mu = self.mu(x)
        log_var = self.log_var(x)
        ###reparameterization
        z= self.reparameterize(mu, log_var)
        return z, mu, log_var


class Decoder(nn.Module):
    def __init__(self, latent, dim=5):      
        super().__init__()
        
        self.decoder_lin1 = nn.Sequential(nn.Linear(latent, par["nodes"][1]),
                                         nn.LeakyReLU(),
                                         nn.BatchNorm1d(par["nodes"][1]))

        self.decoder_lin2 = nn.Sequential(nn.Linear(par["nodes"][1], par["filters"][3]*dim*dim),
                                         nn.LeakyReLU(),
                                         nn.BatchNorm1d(par["filters"][3]*dim*dim))
                                         
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(par["filters"][3],dim,dim))  #H=W=2
        
        self.decoder_cnn1 = nn.Sequential(nn.ConvTranspose2d(par["filters"][3], par["filters"][4], par["kernels"][3], par["strides"][3]),
                                         nn.LeakyReLU(),
                                         nn.BatchNorm2d(par["filters"][4]))
        
        self.decoder_cnn2 = nn.Sequential(nn.ConvTranspose2d(par["filters"][4], par["filters"][5], par["kernels"][4], par["strides"][4]),
                                         nn.LeakyReLU(),
                                         nn.BatchNorm2d(par["filters"][5]))
        
        self.decoder_cnn3 = nn.Sequential(nn.ConvTranspose2d(par["filters"][5],1, par["kernels"][5], par["strides"][5]))

    def forward(self, z):
    #forward del decoder
        x = self.decoder_lin2(self.decoder_lin1(z))
        x = self.unflatten(x)
        x = self.decoder_cnn3(self.decoder_cnn2(self.decoder_cnn1(x)))
        reconstruction = torch.sigmoid(x)
        return reconstruction

