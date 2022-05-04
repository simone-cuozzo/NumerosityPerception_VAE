# -*- coding: utf-8 -*-
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision import transforms
import numpy as np
from sklearn.linear_model import Perceptron 
import matplotlib.pyplot as plt
from utils import training, testing
import scipy
import optuna

## Multi-layer Perceptron model class ##
class Forward_MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lin1 = nn.Sequential(nn.Linear(input_dim, 200),
                                 nn.LeakyReLU(0.05))
        self.lin2 = nn.Sequential(nn.Linear(200, 500),
                                 nn.LeakyReLU(0.05))
        self.lin3 = nn.Sequential(nn.Linear(500, 60))
    def forward(self, x):
        out = self.lin3(self.lin2(self.lin1(x)))
        return out

## Perceptron model class ##
class Forward_Perceptron(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lin1 = nn.Sequential(nn.Linear(input_dim, 60),
                                 nn.LeakyReLU(0.05))
    
    def forward(self, x):
        out = self.lin1(x)
        return out

### Inversion of a Multi-layer Perceptron classifier with ReLU activation function ###
def inverse_relu_MLP(num, fa, tsa, weights1, bias1, weights2, bias2, weights3, bias3):
    one_hot = np.zeros((1,60))
    one_hot[0][num - 1] = 1
    one_hot[0][fa + 20-1] = 1
    one_hot[0][tsa + 40-1] = 1
    out3 = (one_hot - bias3) @ np.transpose(np.linalg.pinv(weights3))
    out2 = (out3 - bias2) @ np.transpose(np.linalg.pinv(weights2))
    for i in out3[0]:
        if i>=0:
            i = i
        else:
            i = i*0.05
    for i in out2[0]:
        if i>=0:
            i = i
        else:
            i = i*0.05
    out = (out2 - bias1) @ np.transpose(np.linalg.pinv(weights1))
    return out

### Inversion of a Perceptron classifier with ReLU activation function ###
def inverse_relu_Perceptron(num, fa, tsa, weights1, bias1):
    one_hot = np.zeros((1,60))
    one_hot[0][num - 1] = 1
    one_hot[0][fa + 20-1] = 1
    one_hot[0][tsa + 40-1] = 1
    out = (one_hot - bias1) @ np.transpose(np.linalg.pinv(weights1))
    for i in out[0]:
        if i>=0:
            i = i
        else:
            i = i*0.05
    return out

def inverse_sigmoid(num_class, weights1, bias1, weights2, bias2):
    one_hot = np.zeros((1,20))
    one_hot[0][num_class-1] = 1
    out2 = (one_hot - bias2) @ np.transpose(np.linalg.pinv(weights2))
    out2 = scipy.special.logit(out2)
    out = (out2 - bias1) @ np.transpose(np.linalg.pinv(weights1))
    return out

### Training of a generic classifier function ###
def train_forward(model, device, dataloader, lr = 0.00005, weight_decay = 0.000001):
    loss_classifier = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    for epoch in range(1000):
        print(f'#################################################### \n | EPOCH {epoch+1} | \n')
        model.train()
        for mu, y, fa, tsa in dataloader:
              
              y = F.one_hot(y.to(torch.int64), 20).to(torch.float32).flatten(start_dim = 1).to(device)
              #print(y.size())
              fa = F.one_hot(fa.to(torch.int64), 20).to(torch.float32).flatten(start_dim = 1).to(device)
              #print(f'fa size = {fa.size()}')
              tsa = F.one_hot(tsa.to(torch.int64), 20).to(torch.float32).flatten(start_dim = 1).to(device)
              #print(f' tsa size = {tsa.size()}')
              y = torch.cat((y, fa, tsa), dim = 1)

              mu = mu.to(device)
              y_hat = model(mu)
              loss = loss_classifier(y_hat, y)
              optimizer.zero_grad()
              loss.backward()
              optimizer.step()
        print(f'loss = {loss}')
            

### Classifier hyperoptimization through Optuna library ###
def classifier_hyperopt(data, device, params):
    class Forward(nn.Module):
        def __init__(self, params):
            super().__init__()
            self.lin = nn.Sequential(nn.Linear(params['latent'], 20))
            
        def forward(self, x):
            out = self.lin(x)
            return out
    
    def objective(trial):
        par = {'lr'  :  trial.suggest_categorical("lr", [0.01, 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005]),
               'batch' : trial.suggest_categorical("batch", [80, 150, 300, 400, 500]),
               'w_decay' : trial.suggest_categorical("w_decay", [0.01, 0.001, 0.005, 0.0001, 0.0005, 0.00001, 0.00005])}
        
        model = Forward(params['latent'], par)
        model.to(device)
        batch = par['batch']
        dataloader = DataLoader(data, batch_size = batch, shuffle = True)
        optimizer = optim.Adam(model.parameters(), par['lr'], weight_decay = par['w_decay'])
        model.train()
        loss_epoch = []
        for i in range(15):
            for mu, y in dataloader:
                y = F.one_hot(y.to(torch.int64), 20)
                y = y.flatten(start_dim=1).to(torch.float).to(device)
                mu = mu.to(device)
                
                y_hat = model(mu)
                
                loss_fn = nn.MSELoss()
                loss = loss_fn(y_hat, y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                final_loss = min(loss_epoch)
        loss_epoch.append(np.average(loss))
        print(f'{loss_epoch[i]}')
        return final_loss 
        trial.report(final_loss, i)
    study = optuna.create_study()
    study.optimize(objective, n_trials = 2, gc_after_trial = True)
    print(f'{study.best_trial()}')
    print(f'{study.best_params}')
    return study.trials_dataframe()

### Test function for a classifier ###
def test(encoder, decoder, device, dataloader, loss_fn):
    encoder.eval()
    decoder.eval()
    conc_out = []
    conc_label = []
    latent_codes = dict(mu = list(), y = list())
    means, labels = list(), list()
    with torch.no_grad(): 
        for image_batch, y in dataloader:
            
            image_batch = image_batch.to(device)
            z, mu, log_var = encoder(image_batch)
            reconstruction= decoder(z)
            means.append(mu.detach().cpu())
            labels.append(y.detach().cpu())
            conc_out.append(reconstruction.cpu())
            conc_label.append(image_batch.cpu())

        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label) 
        val_loss = loss_fn(conc_out, conc_label)
    
    latent_codes['mu'].append(torch.cat(means))
    latent_codes['y'].append(torch.cat(labels))
        
    return val_loss.data, latent_codes


def LeakyReLU_inv(alpha,x):
  output = np.copy(x)
  output[ output < 0 ] /= alpha
  return output