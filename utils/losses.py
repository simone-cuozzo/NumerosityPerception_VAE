# -*- coding: utf-8 -*-
import torch 
import matplotlib.pyplot as plt

def train_loss_fn(reconstruction, original_img, mu, log_var, beta):
  loss = torch.nn.BCELoss(reduction='sum')
  BCE= loss(reconstruction, original_img)
  DKL = -0.5* torch.sum(1 + log_var - mu.pow(2) - log_var.exp())  #il primo valore dell'equazione è il parametro beta
  return BCE + (DKL * beta), BCE, DKL*beta

def train_loss_fn_beta(reconstruction, original_img, mu, log_var):
  loss = torch.nn.BCELoss(reduction='sum')
  BCE= loss(reconstruction, original_img)
  DKL = -0.5* torch.sum(1 + log_var - mu.pow(2) - log_var.exp())  #il primo valore dell'equazione è il parametro beta
  return BCE + DKL, BCE, DKL

def test_loss_fn(reconstruction, original_img):
    loss_fn = torch.nn.MSELoss(reduction='mean')
    return loss_fn(reconstruction, original_img)

def loss_plot(BCE, KLD, validation):
    plt.plot(BCE, label = 'BCE loss', color = 'blue')
    plt.legend(loc = 'upper right')
    plt.show()
    plt.close()
    plt.plot(KLD, label = 'KLD loss', color = 'orange')
    plt.legend(loc = 'lower right')
    plt.show()
    plt.close()
    plt.plot(validation, label = 'validation loss', color = 'red')
    plt.legend(loc = 'upper right')
    plt.show()
    plt.close()
