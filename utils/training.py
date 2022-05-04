# -*- coding: utf-8 -*-
import numpy as np
import torch

### Train function ###
def train(encoder, decoder, device, dataloader, loss_fn, optimizer, beta):
    encoder.train()
    decoder.train()
    for image_batch, *y in dataloader:  
        beta = beta  
        
        image_batch = image_batch.to(device)
        
        z, mu, log_var = encoder(image_batch)
        reconstruction = decoder(z)

        loss, BCE, DKL = loss_fn(reconstruction, image_batch, mu, log_var, beta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.data, BCE.data, DKL.data


def train_variable_beta(encoder, decoder, device, dataloader, loss_fn, optimizer):
    encoder.train()
    decoder.train()
    for image_batch, *y in dataloader:
        image_batch = image_batch.to(device)
        
        z, mu, log_var = encoder(image_batch)
        reconstruction = decoder(z)

        loss, BCE, DKL = loss_fn(reconstruction, image_batch, mu, log_var)
        beta = BCE/DKL
        loss = BCE + ((beta)*DKL)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.data, BCE.data, DKL.data

### Early stopping class for training ###
# patience = number of epochs with no progress in training before training is stopped
# delta = additive factor which indicates the actual performance improvement during training; default = 0
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, path, patience=20, delta=0):

        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.early_stop = False
        
    def __call__(self, val_loss, encoder, decoder):
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, encoder, decoder)
        elif score > self.best_score - (score/20):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, encoder, decoder)
            self.counter = 0
        
    def save_checkpoint(self, val_loss, encoder, decoder):
        '''Saves model when validation loss decrease.'''
        print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(encoder.state_dict(), self.path +'/vae_enc.pht')
        torch.save(decoder.state_dict(), self.path +'/vae_dec.pht')
        self.val_loss_min = val_loss


