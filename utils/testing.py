# -*- coding: utf-8 -*-

import torch
from utils import ann_schedules, losses

### Test function ###
# return --> val.loss = validation/test loss 
#            latent_codes = dictionary that contains mu and log_vars vectors, numerosity vectors and continuous non-numerical features vectors 
def test(encoder, decoder, device, dataloader, loss_fn):
    encoder.eval()
    decoder.eval()
    conc_out = []
    conc_label = []
    latent_codes = dict(mu = list(), logvar = list(), y = list(), ch = list(), fa = list(), tsa = list(), a = list())
    means, logvars, labels, CH, FA, TSA, A = list(), list(), list(), list(), list(), list(), list()
    with torch.no_grad(): 
        for image_batch, y, ch, fa, tsa, a in dataloader:
            image_batch = image_batch.to(device)

            z, mu, log_var = encoder(image_batch)
            reconstruction= decoder(z)
            means.append(mu.detach().cpu())
            logvars.append(log_var.detach().cpu())
            labels.append(y.detach().cpu())
            CH.append(ch.detach().cpu())
            FA.append(fa.detach().cpu())
            TSA.append(tsa.detach().cpu())
            A.append(a.detach().cpu())

            conc_out.append(reconstruction.cpu())
            conc_label.append(image_batch.cpu())

        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label) 
        val_loss = loss_fn(conc_out, conc_label)
    
    latent_codes['mu'].append(torch.cat(means))
    latent_codes['logvar'].append(torch.cat(logvars))
    latent_codes['y'].append(torch.cat(labels))
    latent_codes['ch'].append(torch.cat(CH))
    latent_codes['fa'].append(torch.cat(FA))
    latent_codes['tsa'].append(torch.cat(TSA))
    latent_codes['a'].append(torch.cat(A))
        
    return val_loss.data, latent_codes

