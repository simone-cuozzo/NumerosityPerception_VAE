
import os
from utils import *
from models import beta_VAE, beta_VAE2, beta_VAE3, all_conv_VAE, all_conv_VAE2, layers2_VAE, linear_VAE
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision import transforms
from scipy.io import loadmat
from scipy.stats import norm as norm
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import plotly as plo
import seaborn as sn
from sklearn import svm
from sklearn.linear_model import LogisticRegression, Perceptron
from tqdm import tqdm
from random import random
import pylab as plb
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
import optuna 

# %% DATA CONFIGURATION #######
dataset_dir = "/dataset"  # dataset directory that contains data matrices
data = data_config.config(dataset_dir + "/stim_1to20.mat")

# %% parameters dictionary #######
# models: beta_VAE, beta_VAE2, beta_VAE3, all_conv_VAE, all_conv_VAE2, layers2_VAE, linear_VAE
# annealing schedules: linear, cyclical_linear
params = {'num_workers'       : 0,
          "batch"             : 60,
          'lr'                : 0.0005,
          'w_decay'           : 0.00001,
          'num_epochs'        : 1000,
          'latent'            : 90,
          'early_stopping'    : True,
          'seed'              : 10,
          'save_model'        : True,
          'plot_losses'       : True,
          'model'             : 'layers2_VAE',
          'annealing'         : True,
          'ann_par'           : [0,4],
          'ann_type'          : 'linear'}

if params['annealing'] == True:
    params_dir = str(params['model']) + '_ep' + str(params['num_epochs']) + '_lat' + str(params['latent']) +'_seed' + str(params['seed']) + '_beta' + params['ann_type'] + str(params['ann_par'][1])
else:
    params_dir = str(params['model']) + '_ep' + str(params['num_epochs']) + '_lat' + str(params['latent']) +'_seed' + str(params['seed']) + '_beta' + str(params['ann_par'][1])

plots_dir = 'net_params/' + params_dir

if os.path.exists('net_params/' + params_dir) == True:
    params['train'] = False
    print("######   The already exixting parameters will be loaded   ######")
else:
    params['train'] = True
    os.makedirs('net_params/' + params_dir)
    print('######   The defined model will be trained   ######')
    with open('net_params/' + params_dir + '/net_parameters.txt', 'w') as f:
        print(params, file=f)


torch.manual_seed(params["seed"])
torch.cuda.manual_seed(params["seed"])
np.random.seed(params["seed"])

#%%########  DATASET e DATALOADER ########################################

train_points, test_points = train_test_split(data, test_size= 0.2, shuffle=True, random_state = 48)
train_points, val_points = train_test_split(train_points, test_size= 0.2, shuffle=True, random_state = 48)

train_points_t = data_config.Dataset(train_points, transform = transforms.Compose([data_config.ToTensor()]))
test_points_t = data_config.Dataset(test_points, transform = transforms.Compose([data_config.ToTensor()]))
validation_points_t = data_config.Dataset(val_points, transform = transforms.Compose([data_config.ToTensor()]))

batch = params["batch"]
train_points_dl= DataLoader(train_points_t, batch_size = batch, shuffle= True, num_workers = params['num_workers'], pin_memory = True)
validation_points_dl= DataLoader(validation_points_t, batch_size = len(validation_points_t), shuffle=False, num_workers = params['num_workers'], pin_memory = True)
test_points_dl= DataLoader(test_points_t, batch_size = batch, shuffle=False, num_workers = params['num_workers'], pin_memory = True)

#%% PARAMETERS ######

num_epochs = params['num_epochs']
lr = params["lr"] # Learning rate
latent = params["latent"]

if params['model'] == 'beta_VAE':
    vae_enc = beta_VAE.Encoder(latent)
    vae_dec = beta_VAE.Decoder(latent)
elif params['model'] == 'beta_VAE2':
    vae_enc = beta_VAE2.Encoder(latent)
    vae_dec = beta_VAE2.Decoder(latent)
elif params['model'] == 'beta_VAE3':
    vae_enc = beta_VAE3.Encoder(latent)
    vae_dec = beta_VAE3.Decoder(latent)
elif params['model'] == 'all_conv_VAE':
    vae_enc = all_conv_VAE.Encoder(latent)
    vae_dec = all_conv_VAE.Decoder(latent)
elif params['model'] == 'all_conv_VAE2':
    vae_enc = all_conv_VAE2.Encoder(latent)
    vae_dec = all_conv_VAE2.Decoder(latent)
elif params['model'] == 'layers2_VAE':
    vae_enc = layers2_VAE.Encoder(latent)
    vae_dec = layers2_VAE.Decoder(latent)
elif params['model'] == 'linear_VAE':
    vae_enc = linear_VAE.Encoder(latent)
    vae_dec = linear_VAE.Decoder(latent)

if params['train'] == False:
    vae_enc.load_state_dict(torch.load(plots_dir + '/vae_enc.pht'))
    vae_dec.load_state_dict(torch.load(plots_dir + '/vae_dec.pht'))


vae_enc.apply(weights_init.kaiming_uniform)
vae_dec.apply(weights_init.kaiming_uniform)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

optimizer = torch.optim.Adam([{'params': vae_enc.parameters()},{'params': vae_dec.parameters()}],
                                 lr = params["lr"], weight_decay = params['w_decay'])
vae_enc.to(device)
vae_dec.to(device)

#%%##### Training and Validation LOOP ###########

if params['train'] == True:
    recon_epochs = [i for i in np.arange(0, num_epochs+1, 10)]
    training_loss = []
    BCE_train_loss = []
    KLD_train_loss = []
    validation_loss = []
    BCE_DKL_ratio = []
    codici = dict()
    early_stopping = training.EarlyStopping(plots_dir)
    ##### TRAINING LOOP #####
    for epoch in range(num_epochs):
      print(f'#################################################### \n | EPOCH {epoch+1} | \n')
      if params['annealing'] == True:
          if params['ann_type'] == 'linear':
              beta = ann_schedules.linear(epoch, num_epochs, params['ann_par'][0],  params['ann_par'][1], annealing = True)
              print(f' beta value = {beta}')
          elif params['ann_type'] == 'cyclical_linear':
              beta = ann_schedules.cyclical_linear(epoch, num_epochs, params['ann_par'][0], params['ann_par'][1], 4, 0.5)
              print(f' beta value = {beta}')
          else:
              beta = 0
      elif params['annealing'] == False:
          beta = params['ann_par'][1]
      
      #train_loss, BCE, KLD = training.train_variable_beta(vae_enc, vae_dec, device, train_points_dl, losses.train_loss_fn_beta, optimizer)
      train_loss, BCE, KLD = training.train(vae_enc, vae_dec, device, train_points_dl, losses.train_loss_fn, optimizer, beta)
      print(f'AVERAGE TRAIN LOSS for epoch: {train_loss.data/batch}')
      BCE_DKL = BCE/KLD
      #print(f'BCE/DKL = {BCE/KLD},  \n BCE: {BCE}, \n KLD: {KLD}')
      print(f'BCE loss = {BCE}, DKL loss = {KLD}, BCE/KLD ratio = {BCE/KLD}')
      val_loss, latent_codes = testing.test(vae_enc, vae_dec, device, validation_points_dl, losses.test_loss_fn)
      if params['ann_type'] == 'ratio_scheduler':
          beta = ann_schedules.ratio_scheduler(epoch, BCE, KLD)
          print(f' beta value = {beta}')
          
      ############ PLOT RICOSTRUZIONI DEL MODELLO ############
      if epoch in recon_epochs:
        plots.reconstruction(test_points_t, 101, vae_enc, vae_dec, epoch, device)
        plt.savefig(plots_dir + f'/reconstruction_epoch{epoch+1}')
        plt.show()
        plt.close()
    
      '''##### SOLO PER PLOT CON CODICI LATENTI IN DIVERE EPOCHE #######
      if epoch in recon_epochs:
        codici.update({f'epoch{epoch}': latent_codes})
      ###############################################################'''
    
      print('\n\t VALIDATION - EPOCH %d/%d - loss: %f\n' % (epoch + 1, num_epochs, val_loss))
      
      early_stopping(val_loss, vae_enc, vae_dec)
      if early_stopping.early_stop == True:
            print("Early stopping")
            break
        
      ### SAVING LOSSES FOR FUTURE PLOTS ############################
      if params['plot_losses'] == True:
          training_loss.append(train_loss.cpu())
          BCE_train_loss.append(BCE.cpu())
          KLD_train_loss.append(KLD.cpu())
          BCE_DKL_ratio.append(BCE_DKL.cpu())
          validation_loss.append(val_loss.cpu())
          plots.loss(BCE_train_loss, 'BCE loss')
          plt.savefig(plots_dir + '/BCE_loss')
          plt.close()
          plots.loss(KLD_train_loss, 'KLD loss')
          plt.savefig(plots_dir + '/KLD_loss')
          plt.close()
          plots.loss(training_loss, 'Training loss')
          plt.savefig(plots_dir + '/Training_loss')
          plt.close()
          plots.loss(BCE_DKL_ratio, 'BCE/DKL ratio')
          plt.savefig(plots_dir + '/BCE_DKL_ratio')
          plt.close()
          plots.loss(validation_loss, 'Validation loss')
          plt.savefig(plots_dir + '/Validation_loss')
          plt.close()
      ###############################################################
      
      if epoch == num_epochs-1:
          codes = latent_codes
    
      '''########## SAVING DICTIONARIES OF MODEL PARAMETERS ############
      torch.save(vae_enc.state_dict(), 'vae_encoder_b1.pht')
      torch.save(vae_dec.state_dict(), 'vae_decoder_b1.pht')'''
      
    if params['early_stopping'] == False:
        torch.save(vae_enc.state_dict(), plots_dir +'/vae_enc.pht')
        torch.save(vae_dec.state_dict(), plots_dir +'/vae_dec.pht')

 #%% Random samples from the dataset #######

plots.data_images(24, test_points_t)

#%%#### Generation from random latent vector z #############

vae_dec.eval()
plots.gen_from_latent(24, vae_dec, latent, device)
plt.savefig(plots_dir + '/generation_from_latent')

#%%#######  RECONSTRUCTION PERFORMANCE on n random test samples ############

import random
n = 5
images = random.sample(range(1, 6000), n)

for i in images:
    plots.reconstruction(test_points_t, i, vae_enc, vae_dec, 0, device)

#%%### Creating dict {} on test set ######

torch.cuda.empty_cache()

with torch.no_grad():
    vae_enc.eval()
    vae_dec.eval()
    _, codes = testing.test(vae_enc, vae_dec, device, test_points_dl, losses.test_loss_fn)

#%% MORPHING BETWEEN 2 IMAGES

start, end = 13, 15
plots.morphing(vae_dec, codes, latent, start, end)
plt.savefig(plots_dir + f'/morph_from{start}to{end}')
plt.show()
plt.close()

#%% iNCREASE OF 1 DOT FROM A STARTING IMAGE 

start = 5
plots.increase(vae_dec, codes, latent, start)
plt.savefig(plots_dir + f'/increase_from{start}')
plt.show()
plt.close()


#%% FIRST SEPARATION small v. large, THEN PCA

data_list1 = [codes[key][0].detach().cpu().numpy() for key in codes.keys()]

n_range = np.median(data_list1[2])
ch_range = np.median(data_list1[3])
fa_range = np.median(data_list1[4])
tsa_range = np.median(data_list1[5])
a_range = np.median(data_list1[6])

data_dict = {'mu': data_list1[0], 'n': data_list1[2], 'ch': data_list1[3], 'fa': data_list1[4], 'tsa': data_list1[5], 'a': data_list1[5], 'threshold': [-1 for i in range(len(data_list1[0]))]}

for i in range(len(data_dict['mu'])):
    if data_dict['n'][i] < n_range and data_dict['ch'][i] < ch_range and data_dict['fa'][i] < fa_range and data_dict['tsa'][i] < tsa_range and data_dict['a'][i] < a_range:
        data_dict['threshold'][i] = 0
    elif data_dict['n'][i] >= n_range and data_dict['ch'][i] >= ch_range and data_dict['fa'][i] >= fa_range and data_dict['tsa'][i] >= tsa_range and data_dict['a'][i] >= a_range:
        data_dict['threshold'][i] = 1

indeces_small = [i for i in range(len(data_dict['mu'])) if data_dict['threshold'][i] == 0]
indeces_large = [i for i in range(len(data_dict['mu'])) if data_dict['threshold'][i] == 1]
indeces_pca = [i for i in range(len(data_dict['mu'])) if data_dict['threshold'][i] == 1 or data_dict['threshold'][i] == 0]

data_dict_pca = {'mu': codes['mu'][0].numpy(), 'y': codes['y'][0].numpy()}
data_dict_pca['mu'] = [torch.from_numpy(data_dict_pca['mu'][indeces_pca])]
data_dict_pca['y'] = [torch.from_numpy(data_dict_pca['y'][indeces_pca])]

pca, y, redux = plots.latent_space(method = PCA, latent_code = data_dict_pca, n_dimensions = 2, dimensions = ["mu", "y"])
plt.savefig(plots_dir + f'/small_v_large_latent_space{latent}_pca')
plt.show()
plt.close()

#%%   Dimensionality reduction and mu vectors plot   ###
# POSSIBLE DIMENSIONS: y (numerosity), ch (convex hull), fa (field area), tsa (total surface area), a (area)

pca, y, redux = plots.latent_space(method = PCA, latent_code = codes, n_dimensions = 2, dimensions = ["mu", "y"])
plt.savefig(plots_dir + f'/latent_space{latent}_pca')
plt.show()
plt.close()

#%% Cumulative Explained Variance  ###

plt.plot(np.cumsum(redux.explained_variance_ratio_))
plt.savefig(plots_dir + '/cumulative_variance')
plt.show()
plt.close()

pca_components = redux.components_[:80]
with torch.no_grad():
    vae_dec.eval()
    for i in range(len(pca_components)):
        ax = plt.subplot(8, 10, i+1)
        z = torch.Tensor(pca_components[i]).unsqueeze(0).to(device)
        img = vae_dec(z).cpu().squeeze().numpy()
        plt.suptitle("pca single components plot")
        plt.imshow(img)
        plt.axis('off')
    plt.savefig(plots_dir + '/latent_space_eigenpictures')
    plt.show()
    plt.close()
        #plt.plot(vae_dec(z).detach().cpu().reshape(100,100))

diff_pca = pca[1] - pca[2]
pca1, pca2, diff_pca = redux.inverse_transform(pca[0]), redux.inverse_transform(pca[0]), redux.inverse_transform(diff_pca)

start = pca1
n = 7
step = diff_pca/n
for i in range(n):
    with torch.no_grad():  
        vae_dec.eval()
        now = start + step * i 
        print(type(now))
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        rec = vae_dec(torch.Tensor(now).to(device).reshape(1,latent))
        plt.subplot(1,n ,i+1)
        plt.axis('off')
        plt.imshow(rec.cpu().squeeze().detach().numpy(),  cmap='gist_gray')
        plt.close()

#%% plot of average pca/numerosities classes spatial positions ####

y = np.array(y).T
a = list(zip(pca, y))
b = {'pca': pca, 'y':y}

pca_list = []
for i in range(20):
    idx = [int(x) for x in np.array(b['y'] == i+1)]
    idx = np.where(idx)[0]
    sum_pca = 0
    for j in idx:
        sum_pca += b['pca'][j]
    pca_list.append(sum_pca/len(idx))

pca1, pca2 = list(), list()
for i in pca_list:
    pca1.append(i[0])
    pca2.append(i[1])

plt.figure(figsize= (8,3))
plt.scatter(pca1, [x for x in range(len(pca1))], c = np.arange(1,21))
plt.ylim(-10000,10000)
plt.yticks([])
plt.colorbar()
plt.savefig(plots_dir + '/collapsed_dim_mental_line')
plt.show()
plt.close()

#%%#NUMEROSITY ESTIMATION

cm = numerosity_estimation.confusion_matrix(train_points_dl, test_points_dl, vae_enc) #obtain a confusion matrix and an global accuracy score
plt.savefig(plots_dir + '/confusion_matrix(estimation)')
plt.show()
plt.close()

numerosity_estimation.accuracy_plots(cm) # plots 3 images that highlight the accuracy for specific numerosities
plt.savefig(plots_dir + '/estimation_accuracy')
plt.show()
plt.close()


y = []
mu_w = []
std_w = []
for idx,row in enumerate(cm):
    row = row#[row!=0]
    x = np.arange(1,21,1)
    mu, std = numerosity_estimation.weighted_avg_and_std(x, row)
    cv = std/mu
    print(cv)
    y.append(cv)
    mu_w.append(mu)
    std_w.append(std)
plt.scatter(x, y)
plt.plot(x, y)
plt.title('Coefficient of variation v. Numerosity class')
plt.savefig(plots_dir + '/cv_v_numclass')
plt.show()
plt.close()

#%% Probability Density Function v. Numerosity Class
for i in range(len(cm)):
    x = np.arange(1,21,0.1)
    y = norm.pdf(x, mu_w[i], std_w[i])
    plt.plot(x, y)
    #plt.ylim(-0.1, 1.1)
    plt.yscale('linear')
    plt.title('Probability Density Function v. Numerosity Class')
plt.savefig(plots_dir + '/gaussian_fit')
plt.show()
plt.close()

par_list = numerosity_estimation.gaussian_fit(cm)

#%% BINARY CLASSIFICATION and NUMEROSITY COMPARISON TASK**

from sklearn import svm
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import roc_curve, roc_auc_score
from tqdm import tqdm


train_comparison_data = numerosity_comparison.data(train_points_t, vae_enc, 1)
test_comparison_data = numerosity_comparison.data(test_points_t, vae_enc, 11)

X = list(train_comparison_data['X'])
Y = list(train_comparison_data['Y'])
X = X[:1000]
Y = Y[:1000]

##### DIFFERENT MODEL FITTING FOR DIFFERENT PSYCHOMETRIC CURVES #########
clf_svc = svm.LinearSVC(random_state=0)
clf_svc.fit(X,Y)

clf_reg = LogisticRegression(random_state=0)
clf_reg.fit(X,Y)

clf_perceptron = Perceptron(random_state=0)#tol=1e-3, random_state=0)
clf_perceptron.fit(X,Y)


#numerosity of interest for plot (without 1)
numerosity_list = [1/2, 3/5, 2/3, 3/4, 4/5, 5/6, 6/7, 9/10, 10/11,
           11/10, 10/9, 7/6, 6/5, 5/4, 4/3, 3/2, 5/3, 2]

## creazione di un dizionario che ha come chiavi le effettive ratio di interesse per
## il plot convertite in stringhe
from collections import OrderedDict
from fractions import Fraction
num = OrderedDict()
for ratio in numerosity_list:
    num[str(Fraction(ratio).limit_denominator(20))] = test_comparison_data[test_comparison_data['ratios'] == ratio]


########### prediction 'greater' values ################
## of the first 100 values for every numerosity class ##

probabilities1_regressor, probabilities_correct_regressor = numerosity_comparison.predict_greater(clf_reg, num)
probabilities1_svc, probabilities_correct_svc = numerosity_comparison.predict_greater(clf_svc, num)
probabilities1_perceptron, probabilities_correct_perceptron = numerosity_comparison.predict_greater(clf_perceptron, num)

import matplotlib.ticker
from scipy.optimize import curve_fit

popt_perceptron, _ = curve_fit(numerosity_comparison.fit_function, numerosity_list, probabilities1_perceptron, method='lm', maxfev=20000)
popt_svc, _ = curve_fit(numerosity_comparison.fit_function, numerosity_list, probabilities1_svc, method='lm', maxfev=20000)
popt_regressor, _ = curve_fit(numerosity_comparison.fit_function, numerosity_list, probabilities1_regressor, method='lm',maxfev=20000)


#%% Perceptron %

print(f'Perceptron Weber Fraction: {popt_perceptron[0]}')
numerosity_comparison.weber_plot('black', probabilities1_perceptron, popt_perceptron,'Perceptron')
plt.title('Dataset of {:} samples, w = {:.4f}' .format(len(X), popt_perceptron[0]))
plt.savefig(plots_dir + '/perceptron_weber_fraction.png')
plt.show()
plt.close()

#%% Logistical Regressor

print(f'Logistic regressor Weber Fraction: {popt_regressor[0]}')
numerosity_comparison.weber_plot('red', probabilities1_regressor, popt_regressor, 'Regressor')
plt.title('Dataset of {:} samples, w = {:.4f}' .format(len(X), popt_regressor[0]))
plt.savefig(plots_dir + '/regressor_weber_fraction.png')
plt.show()
plt.close()

#%% Support Vector Classifier (Linear) 

print(f'Linear support vector classifier Weber Fraction: {popt_svc[0]}')
numerosity_comparison.weber_plot('gray', probabilities1_svc, popt_svc, 'SVC')
plt.title('Dataset of {:} samples, w = {:.4f}' .format(len(X), popt_svc[0]))
plt.savefig(plots_dir + '/SVC_weber_{:.4f}.png')
plt.show()
plt.close()

#%% MLP prototypical --> classifier inversion to generate prototipycal numerosities top-down
fa = [np.array(codes['fa'][0][i]) for i in range(6760)]
fa_min = 3740
fa_max = 28983
fa_step = (fa_max - fa_min)/20
for i in range(len(fa)):
    for j in range(20):
        if fa[i] >= fa_min + (fa_step*j) and fa[i] <= fa_min + fa_step*(j+1):
            fa[i] = torch.tensor(j+1)
        else:
            continue
        
# conversione tsa  range per one hot encoding
tsa = [np.array(codes['tsa'][0][i]) for i in range(6760)]
tsa_min = 834
tsa_max = 6481
tsa_step = (tsa_max - tsa_min)/20
for i in range(len(tsa)):
    for j in range(20):
        if tsa[i] >= tsa_min + (tsa_step*j) and tsa[i] <= tsa_min + tsa_step*(j+1):
            tsa[i] = torch.tensor(j+1)
        else:
            continue

classifier_data = [(codes['mu'][0][i], codes['y'][0][i]-1, fa[i]-1, tsa[i]-1) for i in range(6760)]
#regressor_data_x = [(np.array(codes_classifier['mu'][0][i])) for i in range(3380)]
#regressor_data_y = [(np.array(codes_classifier['y'][0][i])) for i in range(3380)]

classifier_data_ds = data_config.Dataset(classifier_data)
#data_new = data_config.config_alternative("NumStim_1to20_100x100_TR.mat")
#data_new = [(i[0], i[1]-1) for i in data_new]
#data_new = data_config.Dataset(data_new)
#new_dl = DataLoader(data_new, batch_size = len(data_new))

target_num = 1
classifier_data_10y = []
classifier_data_10mu = []
for i in range(1000):
    if classifier_data[i][1] == torch.tensor(target_num-1):
        classifier_data_10mu.append(classifier_data[i][0])
        classifier_data_10y.append(classifier_data[i][1])

plt.hist(classifier_data_10mu[0], bins=params['latent'], alpha=0.5, label="data1")
plt.hist(classifier_data_10mu[1], bins=params['latent'], alpha=0.5, label="data2")
plt.hist(classifier_data_10mu[2], bins=params['latent'], alpha=0.5, label="data2")
plt.hist(classifier_data_10mu[3], bins=params['latent'], alpha=0.5, label="data2")
plt.hist(classifier_data_10mu[4], bins=params['latent'], alpha=0.5, label="data2")
plt.title(f'5 latent vectors z of examples with numerosity "{target_num}"')
plt.savefig(plots_dir + f'/latent_vectors_hist_num_{target_num}')

#classifier_data = [(codes_classifier['mu'][0][i], codes_classifier['y'][0][i]) for i in range(3380)]
#classifier_data = [(codes['mu'][0][i], codes['y'][0][i]-1) for i in range(6760)]
#regressor_data_x = [(np.array(codes_classifier['mu'][0][i])) for i in range(3380)]
#regressor_data_y = [(np.array(codes_classifier['y'][0][i])) for i in range(3380)]


model = classifier.Forward_MLP(params['latent']).to(device)
classifier_dataloader = DataLoader(classifier_data_ds, batch_size = 200)
classifier.train_forward(model, device, classifier_dataloader)

net_parameters = model.state_dict()
weights1 = np.array(net_parameters['lin1.0.weight'].detach().cpu())
bias1 = np.transpose(np.array(net_parameters['lin1.0.bias'].detach().cpu())).reshape(1, 200)
weights2 = np.array(net_parameters['lin2.0.weight'].detach().cpu())
bias2 = np.transpose(np.array(net_parameters['lin2.0.bias'].detach().cpu())).reshape(1,500)
weights3 = np.array(net_parameters['lin3.0.weight'].detach().cpu())
bias3 = np.transpose(np.array(net_parameters['lin3.0.bias'].detach().cpu())).reshape(1,60)

inverse_classifier_10mu = []
for i in range(10):
    resu = model(classifier_data_10mu[i].to(device))
    net_parameters = model.state_dict()
    out3 = (resu.detach().cpu() - bias3) @ np.transpose(np.linalg.pinv(weights3))
    out2 = (out3 - bias2) @ np.transpose(np.linalg.pinv(weights2))
    for i in out3[0]:
        if i>=0:
            i = i
        else:
            i = i*0.01
    for i in out2[0]:
        if i>=0:
            i = i
        else:
            i = i*0.01
    out = (out2 - bias1) @ np.transpose(np.linalg.pinv(weights1))
    inverse_classifier_10mu.append(out.detach().cpu())
    
plt.hist(inverse_classifier_10mu[0], bins=params['latent'], alpha=0.5, label="data1")
plt.hist(inverse_classifier_10mu[1], bins=params['latent'], alpha=0.5, label="data2")
plt.hist(inverse_classifier_10mu[2], bins=params['latent'], alpha=0.5, label="data2")
plt.hist(inverse_classifier_10mu[3], bins=params['latent'], alpha=0.5, label="data2")
plt.hist(inverse_classifier_10mu[4], bins=params['latent'], alpha=0.5, label="data2")
plt.title(f'Inverse classification of 5 latent vectors with numerosity "{target_num}"')
plt.savefig(plots_dir + f'/latent_vectors_hist_num_{target_num}_inverse_classifier')

print(f'lin1_w = {weights1.shape} ; bias1 = {bias1.shape} ; lin2_w {weights2.shape} ; bias2 {bias2.shape}')
#torch.save(net_parameters, plots_dir +'/inverse.pht')

 #%% PLots MLP
step = np.arange(0,1, 1/20)
one_to_twenty = []
for i in range(20):
    out_z = classifier.inverse_relu_MLP(i+1, i+1, i+1, weights1 , bias1, weights2 , bias2, weights3 , bias3)
    one_to_twenty.append(out_z)

for i in range(20):
    model.eval()
    with torch.no_grad():
        plt.subplot(2, 10, i+1)
        out = vae_dec(torch.tensor(one_to_twenty[i]).to(torch.float32).to(device))
        plt.imshow(out[0].detach().cpu().reshape(100,100).numpy(), cmap= 'gray')
        plt.axis('off')
    plt.savefig(plots_dir + '/1to20prototypical_MLP.png')

  #%% Perceptron prototypical

# conversione fa range per one hot encoding
fa = [np.array(codes['fa'][0][i]) for i in range(6760)]
fa_min = 3740
fa_max = 28983
fa_step = (fa_max - fa_min)/20
for i in range(len(fa)):
    for j in range(20):
        if fa[i] >= fa_min + (fa_step*j) and fa[i] <= fa_min + fa_step*(j+1):
            fa[i] = torch.tensor(j+1)
        else:
            continue
        
# conversione tsa  range per one hot encoding
tsa = [np.array(codes['tsa'][0][i]) for i in range(6760)]
tsa_min = 834
tsa_max = 6481
tsa_step = (tsa_max - tsa_min)/20
for i in range(len(tsa)):
    for j in range(20):
        if tsa[i] >= tsa_min + (tsa_step*j) and tsa[i] <= tsa_min + tsa_step*(j+1):
            tsa[i] = torch.tensor(j+1)
        else:
            continue

classifier_data = [(codes['mu'][0][i], codes['y'][0][i]-1, fa[i]-1, tsa[i]-1) for i in range(6760)]
#regressor_data_x = [(np.array(codes_classifier['mu'][0][i])) for i in range(3380)]
#regressor_data_y = [(np.array(codes_classifier['y'][0][i])) for i in range(3380)]

classifier_data_10y = []
classifier_data_10mu = []
for i in range(1000):
    if classifier_data[i][1] == torch.tensor(19):
        classifier_data_10mu.append(classifier_data[i][0])
        classifier_data_10y.append(classifier_data[i][1])

plt.hist(classifier_data_10mu[0], bins=params['latent'], alpha=0.5, label="data1")
plt.hist(classifier_data_10mu[1], bins=params['latent'], alpha=0.5, label="data2")
plt.hist(classifier_data_10mu[2], bins=params['latent'], alpha=0.5, label="data2")
plt.hist(classifier_data_10mu[3], bins=params['latent'], alpha=0.5, label="data2")
plt.hist(classifier_data_10mu[4], bins=params['latent'], alpha=0.5, label="data2")

classifier_data_ds = data_config.Dataset(classifier_data)


model = classifier.Forward_Perceptron(params['latent']).to(device)
classifier_dataloader = DataLoader(classifier_data_ds, batch_size = 200)
classifier.train_forward(model, device, classifier_dataloader)

inverse_classifier_10mu = []
for i in range(10):
    resu = model(classifier_data_10mu[i].to(device))
    net_parameters = model.state_dict()
    weights1 = np.array(net_parameters['lin1.0.weight'].detach().cpu())
    bias1 = np.transpose(np.array(net_parameters['lin1.0.bias'].detach().cpu())).reshape(1,60)
    out = (resu.detach().cpu() - bias1) @ np.transpose(np.linalg.pinv(weights1))
    for i in out[0]:
        if i>=0:
            i = i
        else:
            i = i*0.05
    inverse_classifier_10mu.append(out.detach().cpu())
    
plt.hist(inverse_classifier_10mu[0], bins=params['latent'], alpha=0.5, label="data1")
plt.hist(inverse_classifier_10mu[1], bins=params['latent'], alpha=0.5, label="data2")
plt.hist(inverse_classifier_10mu[2], bins=params['latent'], alpha=0.5, label="data2")
plt.hist(inverse_classifier_10mu[3], bins=params['latent'], alpha=0.5, label="data2")
plt.hist(inverse_classifier_10mu[4], bins=params['latent'], alpha=0.5, label="data2")

net_parameters = model.state_dict()
weights1 = np.array(net_parameters['lin1.0.weight'].detach().cpu())
bias1 = np.transpose(np.array(net_parameters['lin1.0.bias'].detach().cpu())).reshape(1,60)


print(f'lin1_w = {weights1.shape} ; bias1 = {bias1.shape} ; lin2_w {weights2.shape} ; bias2 {bias2.shape}')
#torch.save(net_parameters, plots_dir +'/inverse.pht')

 #%% PLOTS Perceptron
one_to_twenty = []
for i in range(20):
    out_z = classifier.inverse_relu_Perceptron(i+1, 0, 0, weights1 , bias1)
    one_to_twenty.append(out_z)

for i in range(20):
    model.eval()
    with torch.no_grad():
        plt.subplot(2, 10, i+1)
        out = vae_dec(torch.tensor(one_to_twenty[i]).to(torch.float32).to(device))
        plt.imshow(out[0].detach().cpu().reshape(100,100).numpy(), cmap= 'gray')
        plt.axis('off')
    plt.savefig(plots_dir + '/1to20prototypical_Perceptron.png')

### HYPEROPTIMIZATION OF A LINEAR CLASSIFIER ###
#study = classifier.classifier_hyperopt(classifier_data, device, params)


#%% k-Nearest Neighbor for classification
from sklearn.neighbors import NearestCentroid as nc

classifier_data = {'x': [np.array(codes['mu'][0][i]) for i in range(6760)], 'y': [np.array(codes['y'][0][i]) for i in range(6760)]}
ne_ce = nc()
# training knn classifier
ne_ce.fit(classifier_data['x'], np.ravel(classifier_data['y']))
# come invertire i pesi per ottenere vettori latenti dei centrodi di classe?
centroids = ne_ce.centroids_
model.eval()
ind = 2
with torch.no_grad():
    out = vae_dec(torch.tensor(centroids[ind-1]).to(torch.float32).reshape(1,params['latent']).to(device))
    out2 = out.detach().cpu().reshape(100,100).numpy()
    plt.imshow(out2, cmap= 'gray')
    plt.axis('off')
    plt.savefig(plots_dir + f'/1to20prototypical_nearestcentroid_{ind}.png')
    plt.show()
    plt.close()
    
  #%% ITERATIVE DENOISING PROCEDURE
import random
num_iteration = 10
plots.denoising(num_iteration, params['latent'], vae_enc, vae_dec, device)





