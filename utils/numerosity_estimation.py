# -*- coding: utf-8 -*-

# NUMEROSITY ESTIMATION TASK FUNCTIONS #
from sklearn.neural_network import MLPClassifier as MLPClassifier
from sklearn.datasets import make_classification
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from torch.utils.data import Dataset, TensorDataset, DataLoader
import numpy as np
import torch
import seaborn as sns
from scipy.stats import norm
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import ConfusionMatrixDisplay as display
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import ConfusionMatrixDisplay as display
from scipy.optimize import curve_fit as cf
import scipy.interpolate as ip

# Fit of a Logistic Regressor on the numerosity estimation dataset and test 
## performances of the model plotted through a confusion matrix
def confusion_matrix(train_dataloader, test_dataloader, encoder):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    train_dict = dict(X=list(), Y=list())
    with torch.no_grad():
        encoder.eval()
        for x,y,*_ in train_dataloader:
            x = x.to(device)
            z,_,_ = encoder(x)
            train_dict['X'].append(np.array(z.detach().cpu()))
            train_dict['Y'].append(np.array(y.detach().cpu()))
        train_dict['X'] = np.concatenate(train_dict['X'])
        train_dict['Y'] = np.concatenate(train_dict['Y'])
        
    clf = LogisticRegression(max_iter = 5000)
    clf.fit(train_dict['X'], train_dict['Y'])
    test_dict = dict(X=list(), Y=list())
    with torch.no_grad():
        encoder.eval()
        for x,y,*_ in test_dataloader:
            x = x.to(device)
            z,_,_ = encoder(x)
            test_dict['X'].append(np.array(z.detach().cpu()))
            test_dict['Y'].append(np.array(y.detach().cpu()))
    test_dict['X'] = np.concatenate(test_dict['X'])
    test_dict['Y'] = np.concatenate(test_dict['Y'])
    
    predictions = clf.predict(test_dict['X'])
    pred_list = list(zip(predictions, test_dict['Y']))
    
    ### CLF ACCURACY ###########################
    print(f'The classifier has an accuracy of %.2f percent' % (clf.score(test_dict['X'], test_dict['Y']) * 100))
    
    ### CONFUSION MATRIX #######################
    labels = clf.classes_
    confusion_mat = cm(test_dict['Y'], predictions, normalize = 'true')
    sns.color_palette("muted")
    ### il plot della matrice di confusione adesso è normalizzato on [0,1] --> accuratezza ##########
    sns.heatmap(confusion_mat[::-1]/confusion_mat[::-1].sum(1), annot=False,fmt="d", cmap='viridis', linecolor='black',
                linewidths=0.0000038, xticklabels=[str(int(x)) for x in labels] ,yticklabels=[str(int(x)) for x in labels[::-1]])
    #plt.colorbar(confusion_mat/confusion_mat.sum(1))
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    return confusion_mat


### Model's accuracy plots of 1 to 20 numerosities, 1 to 5 numerosities and only odd numerosities ###
def accuracy_plots(confusion_matrix):
    fig, axs = plt.subplots(3, figsize=(12,13))
    xticks1 = [int(x) for x in np.linspace(0,7,8)]
    xlabels1 = [str(x+1) for x in xticks1]
    axs[0].plot(confusion_matrix[0][:8], label = '1')
    axs[0].plot(confusion_matrix[1][:8], label = '2')
    axs[0].plot(confusion_matrix[2][:8], label = '3')
    axs[0].plot(confusion_matrix[3][:8], label = '4')
    axs[0].plot(confusion_matrix[4][:8], label = '5')
    axs[0].legend()
    axs[0].set_xticks(xticks1, xlabels1)
    axs[0].set_xlabel('Numerosity class')
    axs[0].set_ylabel('Classification Probability')
    
    ########## PLOT only ODD NUMEROSITY ##############
    xticks2 = [x for x in np.arange(20, dtype=int)]
    xlabels2 = [str(x+1) for x in xticks2]
    for i in xticks2:
        axs[1].plot(confusion_matrix[i])
        axs[1].set_xticks(xticks2, xlabels2)
        axs[1].set_xlabel('Numerosity class')
        axs[1].set_ylabel('Classification Probability')
    ##################################################
    
    ####### PLOT EVERY NUMEROSITY ####################
    xticks3 = [int(x) for x in np.arange(20, dtype=int)]
    xlabels3 = [str(x+1) for x in xticks3]
    for i in xticks3:
        axs[2].plot(confusion_matrix[i])
        axs[2].set_xticks(xticks3, xlabels3)
        axs[2].set_xlabel('Numerosity class')
        axs[2].set_ylabel('Classification Probability')
    
    return confusion_matrix

### Gaussian fit of the accuraxcy plots ###
def gaussian_fit(confusion_matrix):
    list_m_s = list()
    for i in range(len(confusion_matrix)):
        #vect = confusion_matrix[i][confusion_matrix[i]!=0]
        vect = confusion_matrix[i]
        print(vect)
        dist = norm.fit(vect, loc=i)
        mean = norm.mean(dist)
        std = norm.std(dist)
        list_m_s.append((mean,std))
    return list_m_s
        #plt.plot(y, norm.pdf(confusion_matrix[i]))

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    mu_w = np.average(values, weights=weights)
    # Fast and numerically precise:
    sigma_w = np.average((values - mu_w)**2, weights = weights)
    return mu_w, np.sqrt(sigma_w)

