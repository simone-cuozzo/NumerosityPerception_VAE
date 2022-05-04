# -*- coding: utf-8 -*-

### NUMEROSITY COMPARISON TASK, comparison between 2 different latent vectors that encode different numerosities information ###

from sklearn import svm
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import roc_curve, roc_auc_score
from tqdm import tqdm
from torch.utils.data import Dataset, TensorDataset, DataLoader
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt

### numerosity comparison dataset generation, obtained by randomly cycling through the original dataset and concatenating 
### pairs of different latent vectors, storing the ratio of their numerosities. All ratios outside the range 0.5 <= r >= 2 are excluded ###
def data(data, encoder, cycles):
    comparison_dl= DataLoader(data, batch_size = 2, shuffle= True)

    train_z, train_labels, train_ratios = list(), list(), list()
    for i in tqdm(range(cycles)):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        torch.manual_seed(i)
        torch.cuda.manual_seed(i)
        encoder.eval()
        for x, y, *_ in comparison_dl:
            x = x.to(device)
            y = y.to(device)
            z,_,_ = encoder(x)
            z = torch.cat((z[0],z[1]))
            train_z.append(z.clone().detach().cpu().numpy())
            
            if y[1] > y[0]:
                label = 1
            if y[1] < y[0]:
                label = 0
            if y[1] == y[0]:
                label = np.random.choice([0, 1])
            train_labels.append(label)
    
            ratio = ((y[1])/(y[0])).detach().cpu().numpy()
            train_ratios.append(ratio)
    values_train = {'X': train_z, 'Y': train_labels, 'ratios': train_ratios}
    df_train = pd.DataFrame(values_train)
    df_train = df_train[df_train['ratios'] <= 2]
    df_train = df_train[df_train['ratios'] >= 0.5]
    df_train = df_train[df_train['ratios']!= 1.0] ## removing all numerosities pairs with ratio = 1
    return df_train


def predict_greater(clf_model, data):
  list_1 = []
  list_correct = []
  for ratio in data:
      predictions = clf_model.predict(list(data[ratio]['X'][:100]))
      pred_target = list(zip(predictions, data[ratio]['Y'][:100]))
      count_1 = 0
      count_correct = 0
      for p in predictions:
          if p==1:
              count_1 += 1
      for (p,y) in pred_target:
          if p==y:
              count_correct += 1

      prob1 = count_1/len(predictions)
      prob_correct = count_correct/len(pred_target)
      list_1.append(prob1)
      list_correct.append(prob_correct)
  return list_1, list_correct


def fit_function(r, w):
    # definition of the sigmoid curve
    from scipy.stats import norm
    return norm.sf(0, loc = np.log(r), scale = np.sqrt(2) * w)

def weber_plot(color, model_prob, model_popt, model_label):
    numerosity_list = [1/2, 3/5, 2/3, 3/4, 4/5, 5/6, 6/7, 9/10, 10/11, 11/10, 10/9, 7/6, 6/5, 5/4, 4/3, 3/2, 5/3, 2]
    ax = plt.axes(xscale='log', yscale='linear')
    ax.grid(True)
    ax.scatter(numerosity_list, model_prob, marker='o', color=color, label = model_label + ' predictions')
    ax.plot(numerosity_list, fit_function(numerosity_list, *model_popt), 'b--', alpha=0.75, label = 'Sigmoidal fit' )
    ticks_labels = [str(x) for x in numerosity_list]
    plt.xticks(numerosity_list, ticks_labels, fontsize=7.2)
    ax.legend(loc='upper left')
    plt.ylabel('p(Choose "greater")')
    plt.xlabel('Numerosity Ratio')