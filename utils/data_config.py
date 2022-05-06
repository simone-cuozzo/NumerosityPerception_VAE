from scipy.io import loadmat
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision import transforms
import numpy as np
import mat73

### dataset configuration from MATLAB matrices. The images matrices are preprocessed with 2 hard thresholds at 0 and 1 ###
def config(dataset_dir):
    path = Path(dataset_dir)
    data = loadmat(path)
    data.pop('__header__')
    data.pop('__version__')
    data.pop('__globals__')
    idx = list(data.keys())
    for i in idx:
        data[i] = data[i].transpose()
    data["D"][data["D"]<=0] = 0
    data["D"][data["D"]>=1] = 1
        
    data_dict = {"D" : data[idx[1]].reshape(33800,1,100,100), "N" :data[idx[3]], "CH" : data[idx[0]],
                 "FA": data[idx[2]], "TSA": data[idx[4]], "A" : data[idx[5]]} 
    
    key = [key for key in data_dict.keys()]
    data = [(data_dict[key[0]][i], data_dict[key[1]][i], data_dict[key[2]][i], data_dict[key[3]][i], data_dict[key[4]][i], data_dict[key[5]][i]) for i in range(len(data_dict["N"]))]
    return data

def config_alternative(dataset):
    path = Path("C:/Users/micheluzzo/Desktop/Simone/NumerosityPerception/dataset/" + dataset)
    data = loadmat(path)
    data.pop('__header__')
    data.pop('__version__')
    data.pop('__globals__')
    idx = list(data.keys())
    for i in idx:
        data[i] = data[i].transpose()
    data["D"][data["D"]<=0] = 0
    data["D"][data["D"]>=1] = 1
        
    data_dict = {"D" : data[idx[1]].reshape(3380,1,100,100), "N" :data[idx[3]], "CH" : data[idx[0]],
                 "FA": data[idx[2]], "TSA": data[idx[4]], "A" : data[idx[5]]} 
    
    key = [key for key in data_dict.keys()]
    data = [(data_dict[key[0]][i], data_dict[key[1]][i], data_dict[key[2]][i], data_dict[key[3]][i], data_dict[key[4]][i], data_dict[key[5]][i]) for i in range(len(data_dict["N"]))]
    return data

def config_73(dataset):
    path = Path("C:/Users/micheluzzo/Desktop/Simone/NumerosityPerception/dataset/" + dataset)
    data = loadmat(path)
    idx = list(data.keys())
    for i in idx:
        data[i] = data[i].transpose()
    data["D"][data["D"]<=0] = 0
    data["D"][data["D"]>=1] = 1
        
    data_dict = {"D" : data[idx[1]].reshape(3380, 100, 100), "N" :data[idx[3]], "CH" : data[idx[0]],
                 "FA": data[idx[2]], "TSA": data[idx[4]], "A" : data[idx[5]]} 
    
    key = [key for key in data_dict.keys()]
    data = [(data_dict[key[0]][i], data_dict[key[1]][i], data_dict[key[2]][i], data_dict[key[3]][i], data_dict[key[4]][i], data_dict[key[5]][i]) for i in range(len(data_dict["N"]))]
    return data

### Pytorch ad hoc dataset class ###
class Dataset(Dataset):
    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #sample_x = self.data[idx][0]
        #sample_y = self.data[idx][1]
        sample = self.data[idx]
        if self.transform:
            #data = (self.transform(data[0]), self.transform(data[1]), self.transform(data[2]), self.transform(data[3]), self.transform(data[4]), self.transform(data[5]))
            sample = self.transform(sample)
        #return (sample_x, torch.tensor(sample_y))
        return sample

### Ad hoc ToTensor transformation ###
class ToTensor(object):
    def __call__(self, sample):
        a,b,c,d,e,f = sample
        return (torch.Tensor(a), torch.Tensor(b), torch.Tensor(c), torch.Tensor(d), torch.Tensor(e),torch.Tensor(f))
        
