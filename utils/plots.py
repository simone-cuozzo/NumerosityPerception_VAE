# -*- coding: utf-8 -*-
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
import random
import numpy as np
from utils import plots

### latent space plot after applying a dimensionality reduction technique ###
def latent_space(method, latent_code, n_dimensions, dimensions = []):
    # method = PCA or TSNE
    # n_dimensions = 2 or 3
    # dimensions = list of str between "y", "ch", "fa", "tsa", "a"
    X, Y, E = list(), list(), list()
    X.append(latent_code[dimensions[0]][0]) #aggiungere se si usa codes normale
    Y.append(latent_code[dimensions[1]][0])
    if n_dimensions == 2:
        print(f'### 2-D plot of {dimensions[0]} vectors and color gradient based on {dimensions[1]} ###')
        if method == PCA:
            redux = PCA()
            redux_fit = redux.fit_transform(X[-1])
            print(redux.explained_variance_ratio_)
            redux_1, redux_2 = [redux_fit[:,0]], [redux_fit[:,1]]
            Y[0] = [int(i) for i in Y[0]]
            plt.scatter(redux_1, redux_2, c = Y)
            plt.xlim(-5,5)
            plt.ylim(-5,5)
            plt.colorbar()
            
        if method == TSNE:
            redux1 = PCA(n_components=50)
            redux_fit = redux1.fit_transform(X[-1])
            redux2 = TSNE(n_components=2, perplexity = 300)
            redux_fit = redux2.fit_transform(redux_fit)
            redux_1, redux_2 = [redux_fit[:,0]], [redux_fit[:,1]]
            Y[0] = [int(i) for i in Y[0]]
            plt.scatter(redux_1, redux_2, c = Y)
            plt.colorbar()
        
    elif n_dimensions == 3:
        if method == PCA:
            redux = PCA(n_components=3)
            redux_fit = redux.fit_transform(X[-1])
            print(redux.explained_variance_ratio_)
            redux_1, redux_2, redux_3 = [redux_fit[:,0]], [redux_fit[:,1]], [redux_fit[:,2]]
            Y[0] = [int(i) for i in Y[0]]
            fig = plt.figure(figsize = (10,7))
            ax = plt.axes(projection = "3d")
            ax.scatter3D(redux_1. redux_2, redux_3, c=Y[0])
            plt.colorbar()
            
        if method == TSNE:
            redux1 = PCA(n_components=50)
            redux_fit = redux.fit_transform(X[-1])
            redux2 = TSNE(n_components=3)
            redux_fit = redux2.fit_transform(redux_fit)
            redux_1, redux_2, redux_3 = [redux_fit[:,0]], [redux_fit[:,1]], [redux_fit[:,2]]
            Y[0] = [int(i) for i in Y[0]]
            fig = plt.figure(figsize = (10,7))
            ax = plt.axes(projection = "3d")
            ax.scatter3D(redux_1. redux_2, redux_3, c=Y[0])
            plt.colorbar()
    else:
        print("The plot can have only 2 or 3 dimensions")
    return redux_fit, Y, redux

### plot of the model reconstruction of a specific data sample ###
def reconstruction(data, img, encoder, decoder, epoch, device, iterative = False, output = False):
    # img = index of an image from data
    if iterative == False:
        img1 = data[img][0].unsqueeze(0).to(device)
        encoder.eval()
        decoder.eval() 
        if output == False:
            with torch.no_grad(): 
                z1,_,_ = encoder(img1)
                rec_img1  = decoder(z1)
                fig, axs = plt.subplots(1, 2)
                axs[0].imshow(img1.cpu().squeeze().numpy(), cmap='gist_gray')
                axs[0].set_title('Original img', fontsize='x-small')
                axs[1].imshow(rec_img1.cpu().squeeze().numpy(), cmap='gist_gray')
                axs[1].set_title('Reconstructed img (EPOCH %d)' % (epoch + 1), fontsize='x-small')
                plt.tight_layout()
                plt.axis('off')
        elif output == True:
            with torch.no_grad(): 
                z1,_,_ = encoder(img1)
                rec_img1  = decoder(z1)
                fig, axs = plt.subplots(1, 2)
                axs[0].imshow(img1.cpu().squeeze().numpy(), cmap='gist_gray')
                axs[0].set_title('Original img', fontsize='x-small')
                axs[1].imshow(rec_img1.cpu().squeeze().numpy(), cmap='gist_gray')
                axs[1].set_title('Reconstructed img (%s)' % (epoch), fontsize='x-small')
                plt.tight_layout()
                plt.axis('off')
                return rec_img1, z1
    elif iterative == True:
        img1 = img1.to(device)
        encoder.eval()
        decoder.eval() 
        if output == True:
            with torch.no_grad(): 
                z1,_,_ = encoder(img1)
                rec_img1  = decoder(z1)
                fig, axs = plt.subplots(1, 2)
                axs[0].imshow(img1.cpu().squeeze().numpy(), cmap='gist_gray')
                axs[0].set_title('Original img', fontsize='x-small')
                axs[1].imshow(rec_img1.cpu().squeeze().numpy(), cmap='gist_gray')
                axs[1].set_title('Reconstructed img (%s)' % (epoch), fontsize='x-small')
                plt.tight_layout()
                plt.axis('off')
            return rec_img1, z1

### plot of the gradual increase of numerosity +1 from a starting latent vector ###
def increase(decoder, vectors_dict, latent, start, coeff=1):   
  # vectors_dict = dictionary of the test set latent vectors after VAE has been trained
  # latent = VAE latent space dimension
  # coeff = multiplicative factor that determines the increase of numerosity at each step
  N = 7
  v_1 = [int(x) for x in np.array((vectors_dict['y'][0] == 1).nonzero(as_tuple=True)[0])]
  v_2 = [int(x) for x in np.array((vectors_dict['y'][0] == 2).nonzero(as_tuple=True)[0])]
  diff = vectors_dict['mu'][0][v_2[0]] - vectors_dict['mu'][0][v_1[0]]
  v = [int(x) for x in np.array((vectors_dict['y'][0] == start).nonzero(as_tuple=True)[0])]
  v = vectors_dict['mu'][0][v[random.randint(0,100)]]
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  step = diff/N
  for i in range(N):
    with torch.no_grad():
        decoder.eval()
        now = v + step * i * coeff 
        rec = decoder(now.to(device).reshape(1, latent))
        plt.subplot(1,N,i+1)
        plt.axis('off')
        plt.imshow(rec.cpu().squeeze().detach().numpy(),  cmap='gist_gray')


### generation of new images from latent space normal sampling ###
def gen_from_latent(images_num, decoder, latent, device):
    # images_num = number of images for the plot --> only multiple of 3 or 4 
    decoder.eval()
    z_grid = torch.randn((images_num, latent))
    z_grid = z_grid.to(device)
    with torch.no_grad():
        rec_grid = decoder(z_grid)
    for i in range(images_num):
        if images_num==1:
            plt.suptitle("Normal sampling of {images_num} latent vectors")
            plt.imshow(rec_grid[i].cpu().squeeze().detach().numpy(),  cmap='gist_gray')
        elif images_num%4 == 0:
            plt.subplot(4, int(images_num/4), i+1)
            plt.axis('off')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.suptitle(f"Normal sampling of {images_num} latent vectors")
            plt.imshow(rec_grid[i].cpu().squeeze().detach().numpy(),  cmap='gist_gray')
        elif images_num%3 == 0:
            plt.subplot(3,int(images_num/3), i+1)
            plt.axis('off')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.suptitle(f"Normal sampling of {images_num} latent vectors")
            plt.imshow(rec_grid[i].cpu().squeeze().detach().numpy(),  cmap='gist_gray')
        else:
            print("Choose another number of images for the plot")
        
### Plots of images randomly selected from the dataset ###
def data_images(num_images, data):
    if num_images == 1:
        img = random.sample(range(len(data)), k=1)
        plt.axis('off')
        plt.suptitle("Random sample from the dataset")
        plt.imshow(data[img[0]][0].squeeze(), cmap='gist_gray')
    elif num_images%4 == 0:
        img = random.sample(range(len(data)), k = num_images)
        for i in range(len(img)):
            plt.subplot(4, int(len(img)/4), i+1)
            plt.axis('off')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.subplots_adjust(hspace = .05, wspace = .05)
            plt.suptitle("Random samples from the dataset")
            plt.imshow(data[img[i]][0].squeeze(), cmap='gist_gray')
    elif num_images%3 == 0:
        img = random.sample(range(len(data)), k = num_images)
        for i in range(len(img)):
            plt.subplot(3, int(len(img)/3), i+1)
            plt.axis('off')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.subplots_adjust(hspace = .05, wspace = .05)
            plt.suptitle("Random samples from the dataset")
            plt.imshow(data[img[i]][0].squeeze(), cmap='gist_gray')        
        
### morphing between 2 randomly choesen images of selected numerosity through latent vectors algebraic operations ###
def morphing(decoder, vectors_dict, latent, start, end):
    v_1 = [int(x) for x in np.array((vectors_dict['y'][0] == start).nonzero(as_tuple=True)[0])]
    v_2 = [int(x) for x in np.array((vectors_dict['y'][0] == end).nonzero(as_tuple=True)[0])]
    start = vectors_dict['mu'][0][v_1[random.randint(0,100)]]
    v = vectors_dict['mu'][0][v_2[random.randint(0,100)]]  - start
    num_images = 7
    coeff = 1
    step = v/num_images
    for i in range(num_images):
        with torch.no_grad():
            decoder.eval()
            now = start + step * i * coeff
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            rec = decoder(now.to(device).reshape(1,latent))
            plt.subplot(1,num_images,i+1)
            plt.axis('off')
            plt.imshow(rec.cpu().squeeze().detach().numpy(),  cmap='gist_gray')
            
### model loss plot ###
def loss(loss, title=str):
    plt.plot(loss)
    plt.title(title)
        
### iterative denoising process of a randomly selected latent vector ###
def denoising(num_iter, latent, encoder, decoder, device):
    sample = torch.randn(1, latent)
    decoder.eval()
    with torch.no_grad():
        original = decoder(sample.to(device))
        plt.axis('off')
        plt.imshow(original.cpu().squeeze().numpy(), cmap='gist_gray')
        plt.title('random sample from latent space')
        count = 0
        while count <= num_iter:
            count += 1
            plt.axis('off')
            z, _, _ = encoder(original)
            rec_img = decoder(z)
            fig, axs = plt.subplots(1,2)
            axs[0].imshow(original.cpu().squeeze().numpy(), cmap='gist_gray')
            axs[0].set_title('Original img', fontsize='x-small')
            axs[1].imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
            axs[1].set_title('Denoised img', fontsize='x-small')
            original = rec_img
        

def denoising_random_sample(num_iter, encoder, latent, decoder, device):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        while count <= num_iter:
            sample = torch.randn(1, latent)
            count += 1
            rec_img, z = plots.reconstruction(dataset, rec_img, encoder, decoder, 'training ended', device, iterative = True, output = True)
                    
        
def positive_tail_plot(color, model_prob, model_popt, model_label):
    ax = plt.axes(xscale='log', yscale='linear')
    ax.grid(True)
    ax.scatter(numerosity_list[8:], model_prob, marker='o', color=color, label = model_label + ' predictions')
    ax.plot(numerosity_list[8:], func(numerosity_list[8:], *model_popt), 'b--', alpha=0.75, label = 'Sigmoid fit')
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    plt.xlim(0.995, 2.02)
    plt.ylim(0.5, 1.05)
    plt.xticks(xticks[5:], ticks_labels[5:], fontsize=7.2)
    ax.legend(loc='upper left')
    plt.ylabel('p(Choose "greater")')
    plt.xlabel('Numerosity Ratio')

        
        
        
        