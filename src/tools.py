import pandas as pd
import numpy as np

import os
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm_notebook
import multiprocessing

from PIL import Image
from .inception import InceptionV3
from tqdm import tqdm_notebook as tqdm
from .fid_score import calculate_frechet_distance
from .distributions import LoaderSampler
import torchvision.datasets as datasets
import h5py
from torch.utils.data import TensorDataset, ConcatDataset

import gc

from torch.utils.data import Subset, DataLoader, Dataset, ConcatDataset
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, Lambda, Pad, CenterCrop, RandomResizedCrop
from torchvision.datasets import ImageFolder


def load_dataset(name, path, img_size=64, batch_size=64, device='cuda'):
    if name in ['shoes', 'handbag', 'outdoor', 'church']:
        dataset = h5py_to_dataset(path, img_size)
    elif name in ['celeba_female', 'aligned_anime_faces', 'celeba', 'ffhq_faces', 'celeba_female_subset', 'aligned_anime_faces_subset']:
        transform = Compose([Resize((img_size, img_size)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = ImageFolder(path, transform=transform)
    elif name in ['comic_faces', 'comics_v1', 'chairs']:
        transform_train = Compose([Resize((img_size, img_size)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), RandomHorizontalFlip()])
        transform_test = Compose([Resize((img_size, img_size)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = ImageFolder(path, transform=transform_train)
        dataset_test = ImageFolder(path, transform=transform_test)
    elif name in ['fruits-360']:
        transform = Compose([Pad(14, fill=(255,255,255)), Resize(img_size), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = ImageFolder(path+'/'+'Training/', transform=transform)
        test_set = ImageFolder(path+'/'+'Test/', transform=transform)
    elif name in ['afhq_cat']:
        transform_train = Compose([Resize((img_size, img_size)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), RandomHorizontalFlip()])
        transform_test = Compose([Resize((img_size, img_size)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = ImageFolder(path+'/'+'train/cat/', transform=transform_train)
        test_set = ImageFolder(path+'/'+'val/cat/', transform=transform_test)
    elif name in ['afhq_dog']:
        transform_train = Compose([Resize((img_size, img_size)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), RandomHorizontalFlip()])
        transform_test = Compose([Resize((img_size, img_size)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = ImageFolder(path+'/'+'train/dog/', transform=transform_train)
        test_set = ImageFolder(path+'/'+'val/dog/', transform=transform_test)
    elif name in ['dtd']:
        transform = Compose(
            [Resize(300), RandomResizedCrop((img_size,img_size), scale=(128./300, 1.), ratio=(1., 1.)),
             RandomHorizontalFlip(0.5), RandomVerticalFlip(0.5),
             ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        dataset = ImageFolder(path, transform=transform)
    elif name in ['bitmojis']:
        transform=Compose([Resize((143, 143)), ToTensor(), CenterCrop((128, 128)), Pad((0,-7,0,7), fill=1), Resize((img_size, img_size)), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = ImageFolder(path, transform=transform)
    elif name in ['cartoon_faces']:
        transform=Compose([Resize((158, 158)), CenterCrop((128, 128)), ToTensor(), Resize((img_size, img_size)), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = ImageFolder(path, transform=transform)
    elif name in ['cartoon_bitmojis_faces']:
        # test set only from cartoon
        if img_size == 128:
            transform_cartoon = Compose([Resize((img_size, img_size)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            transform_bitmojis = Compose([Resize((118, 118)), ToTensor(), Pad((5,-1,5,11), fill=1), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        elif img_size == 64:
            transform_cartoon=Compose([Resize((img_size, img_size)), ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            transform_bitmojis=Compose([Resize((60, 60)), ToTensor(), Pad((2,-1,2,5), fill=1), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            
        dataset0 = ImageFolder('/gpfs/data/gpfs0/m.gazdieva/data/cartoonset100k_jpg/', 
                               transform=transform_cartoon)
        dataset1 = ImageFolder('/gpfs/data/gpfs0/m.gazdieva/data/bitmojis/', 
                               transform=transform_bitmojis)
        idx = list(range(len(dataset0)))
        
        test_ratio=0.1
        train_ratio=0.5
        test_size = int(len(idx) * test_ratio)
        train_size = int(len(idx) * (1 - test_ratio) * train_ratio)
        
        train_idx, test_idx = idx[:-test_size], idx[-test_size:]
        train_idx_lr, train_idx_hr = train_idx[:-train_size], train_idx[-train_size:]
#         print(train_idx_hr, test_idx)
        
        train_set, test_set = Subset(dataset0, train_idx), Subset(dataset0, test_idx)
        train_lr_set, train_hr_set = Subset(train_set, train_idx_lr), Subset(train_set, train_idx_hr)

        train_set = ConcatDataset((train_hr_set, dataset1))
    elif name in ['aim19']:
        assert img_size == 128
        scale_factor = 4
        transform_train = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_test_hr = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), CenterCrop(img_size)])
        transform_test_lr = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), CenterCrop(img_size//scale_factor)])
        train_hr_set = datasets.ImageFolder('/gpfs/gpfs0/3ddl/unpaired_sr/data/aim19/hr_patches128/', transform=transform_train)
        train_lr_set = datasets.ImageFolder('/gpfs/gpfs0/3ddl/unpaired_sr/data/aim19/lr_patches32/', transform=transform_train)
        test_hr_set = datasets.ImageFolder('/gpfs/gpfs0/3ddl/unpaired_sr/data/aim19/val_hr/', transform=transform_test_hr)
        test_lr_set = datasets.ImageFolder('/gpfs/gpfs0/3ddl/unpaired_sr/data/aim19/val_lr/', transform=transform_test_lr)
    else:
        raise Exception('Unknown dataset')
    
    if not name.startswith('cartoon_bitmojis') and not name in ['aim19', 'fruits-360', 'afhq_cat', 'afhq_dog']:
        if name in ['celeba_female', 'celeba_female_subset']:
            with open('/gpfs/gpfs0/optimaltransport-lab/datasets/celeba/list_attr_celeba.txt', 'r') as f:
                lines = f.readlines()[2:]
            idx = [i for i in list(range(len(lines))) if lines[i].replace('  ', ' ').split(' ')[21] == '-1']
        else:
            idx = list(range(len(dataset)))

        test_ratio=0.1
        test_size = int(len(idx) * test_ratio)
        if name == 'dtd':
            np.random.seed(0x000000); np.random.shuffle(idx)
            train_idx, test_idx = idx[:-test_size], idx[-test_size:]
        elif name in ['celeba_female_subset', 'aligned_anime_faces_subset']:
            train_idx, test_idx = idx[:test_size], idx[-test_size:]
        else:
            train_idx, test_idx = idx[:-test_size], idx[-test_size:]
            
        if name in ['comics_v1', 'comic_faces', 'chairs']:
            train_set, test_set = Subset(dataset, train_idx), Subset(dataset_test, test_idx)
        else:
            train_set, test_set = Subset(dataset, train_idx), Subset(dataset, test_idx)
    
    if not name in ['aim19', 'celeba_female_subset', 'aligned_anime_faces_subset']:
        train_sampler = LoaderSampler(DataLoader(train_set, shuffle=True, num_workers=8, batch_size=batch_size), device)
        test_sampler = LoaderSampler(DataLoader(test_set, shuffle=False, num_workers=8, batch_size=batch_size), device)
        return train_sampler, test_sampler
    elif name in ['celeba_female_subset', 'aligned_anime_faces_subset']:
        train_sampler = LoaderSampler(DataLoader(train_set, shuffle=False, num_workers=8, batch_size=batch_size), device)
        test_sampler = LoaderSampler(DataLoader(test_set, shuffle=False, num_workers=8, batch_size=batch_size), device)
        return train_sampler, test_sampler
    else:
        train_lr_sampler = LoaderSampler(DataLoader(train_lr_set, shuffle=True, num_workers=8, batch_size=batch_size), device)
        train_hr_sampler = LoaderSampler(DataLoader(train_hr_set, shuffle=True, num_workers=8, batch_size=batch_size), device)
        test_lr_sampler = LoaderSampler(DataLoader(test_lr_set, shuffle=False, num_workers=8, batch_size=batch_size), device)
        test_hr_sampler = LoaderSampler(DataLoader(test_hr_set, shuffle=False, num_workers=8, batch_size=batch_size), device)
        return train_lr_sampler, train_hr_sampler, test_lr_sampler, test_hr_sampler
import random

def ewma(x, span=200):
    return pd.DataFrame({'x': x}).ewm(span=span).mean().values[:, 0]

def downsample(y, scale_factor=4):
    y = F.interpolate(y, scale_factor = 1/scale_factor, mode='bilinear') # downsample
    return y

def upsample(y, scale_factor=4):
    y = F.interpolate(y, scale_factor = scale_factor, mode='bilinear') # upsample
    return y

def freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()    
    
def unfreeze(model):
    for p in model.parameters():
        p.requires_grad_(True)
    model.train(True)
    
def weights_init_D(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        
def weights_init_mlp(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf

def fig2img ( fig ):
    buf = fig2data ( fig )
    w, h, d = buf.shape
    return Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) )

def h5py_to_dataset(path, img_size=64):
    with h5py.File(path, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]

        # Get the data
        data = list(f[a_group_key])
    with torch.no_grad():
        dataset = 2 * (torch.tensor(np.array(data), dtype=torch.float32) / 255.).permute(0, 3, 1, 2) - 1
        dataset = F.interpolate(dataset, img_size, mode='bilinear')    

    return TensorDataset(dataset, torch.zeros(len(dataset)))

def calculate_cost(T, loader, cost_type='mse', verbose=False, upgrade=False):
    size = len(loader.dataset)
    
    cost = 0
    if cost_type == 'injvgg':
        from .losses import InjectiveVGGPerceptualLoss
        injvgg = InjectiveVGGPerceptualLoss().cuda()
    for step, (X, _) in tqdm(enumerate(loader)) if verbose else enumerate(loader):
        X = X.cuda()
        if upgrade == True:
            X = upsample(X)
        T_X = T(X)
        if cost_type == 'mse':
            cost += (F.mse_loss(X, T_X) * X.shape[0]).item()
        elif cost_type == 'l1':
            cost += (F.l1_loss(X, T_X) * X.shape[0]).item()
        elif cost_type == 'injvgg':
            cost += (injvgg(X, T_X) * X.shape[0]).item()
        else:
            raise Exception('Unknown COST')
        del X, T_X
    
    cost = cost / size
    gc.collect(); torch.cuda.empty_cache()
    return cost

def get_loader_stats(loader, batch_size=8, n_epochs=1, verbose=False, classes=True):
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).cuda()
    freeze(model)
    
    size = len(loader.dataset)
    pred_arr = []
    
    for epoch in range(n_epochs):
        with torch.no_grad():
            if classes:
                for step, (X, Y) in enumerate(loader) if not verbose else tqdm(enumerate(loader)):
                    for i in range(0, len(X), batch_size):
                        start, end = i, min(i + batch_size, len(X))
                        batch = ((X[start:end] + 1) / 2).type(torch.FloatTensor).cuda()
                        pred_arr.append(model(batch)[0].cpu().data.numpy().reshape(end-start, -1))
            else:
                for step, X in enumerate(loader) if not verbose else tqdm(enumerate(loader)):
                    for i in range(0, len(X), batch_size):
                        start, end = i, min(i + batch_size, len(X))
                        batch = ((X[start:end] + 1) / 2).type(torch.FloatTensor).cuda()
                        pred_arr.append(model(batch)[0].cpu().data.numpy().reshape(end-start, -1))

    pred_arr = np.vstack(pred_arr)
    mu, sigma = np.mean(pred_arr, axis=0), np.cov(pred_arr, rowvar=False)
    gc.collect(); torch.cuda.empty_cache()
    return mu, sigma

def get_loader_weighted_stats(loader, weights, batch_size=8, n_epochs=1, verbose=False, classes=True):
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).cuda()
    freeze(model)
    
    size = len(loader.dataset)
    pred_arr = []
    
    for epoch in range(n_epochs):
        with torch.no_grad():
            if classes:
                for step, (X, Y) in enumerate(loader) if not verbose else tqdm(enumerate(loader)):
                    for i in range(0, len(X), batch_size):
                        start, end = i, min(i + batch_size, len(X))
                        batch = ((X[start:end] + 1) / 2).type(torch.FloatTensor).cuda()
                        pred_arr.append(model(batch)[0].cpu().data.numpy().reshape(end-start, -1))
            else:
                for step, X in enumerate(loader) if not verbose else tqdm(enumerate(loader)):
                    for i in range(0, len(X), batch_size):
                        start, end = i, min(i + batch_size, len(X))
                        batch = ((X[start:end] + 1) / 2).type(torch.FloatTensor).cuda()
                        pred_arr.append(model(batch)[0].cpu().data.numpy().reshape(end-start, -1))
            del batch, X, Y

    pred_arr = np.vstack(pred_arr)
    mu = np.matmul(weights, pred_arr)
    sigma = np.cov(pred_arr, rowvar=False, aweights=weights)
    gc.collect(); torch.cuda.empty_cache()
    return mu, sigma

def loader2array(loader, batch_size=64, img_size=16, verbose=False):
    arr = []

    for i, (X, Y) in enumerate(loader) if not verbose else tqdm(enumerate(loader)):
        start, end = i * batch_size, i * batch_size + min(batch_size, X.shape[0])
        batch = (X + 1) / 2
        batch = batch.permute(0, 2, 3, 1).cpu().data.numpy().reshape(end-start, 3*img_size*img_size)

        arr.append(batch)
    return np.concatenate(arr, axis=0)

def get_pushed_loader_stats(T, loader, batch_size=8, n_epochs=1, verbose=False, device='cuda',
                            use_downloaded_weights=False, upgrade=False):
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx], use_downloaded_weights=use_downloaded_weights).to(device)
    freeze(model); freeze(T)
    
    size = len(loader.dataset)
    pred_arr = []
    
    for epoch in range(n_epochs):
        with torch.no_grad():
            for step, (X, _) in enumerate(loader) if not verbose else tqdm(enumerate(loader)):
                for i in range(0, len(X), batch_size):
                    start, end = i, min(i + batch_size, len(X))
                    if upgrade == True:
                        X = upsample(X)
                    batch = T(X[start:end].type(torch.FloatTensor).to(device)).add(1).mul(.5)
                    pred_arr.append(model(batch)[0].cpu().data.numpy().reshape(end-start, -1))

    pred_arr = np.vstack(pred_arr)
    mu, sigma = np.mean(pred_arr, axis=0), np.cov(pred_arr, rowvar=False)
    gc.collect(); torch.cuda.empty_cache()
    return mu, sigma

def get_W_pushed_loader_stats(T, G, loader, WC=1, batch_size=8, n_epochs=1, verbose=False, device='cuda',
                            use_downloaded_weights=False):
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx], use_downloaded_weights=use_downloaded_weights).to(device)
    freeze(model); freeze(T)
    
    size = len(loader.dataset)
    pred_arr = []
    
    for epoch in range(n_epochs):
        with torch.no_grad():
            for step, (X, _) in enumerate(loader) if not verbose else tqdm(enumerate(loader)):
                for i in range(0, len(X), batch_size):
                    start, end = i, min(i + batch_size, len(X))
                    W = 1 * np.ones((end-start, WC, 1, 1))
                    W = torch.tensor(W).type(torch.FloatTensor).to(device)
                    G_W = G(W)
                    XW = torch.cat((X[start:end].type(torch.FloatTensor).to(device), G_W), dim=1)
                    batch = T(XW).add(1).mul(.5)
                    pred_arr.append(model(batch)[0].cpu().data.numpy().reshape(end-start, -1))

    pred_arr = np.vstack(pred_arr)
    mu, sigma = np.mean(pred_arr, axis=0), np.cov(pred_arr, rowvar=False)
    gc.collect(); torch.cuda.empty_cache()
    return mu, sigma


def get_Z_pushed_loader_stats(T, loader, ZC=1, Z_STD=0.1, batch_size=8, n_epochs=1, verbose=False,
                              device='cuda',
                              use_downloaded_weights=False):
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx], use_downloaded_weights=use_downloaded_weights).to(device)
    freeze(model); freeze(T)
    
    size = len(loader.dataset)
    pred_arr = []
    
    for epoch in range(n_epochs):
        with torch.no_grad():
            for step, (X, _) in enumerate(loader) if not verbose else tqdm(enumerate(loader)):
                Z = torch.randn(len(X), ZC, 1, 1) * Z_STD
                XZ = (X, Z)
                for i in range(0, len(X), batch_size):
                    start, end = i, min(i + batch_size, len(X))
                    batch = T(
                        XZ[0][start:end].type(torch.FloatTensor).to(device),
                        XZ[1][start:end].type(torch.FloatTensor).to(device)
                    ).add(1).mul(.5)
                    pred_arr.append(model(batch)[0].cpu().data.numpy().reshape(end-start, -1))

    pred_arr = np.vstack(pred_arr)
    mu, sigma = np.mean(pred_arr, axis=0), np.cov(pred_arr, rowvar=False)
    gc.collect(); torch.cuda.empty_cache()
    return mu, sigma