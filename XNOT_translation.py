# Importing necessary libraries
import os, sys
import argparse
import wandb
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import ot
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import gc
from sklearn.decomposition import PCA
from copy import deepcopy
import json
from tqdm import tqdm_notebook as tqdm
from IPython.display import clear_output
from src.resnet2 import ResNet_D
from src.unet import UNet
from src.tools import unfreeze, freeze, downsample, upsample
from src.tools import weights_init_D
from src.tools import load_dataset, get_pushed_loader_stats, get_loader_weighted_stats, fig2img
from src.tools import get_loader_stats, loader2array, calculate_cost
from src.fid_score import calculate_frechet_distance
from src.plotters import plot_random_images, plot_images, plot_generated_pca
from src.losses import VGGPerceptualLoss, InjectiveVGGPerceptualLoss

# Defining main configuration
def main():
    # Preparation
    config = dict(
    DATASET1=DATASET1,
    DATASET2=DATASET2, 
    T_ITERS=T_ITERS,
    D_LR=D_LR, T_LR=T_LR,
    BATCH_SIZE=BATCH_SIZE
)
    
    torch.cuda.set_device(f'cuda:{DEVICE_IDS[0]}')
    torch.manual_seed(SEED); np.random.seed(SEED)

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    
    # Prepare samples (X, Y)
    X_sampler, X_test_sampler = load_dataset(DATASET1, DATASET1_PATH, img_size=IMG_SIZE, batch_size=BATCH_SIZE)
    Y_sampler, Y_test_sampler = load_dataset(DATASET2, DATASET2_PATH, img_size=IMG_SIZE, batch_size=BATCH_SIZE)

    torch.cuda.empty_cache(); gc.collect()
    
    # Loading data stats for testing
    try:
        filename = 'stats/{}_{}_test.json'.format(DATASET2, IMG_SIZE)
        with open(filename, 'r') as fp:
            data_stats = json.load(fp)
    except:
        mu, sigma = get_loader_stats(Y_test_sampler.loader, n_epochs=1, verbose=True)
        data_stats = {'mu' : mu.tolist(), 'sigma' : sigma.tolist()}
        filename = 'stats/{}_{}_test.json'.format(DATASET2, IMG_SIZE)
        with open(filename, 'w') as fp:
            json.dump(data_stats, fp)
    mu_data, sigma_data = data_stats['mu'], data_stats['sigma']
    del data_stats
    
    # Initializing networks
    D = ResNet_D(IMG_SIZE, nc=3).cuda()
    D.apply(weights_init_D)
    T = UNet(3, 3, base_factor=BASE_FACTOR).cuda()
    
    if len(DEVICE_IDS) > 1:
        T = nn.DataParallel(T, device_ids=DEVICE_IDS)
        D = nn.DataParallel(D, device_ids=DEVICE_IDS)
        
    if CONTINUE > 0:
        T.load_state_dict(torch.load(OUTPUT_PATH+'0_best.pt'))
        D.load_state_dict(torch.load(OUTPUT_PATH+'D_0_best.pt'))
    
    torch.manual_seed(SEED); np.random.seed(SEED)
    X_fixed = X_sampler.sample(10)
    Y_fixed = Y_sampler.sample(10)
    X_test_fixed = X_test_sampler.sample(10)
    Y_test_fixed = Y_test_sampler.sample(10)
    
    # Run Training
    wandb.init(name=EXP_NAME, project='project_name', entity='entity_name', config=config)
    
    T_opt = torch.optim.Adam(T.parameters(), lr=T_LR, weight_decay=1e-10)
    D_opt = torch.optim.Adam(D.parameters(), lr=D_LR, weight_decay=1e-10)
    T_scheduler = torch.optim.lr_scheduler.MultiStepLR(T_opt, milestones=[10000+5000*W1, 20000+5000*W1, 40000+5000*W1, 70000+5000*W1], gamma=0.5)
    D_scheduler = torch.optim.lr_scheduler.MultiStepLR(D_opt, milestones=[10000+5000*W1, 20000+5000*W1, 40000+5000*W1, 70000+5000*W1], gamma=0.5)

    if CONTINUE > 0:
        T_opt.load_state_dict(torch.load(OUTPUT_PATH+f'T_opt_{SEED}_best.pt'))
        D_opt.load_state_dict(torch.load(OUTPUT_PATH+f'D_opt_{SEED}_best.pt'))
        T_scheduler.load_state_dict(torch.load(OUTPUT_PATH+f'T_scheduler_{SEED}_best.pt'))
        D_scheduler.load_state_dict(torch.load(OUTPUT_PATH+f'D_scheduler_{SEED}_best.pt'))
        
    if COST == 'injvgg':
        injvgg_loss = InjectiveVGGPerceptualLoss().cuda()
    
    best_fid = np.inf
    
    for step in range(MAX_STEPS):
        try:
            W = min(W1, W0 + (W1-W0) * step / W_ITERS)
        except:
            W = W1
        # T optimization
        unfreeze(T); freeze(D)
        for t_iter in range(T_ITERS): 
            T_opt.zero_grad()
            X = X_sampler.sample(BATCH_SIZE)
            T_X = T(X)
            if COST == 'mse':
                T_loss = F.mse_loss(X, T_X).mean() - D(T_X).mean()
            elif COST == 'injvgg':
                T_loss = injvgg_loss(X, T_X).mean() - D(T_X).mean()
            else:
                raise Exception('Unknown COST')
            T_loss.backward(); T_opt.step()
        T_scheduler.step()
        del T_loss, T_X, X; gc.collect(); torch.cuda.empty_cache()
        
        # D optimization
        freeze(T); unfreeze(D)
        X = X_sampler.sample(BATCH_SIZE)
        with torch.no_grad():
            T_X = T(X)
        Y = Y_sampler.sample(BATCH_SIZE)
        D_opt.zero_grad()
        D_loss = D(T_X).mean() - W * D(Y).mean()
        D_loss.backward(); D_opt.step();
        D_scheduler.step()
        wandb.log({f'D_loss' : D_loss.item()}, step=step)
        wandb.log({f'Cost' : F.mse_loss(X, T_X).mean().item()}, step=step)
        del D_loss, Y, X, T_X; gc.collect(); torch.cuda.empty_cache()
        if step % PLOT_INTERVAL == 0:
            print('Plotting')
            clear_output(wait=True)
            fig, axes = plot_images(X_fixed, Y_fixed, T)
            wandb.log({'Fixed Images' : [wandb.Image(fig2img(fig))]}, step=step) 
            plt.close(fig) 
            
            X_random = X_sampler.sample(10)
            Y_random = Y_sampler.sample(10)
            fig, axes = plot_images(X_random, Y_random, T)
            wandb.log({'Random Images' : [wandb.Image(fig2img(fig))]}, step=step) 
            plt.close(fig) 
            fig, axes = plot_images(X_test_fixed, Y_test_fixed, T)
            wandb.log({'Fixed Test Images' : [wandb.Image(fig2img(fig))]}, step=step) 
            plt.close(fig) 
            
            X_test_random = X_test_sampler.sample(10)
            Y_test_random = Y_test_sampler.sample(10)
            fig, axes = plot_images(X_test_random, Y_test_random, T)
            wandb.log({'Random Test Images' : [wandb.Image(fig2img(fig))]}, step=step) 
            plt.close(fig) 
        if step % CPKT_INTERVAL == CPKT_INTERVAL - 1:
            freeze(T); freeze(D);
            torch.save(T.state_dict(), os.path.join(OUTPUT_PATH, f'{SEED}_{step}.pt'))
        if step % FID_INTERVAL == FID_INTERVAL - 1:
            freeze(T); freeze(D);
            mu, sigma = get_pushed_loader_stats(T, X_test_sampler.loader, n_epochs=1, device='cuda', verbose=True, upgrade=False)
            fid = calculate_frechet_distance(mu_data, sigma_data, mu, sigma)
            wandb.log({f'Test_FID' : fid.item()}, step=step) 
            if fid < best_fid:
                best_fid = fid
                torch.save(T.state_dict(), os.path.join(OUTPUT_PATH, f'{SEED}_best.pt'))
                torch.save(D.state_dict(), os.path.join(OUTPUT_PATH, f'D_{SEED}_best.pt'))
                torch.save(D_opt.state_dict(), os.path.join(OUTPUT_PATH, f'D_opt_{SEED}_best.pt'))
                torch.save(T_opt.state_dict(), os.path.join(OUTPUT_PATH, f'T_opt_{SEED}_best.pt'))
                torch.save(D_scheduler.state_dict(), os.path.join(OUTPUT_PATH, f'D_scheduler_{SEED}_best.pt'))
                torch.save(T_scheduler.state_dict(), os.path.join(OUTPUT_PATH, f'T_scheduler_{SEED}_best.pt'))
            
            try:
                mean_cost = calculate_cost(T, X_test_sampler.loader, verbose=False, upgrade=False if not DATASET1=='aim19' else True)
                wandb.log({f'Test_Cost' : mean_cost}, step=step)
            except Exception as e:
                print(e)
            
        freeze(T); freeze(D);
        torch.save(T.state_dict(), os.path.join(OUTPUT_PATH, f'{SEED}_last.pt'))
        torch.save(D.state_dict(), os.path.join(OUTPUT_PATH, f'D_{SEED}_last.pt'))
        torch.save(D_opt.state_dict(), os.path.join(OUTPUT_PATH, f'D_opt_{SEED}_last.pt'))
        torch.save(T_opt.state_dict(), os.path.join(OUTPUT_PATH, f'T_opt_{SEED}_last.pt'))
        torch.save(D_scheduler.state_dict(), os.path.join(OUTPUT_PATH, f'D_scheduler_{SEED}_last.pt'))
        torch.save(T_scheduler.state_dict(), os.path.join(OUTPUT_PATH, f'T_scheduler_{SEED}_last.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prefix_chars='--')
    parser.add_argument('--W', type=int, default=1,
                         help='weight')
    ARGS = parser.parse_args()
    
    DATASET1, DATASET1_PATH = 'celeba_female', '../../data/celeba'
    DATASET2, DATASET2_PATH = 'aligned_anime_faces',  '../../data/aligned_anime_faces'

    T_ITERS = 10
    D_LR, T_LR = 1e-4, 1e-4
    IMG_SIZE = 64
    W0, W1 = 1, ARGS.W
    CONTINUE = 1
    if CONTINUE > 0:
        W_ITERS = 0
    else:
        W_ITERS = 20000
    
    BATCH_SIZE = 64*(W1//8)
    ARCH = 'UNET'
    BASE_FACTOR = 48 # Parameter for UNet/CUNet

    PLOT_INTERVAL = 500
    COST = 'mse' # Mean Squared Error
    CPKT_INTERVAL = 500
    FID_INTERVAL = 1000
    PCA_INTERVAL = 2000
    MAX_STEPS = 200001
    SEED = 0x000000

    EXP_NAME = f'{DATASET1}_{DATASET2}_T{T_ITERS}_{COST}_{IMG_SIZE}_{BATCH_SIZE}_{BASE_FACTOR}_W{W1}'
    OUTPUT_PATH = 'checkpoints/{}_{}_{}_{}_{}_{}_{}/'.format(COST, DATASET1, DATASET2, IMG_SIZE, BATCH_SIZE, BASE_FACTOR, W1)
    
    assert torch.cuda.is_available()    
    DEVICE_IDS = [i for i in range(torch.cuda.device_count())]
    
    main()