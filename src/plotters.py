import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from .tools import ewma, freeze
from sklearn.decomposition import PCA

import torch
import gc

def plot_images(X, Y, T):
    freeze(T);
    with torch.no_grad():
        T_X = T(X)
        imgs = torch.cat([X, T_X, Y]).to('cpu').permute(0,2,3,1).mul(0.5).add(0.5).numpy().clip(0,1)

    fig, axes = plt.subplots(3, 10, figsize=(15, 4.5), dpi=150)
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(imgs[i])
        ax.get_xaxis().set_visible(False)
        ax.set_yticks([])
        
    axes[0, 0].set_ylabel('X', fontsize=24)
    axes[1, 0].set_ylabel('T(X)', fontsize=24)
    axes[2, 0].set_ylabel('Y', fontsize=24)
    
    fig.tight_layout(pad=0.001)
    torch.cuda.empty_cache(); gc.collect()
    return fig, axes

def plot_random_images(X_sampler, Y_sampler, T):
    X = X_sampler.sample(10)
    Y = Y_sampler.sample(10)
    return plot_images(X, Y, T)

def plot_W_images(X, Y, T, G, WC=1):
    freeze(T); freeze(G);
    W1 = 1 * np.ones((len(X), WC, 1, 1))
    W1 = torch.tensor(W1).float().cuda()
    G_W1 = G(W1)
    W2 = 2 * np.ones((len(X), WC, 1, 1))
    W2 = torch.tensor(W2).float().cuda()
    G_W2 = G(W2)
    W4 = 4 * np.ones((len(X), WC, 1, 1))
    W4 = torch.tensor(W4).float().cuda()
    G_W4 = G(W4)
    W8 = 8 * np.ones((len(X), WC, 1, 1))
    W8 = torch.tensor(W8).float().cuda()
    G_W8 = G(W8)
    
    with torch.no_grad():
        T_X_1 = T(torch.cat((X, G_W1), dim=1))
        T_X_2 = T(torch.cat((X, G_W2), dim=1))
        T_X_4 = T(torch.cat((X, G_W4), dim=1))
        T_X_8 = T(torch.cat((X, G_W8), dim=1))
        imgs = torch.cat([X, T_X_1, T_X_2, T_X_4, T_X_8, Y]).to('cpu').permute(0,2,3,1).mul(0.5).add(0.5).numpy().clip(0,1)

    fig, axes = plt.subplots(6, 10, figsize=(15, 9.5), dpi=150)
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(imgs[i])
        ax.get_xaxis().set_visible(False)
        ax.set_yticks([])
        
    axes[0, 0].set_ylabel('X', fontsize=24)
    axes[1, 0].set_ylabel('T(X, 1)', fontsize=24)
    axes[2, 0].set_ylabel('T(X, 2)', fontsize=24)
    axes[3, 0].set_ylabel('T(X, 4)', fontsize=24)
    axes[4, 0].set_ylabel('T(X, 8)', fontsize=24)
    axes[5, 0].set_ylabel('Y', fontsize=24)
    
    fig.tight_layout(pad=0.001)
    torch.cuda.empty_cache(); gc.collect()
    return fig, axes

def plot_generated_pca(X_sampler, Y_sampler, T, emb_Y, IMG_SIZE):
    freeze(T)
    
    X_full = torch.zeros((0, IMG_SIZE*IMG_SIZE*3))
    Y_full = torch.zeros((0, IMG_SIZE*IMG_SIZE*3))
    T_X_full = torch.zeros((0, IMG_SIZE*IMG_SIZE*3))
    for i in range(100):
        X = X_sampler.sample().cuda()
        Y = Y_sampler.sample().cuda()

        T_X = T(X)
        X = X.reshape(X.shape[0], X.shape[1]*X.shape[2]*X.shape[3])
        Y = Y.reshape(Y.shape[0], Y.shape[1]*Y.shape[2]*Y.shape[3])
        T_X = T_X.reshape(T_X.shape[0], T_X.shape[1]*T_X.shape[2]*T_X.shape[3])

        X_full = torch.cat((X_full, X.cpu().detach()))
        Y_full = torch.cat((Y_full, Y.cpu().detach()))
        T_X_full = torch.cat((T_X_full, T_X.cpu().detach()))

    X = emb_Y.transform(X_full)
    Y = emb_Y.transform(Y_full)
    T_X = emb_Y.transform(T_X_full)

    fig, axes = plt.subplots(1, 3, figsize=(4*3, 4), dpi=100, sharex=True, sharey=True)
    axes[0].scatter(X[:, 0], X[:, 1], c='Orange', edgecolors='black')
    axes[1].scatter(Y[:, 0], Y[:, 1], c='Yellow', edgecolors='black')
    axes[2].scatter(T_X[:, 0], T_X[:, 1], c='LimeGreen', edgecolors='black')

    axes[0].set_title('X', fontsize=24)
    axes[1].set_title('Y', fontsize=24)
    axes[2].set_title('T(X)', fontsize=24)
    
    fig.tight_layout()
    return fig, axes

def plot_W_generated_pca(X_sampler, Y_sampler, T, G, WC, emb_Y, IMG_SIZE, batch_size=5):
    freeze(T); freeze(G);
    
    X_full = torch.zeros((0, IMG_SIZE*IMG_SIZE*3))
    Y_full = torch.zeros((0, IMG_SIZE*IMG_SIZE*3))
    T_X_W1_full = torch.zeros((0, IMG_SIZE*IMG_SIZE*3))
    T_X_W2_full = torch.zeros((0, IMG_SIZE*IMG_SIZE*3))
    T_X_W4_full = torch.zeros((0, IMG_SIZE*IMG_SIZE*3))
    T_X_W8_full = torch.zeros((0, IMG_SIZE*IMG_SIZE*3))
    W1 = 1 * np.ones((batch_size, WC, 1, 1))
    W1 = torch.tensor(W1).float().cuda()
    G_W1 = G(W1)
    W2 = 2 * np.ones((batch_size, WC, 1, 1))
    W2 = torch.tensor(W2).float().cuda()
    G_W2 = G(W2)
    W4 = 4 * np.ones((batch_size, WC, 1, 1))
    W4 = torch.tensor(W4).float().cuda()
    G_W4 = G(W4)
    W8 = 8 * np.ones((batch_size, WC, 1, 1))
    W8 = torch.tensor(W8).float().cuda()
    G_W8 = G(W8)

    for i in range(100):
        X = X_sampler.sample(batch_size).cuda()
        Y = Y_sampler.sample(batch_size).cuda()

        T_X_W1 = T(torch.cat((X, G_W1), dim=1))
        T_X_W2 = T(torch.cat((X, G_W2), dim=1))
        T_X_W4 = T(torch.cat((X, G_W4), dim=1))
        T_X_W8 = T(torch.cat((X, G_W8), dim=1))
        X = X.reshape(X.shape[0], X.shape[1]*X.shape[2]*X.shape[3])
        Y = Y.reshape(Y.shape[0], Y.shape[1]*Y.shape[2]*Y.shape[3])
        T_X_W1 = T_X_W1.reshape(T_X_W1.shape[0], T_X_W1.shape[1]*T_X_W1.shape[2]*T_X_W1.shape[3])
        T_X_W2 = T_X_W2.reshape(T_X_W2.shape[0], T_X_W2.shape[1]*T_X_W2.shape[2]*T_X_W2.shape[3])
        T_X_W4 = T_X_W4.reshape(T_X_W4.shape[0], T_X_W4.shape[1]*T_X_W4.shape[2]*T_X_W4.shape[3])
        T_X_W8 = T_X_W8.reshape(T_X_W8.shape[0], T_X_W8.shape[1]*T_X_W8.shape[2]*T_X_W8.shape[3])

        X_full = torch.cat((X_full, X.cpu().detach()))
        Y_full = torch.cat((Y_full, Y.cpu().detach()))
        T_X_W1_full = torch.cat((T_X_W1_full, T_X_W1.cpu().detach()))
        T_X_W2_full = torch.cat((T_X_W2_full, T_X_W2.cpu().detach()))
        T_X_W4_full = torch.cat((T_X_W4_full, T_X_W4.cpu().detach()))
        T_X_W8_full = torch.cat((T_X_W8_full, T_X_W8.cpu().detach()))

    X = emb_Y.transform(X_full)
    Y = emb_Y.transform(Y_full)
    T_X_W1 = emb_Y.transform(T_X_W1_full)
    T_X_W2 = emb_Y.transform(T_X_W2_full)
    T_X_W4 = emb_Y.transform(T_X_W4_full)
    T_X_W8 = emb_Y.transform(T_X_W8_full)

    fig, axes = plt.subplots(1, 6, figsize=(4*6, 4), dpi=100, sharex=True, sharey=True)
    axes[0].scatter(X[:, 0], X[:, 1], c='Orange', edgecolors='black')
    axes[1].scatter(Y[:, 0], Y[:, 1], c='Yellow', edgecolors='black')
    axes[2].scatter(T_X_W1[:, 0], T_X_W1[:, 1], c='LimeGreen', edgecolors='black')
    axes[3].scatter(T_X_W2[:, 0], T_X_W2[:, 1], c='LimeGreen', edgecolors='black')
    axes[4].scatter(T_X_W4[:, 0], T_X_W4[:, 1], c='LimeGreen', edgecolors='black')
    axes[5].scatter(T_X_W8[:, 0], T_X_W8[:, 1], c='LimeGreen', edgecolors='black')

    axes[0].set_title('X', fontsize=24)
    axes[1].set_title('Y', fontsize=24)
    axes[2].set_title('T(X, 1)', fontsize=24)
    axes[3].set_title('T(X, 2)', fontsize=24)
    axes[4].set_title('T(X, 4)', fontsize=24)
    axes[5].set_title('T(X, 8)', fontsize=24)
    
    fig.tight_layout()
    return fig, axes