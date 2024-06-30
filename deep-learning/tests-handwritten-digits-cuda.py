import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def main():
    test_minibatch_loading()
    test_params_distribution()
    test_matmul()
    
def test_minibatch_loading():
    Xb = np.loadtxt('Xb.csv', delimiter=',', dtype=np.float32)
    yb = np.loadtxt('yb.csv', delimiter=',', dtype=np.float32)
    
    Xrows, Xcols = Xb.shape
    yrows, = yb.shape
    
    print(f'Size of Xb is {Xrows} rows and {Xcols} columns.')
    print(f'Size of yb is {yrows} rows.')
    
    img_sz = 28 # 28 x 28 image
    N = 4 # checking 4 x 4 = 16 images
    
    fig, axs = plt.subplots(nrows=N, ncols=N)
    
    for j in range(N):
        for i in range(N):
            idx = j * N + i
            image = Xb[idx].reshape(img_sz,img_sz)
            n = int(yb[j * N + i])
            axs[j,i].imshow(image, cmap='gray')
            axs[j,i].set_title(f'Label: {n}', fontsize='medium')
    
    for ax in axs.flat:
        ax.xaxis.set_ticks([]); ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticks([]); ax.yaxis.set_ticklabels([])
    
    fig.tight_layout()
    plt.show()
    
def test_params_distribution():
    W1 = np.loadtxt('W1.csv', delimiter=',', dtype=np.float32)
    b1 = np.loadtxt('b1.csv', delimiter=',', dtype=np.float32)
    
    fig, axs = plt.subplots(nrows=2)
    axs[0].hist(W1.flat, bins=200)
    axs[1].hist(b1.flat, bins=200)
    axs[0].set_title(f'Weights')
    axs[1].set_title(f'Biases')
    fig.tight_layout()
    plt.show()
    
def test_matmul():
    Xb  = torch.from_numpy(np.loadtxt('Xb.csv',  delimiter=',', dtype=np.float32))
    W1  = torch.from_numpy(np.loadtxt('W1.csv',  delimiter=',', dtype=np.float32))
    b1  = torch.from_numpy(np.loadtxt('b1.csv',  delimiter=',', dtype=np.float32)).view((1,100))
    lin = torch.from_numpy(np.loadtxt('lin.csv', delimiter=',', dtype=np.float32))
    
    lin_torch = Xb @ W1 + b1
    
    maxdiff = (lin - lin_torch).abs().max().item()
    
    if torch.allclose(lin, lin_torch):
        print(f'Matmul is close, maxdiff: {maxdiff}')
    else:
        print(f'Matmul failed, maxdiff: {maxdiff}')

if __name__ == '__main__':
    main()