import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# doing computations on the GPU is VERY important to make sure PyTorch/CUDA comparisons match, otherwise they will not
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    #test_minibatch_loading()
    #test_params_distribution()
    #test_matmul1()
    #test_matmul2()
    #test_tanh()
    #test_softmax()
    test_cross_entropy()
    
def compare(operation, t, pt, n = 100):
    maxdiff = (t - pt).abs().max().item()
    
    if torch.allclose(t, pt):
        print(f'{operation} is close, maxdiff: {maxdiff}')
    else:
        print(f'{operation} failed, maxdiff: {maxdiff}')
        
    if t.ndim == 2:
        plt.imshow((t - pt)[:n].tolist())
        plt.title(f'{operation} differences')
        plt.show()

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
    
def test_matmul1():
    Xb = torch.from_numpy(np.loadtxt('Xb.csv', delimiter=',', dtype=np.float32)).to(device)
    W1 = torch.from_numpy(np.loadtxt('W1.csv', delimiter=',', dtype=np.float32)).to(device)
    b1 = torch.from_numpy(np.loadtxt('b1.csv', delimiter=',', dtype=np.float32)).to(device).view((1,W1.shape[1]))
    lin = torch.from_numpy(np.loadtxt('lin.csv', delimiter=',', dtype=np.float32)).to(device)
    lin_t = Xb @ W1 + b1
    compare('matmul1', lin, lin_t)
    
def test_matmul2():
    act = torch.from_numpy(np.loadtxt('act.csv', delimiter=',', dtype=np.float32)).to(device)
    W2 = torch.from_numpy(np.loadtxt('W2.csv', delimiter=',', dtype=np.float32)).to(device)
    b2 = torch.from_numpy(np.loadtxt('b2.csv', delimiter=',', dtype=np.float32)).to(device).view((1,W2.shape[1]))
    logits = torch.from_numpy(np.loadtxt('logits.csv', delimiter=',', dtype=np.float32)).to(device)
    logits_t = act @ W2 + b2
    compare('matmul2', logits, logits_t)
    
def test_tanh():
    lin = torch.from_numpy(np.loadtxt('lin.csv', delimiter=',', dtype=np.float32)).to(device)
    act = torch.from_numpy(np.loadtxt('act.csv', delimiter=',', dtype=np.float32)).to(device)
    act_t = torch.tanh(lin)
    compare('tanh', act, act_t)

def test_softmax():
    logits = torch.from_numpy(np.loadtxt('logits.csv', delimiter=',', dtype=np.float32)).to(device)
    probs = torch.from_numpy(np.loadtxt('probs.csv', delimiter=',', dtype=np.float32)).to(device)
    probs_t = F.softmax(logits, dim=1)
    compare('softmax', probs, probs_t)
    
def test_cross_entropy():
    yb = torch.from_numpy(np.loadtxt('yb.csv', delimiter=',', dtype=np.int64)).to(device)
    probs = torch.from_numpy(np.loadtxt('probs.csv', delimiter=',', dtype=np.float32)).to(device)
    losses = torch.from_numpy(np.loadtxt('losses.csv', delimiter=',', dtype=np.float32)).to(device)
    losses_t = -probs[range(yb.shape[0]), yb].log()
    np.savetxt('losses_t.csv', losses_t.view(-1).cpu().numpy(), delimiter=',')
    compare('cross entropy', losses, losses_t)
    
if __name__ == '__main__':
    main()