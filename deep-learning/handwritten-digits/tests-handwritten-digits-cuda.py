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
    test_matmul1()
    test_tanh()
    test_matmul2()
    test_softmax()
    test_cross_entropy()
    test_logits_backward()
    test_transpose()
    test_act_backward()
    test_W2_backward()
    test_b2_backward()
    test_tanh_backward()
    test_W1_backward()
    test_b1_backward()
    
def compare(operation, t, pt, n = 100):
    maxerr = (t - pt).abs().max().item()
    
    if torch.allclose(t, pt):
        print(f'{operation:20} success, max error: {maxerr:.2e}')
    else:
        print(f'{operation:20} failure, max error: {maxerr:.2e}')
        
    if False:#t.ndim == 2:
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
    compare('Matmul1', lin, lin_t)
    
def test_tanh():
    lin = torch.from_numpy(np.loadtxt('lin.csv', delimiter=',', dtype=np.float32)).to(device)
    act = torch.from_numpy(np.loadtxt('act.csv', delimiter=',', dtype=np.float32)).to(device)
    act_t = torch.tanh(lin)
    compare('Tanh', act, act_t)

def test_matmul2():
    act = torch.from_numpy(np.loadtxt('act.csv', delimiter=',', dtype=np.float32)).to(device)
    W2 = torch.from_numpy(np.loadtxt('W2.csv', delimiter=',', dtype=np.float32)).to(device)
    b2 = torch.from_numpy(np.loadtxt('b2.csv', delimiter=',', dtype=np.float32)).to(device).view((1,W2.shape[1]))
    logits = torch.from_numpy(np.loadtxt('logits.csv', delimiter=',', dtype=np.float32)).to(device)
    logits_t = act @ W2 + b2
    compare('Matmul2', logits, logits_t)
    
def test_softmax():
    logits = torch.from_numpy(np.loadtxt('logits.csv', delimiter=',', dtype=np.float32)).to(device)
    probs = torch.from_numpy(np.loadtxt('probs.csv', delimiter=',', dtype=np.float32)).to(device)
    probs_t = F.softmax(logits, dim=1)
    compare('Softmax', probs, probs_t)
    
def test_cross_entropy():
    yb = torch.from_numpy(np.loadtxt('yb.csv', delimiter=',', dtype=np.int64)).to(device)
    probs = torch.from_numpy(np.loadtxt('probs.csv', delimiter=',', dtype=np.float32)).to(device)
    losses = torch.from_numpy(np.loadtxt('losses.csv', delimiter=',', dtype=np.float32)).to(device)
    losses_t = -probs[range(yb.shape[0]), yb].log()
    compare('Cross entropy', losses, losses_t)
    
def test_logits_backward():
    yb = torch.from_numpy(np.loadtxt('yb.csv', delimiter=',', dtype=np.int64)).to(device)
    logits = torch.from_numpy(np.loadtxt('logits.csv', delimiter=',', dtype=np.float32)).to(device)
    dlogits = torch.from_numpy(np.loadtxt('dlogits.csv', delimiter=',', dtype=np.float32)).to(device)
    B = yb.shape[0]
    dlogits_t = F.softmax(logits, dim=1)
    dlogits_t[range(B), yb] -= 1
    dlogits_t /= B
    compare('Logits backward', dlogits, dlogits_t)
    
def test_transpose():
    W2 = torch.from_numpy(np.loadtxt('W2.csv', delimiter=',', dtype=np.float32)).to(device)
    W2T = torch.from_numpy(np.loadtxt('W2T.csv', delimiter=',', dtype=np.float32)).to(device)
    compare('Transpose', W2T, W2.T)

def test_act_backward():
    dlogits = torch.from_numpy(np.loadtxt('dlogits.csv', delimiter=',', dtype=np.float32)).to(device)
    W2 = torch.from_numpy(np.loadtxt('W2.csv', delimiter=',', dtype=np.float32)).to(device)
    dact = torch.from_numpy(np.loadtxt('dact.csv', delimiter=',', dtype=np.float32)).to(device)
    dact_t = dlogits @ W2.T
    compare('Act backward', dact, dact_t)

def test_W2_backward():
    act = torch.from_numpy(np.loadtxt('act.csv', delimiter=',', dtype=np.float32)).to(device)
    dlogits = torch.from_numpy(np.loadtxt('dlogits.csv', delimiter=',', dtype=np.float32)).to(device)
    dW2 = torch.from_numpy(np.loadtxt('dW2.csv', delimiter=',', dtype=np.float32)).to(device)
    dW2_t = act.T @ dlogits
    compare('W2 backward', dW2, dW2_t)

def test_b2_backward():
    dlogits = torch.from_numpy(np.loadtxt('dlogits.csv', delimiter=',', dtype=np.float32)).to(device)
    db2 = torch.from_numpy(np.loadtxt('db2.csv', delimiter=',', dtype=np.float32)).to(device).view((1,dlogits.shape[1]))
    db2_t = dlogits.sum(dim=0, keepdims=True)
    compare('b2 backward', db2, db2_t)
    
def test_tanh_backward():
    dact = torch.from_numpy(np.loadtxt('dact.csv', delimiter=',', dtype=np.float32)).to(device)
    act = torch.from_numpy(np.loadtxt('act.csv', delimiter=',', dtype=np.float32)).to(device)
    dlin = torch.from_numpy(np.loadtxt('dlin.csv', delimiter=',', dtype=np.float32)).to(device)
    dlin_t = dact * (1 - act ** 2)
    compare('Tanh backward', dlin, dlin_t)

def test_W1_backward():
    Xb = torch.from_numpy(np.loadtxt('Xb.csv', delimiter=',', dtype=np.float32)).to(device)
    dlin = torch.from_numpy(np.loadtxt('dlin.csv', delimiter=',', dtype=np.float32)).to(device)
    dW1 = torch.from_numpy(np.loadtxt('dW1.csv', delimiter=',', dtype=np.float32)).to(device)
    dW1_t = Xb.T @ dlin
    compare('W1 backward', dW1, dW1_t)

def test_b1_backward():
    dlin = torch.from_numpy(np.loadtxt('dlin.csv', delimiter=',', dtype=np.float32)).to(device)
    db1 = torch.from_numpy(np.loadtxt('db1.csv', delimiter=',', dtype=np.float32)).to(device).view((1,dlin.shape[1]))
    db1_t = dlin.sum(dim=0, keepdims=True)
    compare('b1 backward', db1, db1_t)
    
if __name__ == '__main__':
    main()