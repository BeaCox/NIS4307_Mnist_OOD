import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from model_vae import VAE_CNN
import config

# 超参数定义
cfg = config.load()
num_epochs, batch_size, learning_rate, thresholds_percent, KLD_weight = cfg[
    'n_epochs'], cfg['batch_size_train'], cfg['learning_rate'], cfg['thresholds_percent'], cfg['KLD_weight']

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# 数据增强
transform = transforms.Compose([
    transforms.RandomRotation(25),
    transforms.ToTensor(),
])
transform2 = transforms.Compose([
    transforms.ToTensor(),
])

# 加载全部数据
mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_data2 = datasets.MNIST(root='./data', train=False, download=True, transform=transform2)
# 合并数据集
mnist_data = torch.utils.data.ConcatDataset([mnist_data, mnist_data2])
# 获取数据集的大小
total_size = len(mnist_data)
# 计算训练数据集的大小（80%）
train_size = int(total_size * 0.8)
# 计算测试数据集的大小（20%）
test_size = total_size - train_size
# 分割数据集
train_data, test_data = torch.utils.data.random_split(mnist_data, [train_size, test_size])

# 对于每个数字，创建一个数据加载器
digit_train_loaders = [DataLoader(Subset(train_data, 
                       [idx for idx, (_, target) in enumerate(train_data) if target == i]),
                       batch_size=batch_size, shuffle=True) for i in range(10)]
    
    # 用于存储每个数字对应的模型
vaes = [VAE_CNN().to(device) for _ in range(10)]
optimizers = [optim.Adamax(vae.parameters(), lr=learning_rate) for vae in vaes]

def loss_function(recon_x, x, mu, logvar): # 重构损失 + KL散度
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 1, 28, 28), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (1-KLD_weight)*BCE + KLD_weight*KLD

def train():
    count = 0
    LOSS = [[] for j in range(10)]
    for epoch in range(num_epochs):
        for i in range(10):
            for data in digit_train_loaders[i]:
                img, _ = data
                img = img.to(device)
                recon_batch, mu, logvar = vaes[i](img)
                loss = loss_function(recon_batch, img, mu, logvar)
                if count%50==0:
                    print('number:{}:epoch [{}/{}], loss:{:.4f}'.format(i,epoch + 1, num_epochs, loss.item()))
                optimizers[i].zero_grad()
                loss.backward()
                optimizers[i].step()
                count = count+1
                LOSS[i].append(loss.item())

    for num in range (10):
        plt.subplot(2, 5, num+1)
        x = np.arange(len(LOSS[num]))
        plt.plot(x, LOSS[num])
        plt.title("train loss of %d" %num)
    plt.suptitle("train loss")
    plt.show()

def compute_reconstruction_error(data, model): # 计算重构误差
    with torch.no_grad():
        recon_data, _, _ = model(data)
        recon_error = ((recon_data - data)**2).sum(dim=(1,2,3))
        return recon_error.cpu().numpy()

train()

model_dict = {f'VAE2_digit_{i}': vae.state_dict() for i, vae in enumerate(vaes)}
torch.save(model_dict, 'VAE2s.pth')

# 计算阈值，设置为训练集上的thresholds_percent重构误差百分位数
thresholds = [np.percentile([compute_reconstruction_error(data, vaes[i]).max() 
                             for data, _ in digit_train_loaders[i]], thresholds_percent) for i in range(10)]
np.save('./thresholds_2.npy', thresholds)