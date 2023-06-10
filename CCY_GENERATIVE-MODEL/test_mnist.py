import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from model_vae import VAE_CNN
import config


# 超参数定义
cfg = config.load()
batch_size = cfg['batch_size_test']

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# 创建VAE模型列表
num_vaes = 10
vaes = [VAE_CNN() for i in range(num_vaes)]
# 加载模型结构
model_dict = torch.load('./VAEs.pth')
for i in range(num_vaes):
    vaes[i].load_state_dict(model_dict[f'VAE_digit_{i}'])

# 数据增强
transform = transforms.Compose([
    transforms.RandomRotation(25),
    transforms.ToTensor(),
])
transform2 = transforms.Compose([
    transforms.ToTensor(),
])

# 加载全部数据
# mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# mnist_data2 = datasets.MNIST(root='./data', train=False, download=True, transform=transform2)
# 合并数据集
# mnist_data = torch.utils.data.ConcatDataset([mnist_data, mnist_data2])
mnist_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
# # 获取数据集的大小
# total_size = len(mnist_data)
# # 计算训练数据集的大小（80%）
# train_size = int(total_size * 0.8)
# # 计算测试数据集的大小（20%）
# test_size = total_size - train_size
# # 分割数据集
# train_data, test_data = torch.utils.data.random_split(mnist_data, [train_size, test_size])

# 对于每个数字，创建一个测试数据加载器
digit_test_loaders = [DataLoader(Subset(mnist_data, 
                       [idx for idx, (_, target) in enumerate(mnist_data) if target == i]),
                       batch_size=batch_size, shuffle=True) for i in range(10)]


def compute_reconstruction_error(data, model): # 计算重构误差
    with torch.no_grad():
        recon_data, _, _ = model(data)
        recon_error = ((recon_data - data)**2).sum(dim=(1,2,3))
        return recon_error.cpu().numpy()


def test():
    y_true = []
    test_errors = []
    with torch.no_grad():
        for i in range(10):
            for data, targets in digit_test_loaders[i]:
                img = data.to(device)
                errors = np.stack([compute_reconstruction_error(img, vae) for vae in vaes], axis=-1)
                test_errors.extend(errors)
                y_true.extend(targets.numpy())
    test_errors = np.array(test_errors)

    # 对于每个测试样本，将其分类为重构误差最小的类别
    y_pred = np.argmin(test_errors, axis=1)
    thresholds = np.load('./thresholds.npy')
    # y_pred = [pred if error[pred] <= thresholds[pred] else 10 for pred, error in zip(y_pred, test_errors)]

    # 计算accuracy，precision，recall和F1-score
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"In MNIST:Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}")

test()