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

# 彩色图转灰度图
transform3 = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), # 彩色图像转灰度图像num_output_channels默认1
    transforms.Resize([28, 28]),
    transforms.ToTensor()
])


def compute_reconstruction_error(data, model): # 计算重构误差
    with torch.no_grad():
        recon_data, _, _ = model(data)
        recon_error = ((recon_data - data)**2).sum(dim=(1,2,3))
        return recon_error.cpu().numpy()

cifar_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform3)
cifar_loader = DataLoader(cifar_data, batch_size=batch_size, shuffle=False)

# 创建VAE模型列表
num_vaes = 10
vaes = [VAE_CNN() for i in range(num_vaes)]

# 加载模型状态字典
model_dict = torch.load('VAEs.pth')
for i in range(num_vaes):
    vaes[i].load_state_dict(model_dict[f'VAE_digit_{i}'])

# 对于每个测试样本，计算其对于每个模型的重构误差
test_errors = []
# 计算 CIFAR 样本在每个模型上的重构误差
cifar_errors = []
with torch.no_grad():
    for data, _ in cifar_loader:
        img = data.to(device)
        errors = np.stack([compute_reconstruction_error(img, vae) for vae in vaes], axis=-1)
        cifar_errors.extend(errors)
cifar_errors = np.array(cifar_errors)

# 对cifarMNIST样本进行分类
cifar_pred = np.argmin(cifar_errors, axis=1)
# 如果重构误差大于阈值，则判断为OOD
thresholds = np.load('./thresholds.npy')
cifar_pred = [pred if error[pred] <= thresholds[pred] else 10 for pred, error in zip(cifar_pred, cifar_errors)]
# 计算OOD检测的准确率，精确度，召回率和F1分数
cifar_true = [10] * len(cifar_pred)  # 10表示OOD
accuracy = accuracy_score(cifar_true, cifar_pred)
precision = precision_score(cifar_true, cifar_pred, average='weighted')
recall = recall_score(cifar_true, cifar_pred, average='weighted')
f1 = f1_score(cifar_true, cifar_pred, average='weighted')
print(f"OOD Detection: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")