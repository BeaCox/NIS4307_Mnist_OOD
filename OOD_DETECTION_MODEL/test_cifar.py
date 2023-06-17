import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score
import numpy as np
from model_vae import VAE_CNN
import config
import oodcls

oodcls = oodcls.OodCls()

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

cifar_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform3)
cifar_loader = DataLoader(cifar_data, batch_size=batch_size, shuffle=False)


with torch.no_grad():
    for data, _ in cifar_loader:
        img = data.to(device)
        cifar_pred = oodcls.classify(img)

# 计算OOD检测的准确率，精确度，召回率和F1分数
cifar_true = [-1] * len(cifar_pred)  # 10表示OOD
accuracy = accuracy_score(cifar_true, cifar_pred)
print("CIFAR Accuracy: {:.4f}\n".format(accuracy))