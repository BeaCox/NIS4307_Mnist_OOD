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


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

def compute_reconstruction_error(data, model): # 计算重构误差
    with torch.no_grad():
        recon_data, _, _ = model(data)
        recon_error = ((recon_data - data)**2).sum(dim=(1,2,3))
        return recon_error.cpu().numpy()

# 创建VAE模型列表
num_vaes = 10
vaes = [VAE_CNN() for i in range(num_vaes)]

# 加载模型状态字典
model_dict = torch.load('VAEs.pth')
for i in range(num_vaes):
    vaes[i].load_state_dict(model_dict[f'VAE_digit_{i}'])

data = np.genfromtxt('..\data.csv', delimiter=',', dtype=np.float32)
# Convert data to tensor
data = torch.from_numpy(data)
# Reshape data
data = data.reshape(60, 1, 1, 28, 28) # 这里需要 reshape 成 5 个维度

thresholds = np.load('./thresholds.npy')

test_errors = []
# 计算生成样本在每个模型上的重构误差
generated_errors = []
with torch.no_grad():
    for num in range(data.size(0)):
        data[num] = data[num].to(device)
        errors = np.stack([compute_reconstruction_error(data[num], vae) for vae in vaes], axis=-1)
        generated_errors.extend(errors)
generated_errors = np.array(generated_errors)

# 对生成样本进行分类
generated_pred = np.argmin(generated_errors, axis=1)
# 如果重构误差大于阈值，则判断为OOD
generated_pred = [pred if error[pred] <= thresholds[pred] else -1 for pred, error in zip(generated_pred, generated_errors)]

for i in range(data.size(0)):
    # 将所有图片呈现在一张图上
    plt.subplot(6, 10, i + 1)
    plt.axis('off')
    plt.imshow(data[i].squeeze().numpy(), cmap='gray_r')
    plt.title(generated_pred[i])
plt.show()