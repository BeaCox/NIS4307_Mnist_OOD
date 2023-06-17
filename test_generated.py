import torch
from torchvision import datasets, transforms
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
import oodcls


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

oodcls = oodcls.OodCls()
data = np.genfromtxt('.\data.csv', delimiter=',', dtype=np.float32)
    # Convert data to tensor
data = torch.from_numpy(data)
    # Reshape data
data = data.reshape(60, 1, 28, 28)

preds = oodcls.classify(data)

for i in range(data.size(0)):
    plt.subplot(6, 10, i + 1)
    plt.axis('off')
    plt.imshow(data[i].squeeze().numpy(), cmap='gray_r')
    plt.title(preds[i])
plt.show()
