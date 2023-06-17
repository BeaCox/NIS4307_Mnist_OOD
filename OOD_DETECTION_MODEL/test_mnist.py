import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import oodcls

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

correct = 0
oodcls = oodcls.OodCls()
mnist_data = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(mnist_data,batch_size=1000,shuffle=False)
for data ,target in test_loader:
    data, target = data.to(device), target.to(device)
    preds = oodcls.classify(data)
    preds = np.array(preds)
    target = np.array(target)
    correct += sum(preds==target)

print("Mnist Accuracy: {}/{} ({:.2f}%)\n".format(correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
