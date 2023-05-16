# -*- coding: utf-8 -*-
import torch
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import oodcls

test = MNIST(
    './data',
    train=False,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
    ]),
)

test_loader = torch.utils.data.DataLoader(test,batch_size=1000,shuffle=True)

oodcls=oodcls.OodCls('./models/model.pth')

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
preds = oodcls.classify(example_data)
fig = plt.figure()
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Prediction: {}".format(preds[i].item()))
    plt.xticks([])
    plt.yticks([])
plt.show()
