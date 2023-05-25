# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F


# 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 卷积层1
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, 
                out_channels=16, 
                kernel_size=5
            ),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
        )
        # 卷积层2
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16, 
                out_channels=32, 
                kernel_size=5
            ),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )
        # 全连接层
        self.fc1 = nn.Sequential(
            nn.Linear(512, 50),
            nn.ReLU(),
            nn.Dropout(),
        )
        # 输出层
        self.fc2 = nn.Sequential(
            nn.Linear(50, 10),
            nn.LogSoftmax(dim=1),
        )

    # 前向传播
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 512)
        early_output = self.fc1(x)
        x = self.fc2(early_output)
        return x, early_output
