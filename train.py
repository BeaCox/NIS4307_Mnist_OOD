# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import MNIST
import config
import model

# 从config.yml中读取超参数
cfg = config.load()
n_epochs, batch_size_train, batch_size_test, learning_rate, log_interval, random_seed = cfg['n_epochs'], cfg['batch_size_train'], cfg['batch_size_test'], cfg['learning_rate'], cfg['log_interval'], cfg['random_seed']

# 设置随机种子
torch.manual_seed(random_seed)

# 训练集
train = MNIST(
    './data/ID',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
    ]),
)

# 测试集
test = MNIST(
    './data/ID',
    train=False,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
    ]),
)

# 迭代器
train_loader = torch.utils.data.DataLoader(train,batch_size=batch_size_train,shuffle=True)
test_loader = torch.utils.data.DataLoader(test,batch_size=batch_size_test,shuffle=True)

# 初始化模型
model = model.Model()

# 优化器选择Adamax，效果最好
optimizer = optim.Adamax(model.parameters(), lr=learning_rate)
# 优化器选择Adam
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# 优化器选择Adagrad
# optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
# 优化器选择ASGD
# optimizer = optim.ASGD(model.parameters(), lr=learning_rate)
# 优化器选择Rprop
# optimizer = optim.Rprop(model.parameters(), lr=learning_rate)

# 损失函数选择negative log likelihood loss
loss_fn = F.nll_loss

# 数据可视化
train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]


# 训练
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # 梯度清零
        optimizer.zero_grad()
        # 预测（前向传播）
        output = model(data)
        # 计算损失
        loss = loss_fn(output, target)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 打印日志
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx * 64) +
                                 ((epoch - 1) * len(train_loader.dataset)))
            # 保存模型
            torch.save(model.state_dict(), './models/model.pth')


# 测试
def evaluate():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += loss_fn(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, n_epochs + 1):
    train(epoch)

evaluate()

# todo: OOD检测

# examples = enumerate(test_loader)
# batch_idx, (example_data, example_targets) = next(examples)
# with torch.no_grad():
#     output = model(example_data)
# fig = plt.figure()
# for i in range():
#     plt.subplot(2, 3, i + 1)
#     plt.tight_layout()
#     plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#     plt.title("Prediction: {}".format(output.data.max(1, keepdim=True)[1][i].item()))
#     plt.xticks([])
#     plt.yticks([])
# plt.show()
