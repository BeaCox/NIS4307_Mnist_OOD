import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.svm import OneClassSVM
import joblib
import config
import model

# 从config.yml中读取超参数
cfg = config.load()
n_epochs, batch_size_train, batch_size_test, learning_rate, random_seed, ood_threshold_percentile = cfg[
    'n_epochs'], cfg['batch_size_train'], cfg['batch_size_test'], cfg[
        'learning_rate'], cfg['random_seed'], cfg['ood_threshold_percentile']

# 设置随机种子
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
np.random.seed(random_seed)

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 实例化模型并将其移动到设备上
model = model.CNN().to(device)
# 优化器选择Adamax，效果最好
optimizer = optim.Adamax(model.parameters(), lr=learning_rate)
# 损失函数选择
loss_fn = F.nll_loss

# 加载MNIST数据集，数据增强
train_dataset = datasets.MNIST(
    './data',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.RandomAffine(degrees=5, translate=(0.1, 0.1)),
        # 转换为Tensor类型
        transforms.ToTensor(),
        # 标准化
        transforms.Normalize((0.5, ), (0.5, ))
    ]))

test_dataset = datasets.MNIST(
    './data',
    train=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ), (0.3081, ))
    ]))

# 加载Fashion-MNIST数据集作为OOD数据集用来训练阈值
ood_train_dataset = datasets.FashionMNIST('./data',
                                          train=True,
                                          download=True,
                                          transform=transforms.Compose([
                                            transforms.RandomRotation(10),  # 随机旋转，角度范围为-10~10度
                                            transforms.RandomHorizontalFlip(),  # 随机水平翻转
                                            transforms.RandomVerticalFlip(),  # 随机垂直翻转
                                            transforms.ToTensor(),  # 转换为Tensor类型
                                            transforms.Normalize((0.5, ), (0.5, ))  # 标准化
                                          ]))

# 加载Fashion-MNIST数据集作为OOD数据集用来测试
ood_test_dataset = datasets.FashionMNIST('./data',
                                         train=False,
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.1307, ),
                                                                  (0.3081, ))
                                         ]))

# 创建数据加载器
train_loader = DataLoader(train_dataset,
                          batch_size=batch_size_train,
                          shuffle=True)
test_loader = DataLoader(test_dataset,
                         batch_size=batch_size_test,
                         shuffle=False)
ood_train_loader = DataLoader(ood_train_dataset,
                              batch_size=batch_size_train,
                              shuffle=False)
ood_test_loader = DataLoader(ood_test_dataset,
                             batch_size=batch_size_test,
                             shuffle=False)


# 定义训练函数
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # 将数据移动到设备上
        data, target = data.to(device), target.to(device)
        # 梯度清零
        optimizer.zero_grad()
        # 预测（前向传播）
        output, _ = model(data)
        # 计算损失
        loss = loss_fn(output, target)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    # 保存模型
    torch.save(model.state_dict(), './models/cnn.pth')


# 定义测试函数
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            test_loss += loss_fn(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


# 提取Early-Layer输出作为特征向量
def extract_early_output(model, device, data_loader):
    model.eval()
    feature_vectors = []
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            _, early_output = model(data)
            feature_vectors.append(early_output.cpu().numpy())
    feature_vectors = np.concatenate(feature_vectors)
    return feature_vectors


# 训练模型
for epoch in range(1, n_epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)

# 提取MNIST和Fashion-MNIST的中间层输出
mnist_features = extract_early_output(model, device, train_loader)
# fashion_mnist_features = extract_early_output(model, device, ood_train_loader)

# 将MNIST和Fashion-MNIST的特征向量合并为训练集
# train_features = np.concatenate((mnist_features, fashion_mnist_features),
#                                 axis=0)

# 训练并保存One-class SVM模型
svm = OneClassSVM()
svm.fit(mnist_features)
ood_threshold = np.percentile(svm.decision_function(mnist_features), ood_threshold_percentile)
# 输出ood_threshold到numpy文件
np.save('./models/ood_threshold.npy', ood_threshold)
joblib.dump(svm, './models/svm.pkl')