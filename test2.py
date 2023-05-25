import torch
from torchvision import transforms
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import joblib
import model

test_dataset = datasets.MNIST(
    './data',
    train=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ), (0.3081, ))
    ]))

# 加载Fashion-MNIST数据集作为OOD数据集用来测试
ood_test_dataset = datasets.FashionMNIST('./data',
                                         train=False,
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.1307, ),
                                                                  (0.3081, ))
                                         ]))
test_loader = DataLoader(test_dataset,
                         batch_size=1000,
                         shuffle=False)
ood_test_loader = DataLoader(ood_test_dataset,
                             batch_size=1000,
                             shuffle=False)
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
svm = joblib.load('./models/svm.pkl')
model = model.CNN()
model.load_state_dict(torch.load('./models/cnn.pth'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
# 读取阈值
ood_threshold = np.load('./models/ood_threshold.npy')
test_features = extract_early_output(model, device, test_loader)
fastion_test_features = extract_early_output(model, device, ood_test_loader)
# 进行OOD检测
ood_scores = svm.decision_function(test_features)
ood_scores2 = svm.decision_function(fastion_test_features)
ood_predictions = np.where(ood_scores <= ood_threshold, 1, -1)
ood_predictions2 = np.where(ood_scores2 <= ood_threshold, 1, -1)
# 统计OOD结果比例
ood_ratio = np.sum(ood_predictions == -1) / len(ood_predictions)
ood_ratio2 = np.sum(ood_predictions2 == -1) / len(ood_predictions2)
print('OOD ratio: {:.2f}%'.format(ood_ratio * 100))
print('OOD ratio: {:.2f}%'.format(ood_ratio2 * 100))
