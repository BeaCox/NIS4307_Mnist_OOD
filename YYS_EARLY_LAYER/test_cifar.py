import torch
from torchvision import transforms
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import joblib
import config
import model

# 从config.yml中读取超参数
cfg = config.load()
n_epochs, batch_size_train, batch_size_test, learning_rate, random_seed, ood_threshold_percentile = cfg[
    'n_epochs'], cfg['batch_size_train'], cfg['batch_size_test'], cfg[
        'learning_rate'], cfg['random_seed'], cfg['ood_threshold_percentile']

transform3 = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), # 彩色图像转灰度图像num_output_channels默认1
    transforms.Resize([28, 28]),
    transforms.ToTensor()
])

# 加载Fashion-MNIST数据集作为OOD数据集用来测试
cifar_data = datasets.CIFAR10(root='..\data', train=False, download=True, transform=transform3)
cifar_loader = DataLoader(cifar_data, batch_size= batch_size_test, shuffle=False)

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
fastion_test_features = extract_early_output(model, device, cifar_loader)
# 进行OOD检测
ood_scores2 = svm.decision_function(fastion_test_features)
ood_predictions2 = np.where(ood_scores2 <= ood_threshold, 1, -1)
# 统计OOD结果比例
ood_ratio2 = np.sum(ood_predictions2 == -1) / len(ood_predictions2)
print('OOD ratio: {:.2f}%'.format(ood_ratio2 * 100))
