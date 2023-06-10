import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import joblib
import model

# Load SVM model
svm = joblib.load('./models/svm.pkl')

# Load CNN model
cnn_model = model.CNN()
cnn_model.load_state_dict(torch.load('./models/cnn.pth'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载阈值
ood_threshold = np.load('./models/ood_threshold.npy')


# Load data from CSV file
data = np.genfromtxt('..\data.csv', delimiter=',', dtype=np.float32)

# Convert data to tensor
data = torch.from_numpy(data)

# Reshape data
data = data.reshape(60, 1, 28, 28)

# Extract features using the CNN model
_, data_features = cnn_model(data)

# Perform OOD detection
ood_scores = svm.decision_function(data_features.detach().numpy())
ood_predictions = np.where(ood_scores <= ood_threshold, 1, -1)

# Loop over predictions and classify
for i in range(ood_predictions.size):
    if ood_predictions[i] == -1:
        continue
    else:
        output, _ = cnn_model(data[i].unsqueeze(0))
        output = torch.Tensor(output)
        output = torch.argmax(output, dim=1)
        ood_predictions[i] = output

# Classification and plotting
preds = ood_predictions

# Traverse each sample and plot the image
# 1000 samples in total
for i in range(data.size(0)):
    # 将所有图片呈现在一张图上
    plt.subplot(6, 10, i + 1)
    plt.axis('off')
    plt.imshow(data[i].squeeze().numpy(), cmap='gray_r')
    plt.title(preds[i].item())
plt.show()