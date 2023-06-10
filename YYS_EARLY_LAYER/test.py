import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import oodcls

oodcls=oodcls.OodCls('./models/cnn.pth', './models/svm.pkl')

# 读取CSV文件
data = np.genfromtxt('tensor.csv', delimiter=',', dtype=np.float32)

# 转换为张量
data = torch.from_numpy(data)
# 变换形状
data = data.reshape(10,1,28,28)

# 分类并画图
preds = oodcls.classify(data)
# 遍历每个样本并绘制图像
for i in range(data.size(0)):
    plt.subplot(2, 5, i + 1)
    img = vutils.make_grid(data[i], normalize=True)
    plt.imshow(img.permute(1, 2, 0))
    plt.axis('off')
    plt.title("{}".format(preds[i].item()))

# 调整子图布局
plt.tight_layout()

# 显示图像
plt.show()