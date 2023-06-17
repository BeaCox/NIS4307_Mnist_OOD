# -*- coding: utf-8 -*-
import torch
import torchvision.transforms as transforms
import numpy as np
from model_vae import VAE_CNN
from matplotlib import pyplot as plt
import torchvision.utils as vutils

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

class OodCls:
    # 加载模型
    def __init__(self):
        self.model = [VAE_CNN() for i in range(10)]
        for i in range (10):
            self.model[i].load_state_dict(torch.load('VAEs.pth')[f'VAE_digit_{i}'])
            self.model[i].eval()
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])


    # 分类
    def classify(self, img):
        """
        img: tensor n*1*28*28
        preds: tensor n*1
        """
        
        def compute_reconstruction_error(data, model): # 计算重构误差
            with torch.no_grad():
                recon_data, _, _ = model(data)
                recon_error = ((recon_data - data)**2).sum(dim=(1,2,3))
                return recon_error.cpu().numpy()
        
        thresholds = np.load('./thresholds.npy')

        # 计算生成样本在每个模型上的重构误差
        test_errors = []
        with torch.no_grad():
            img = img.to(device)
            errors = np.stack([compute_reconstruction_error(img, vae) for vae in self.model], axis=-1)
            test_errors.extend(errors)
        test_errors = np.array(test_errors)

        # 对生成样本进行分类
        test_pred = np.argmin(test_errors, axis=1)
        # 如果重构误差大于阈值，则判断为OOD
        test_pred = [pred if error[pred] <= thresholds[pred] else -1 for pred, error in zip(test_pred, test_errors)]

        return test_pred

if __name__ == '__main__':

    
    oodcls = OodCls()
    data = np.genfromtxt('..\YYS_EARLY_LAYER\\tensor2.csv', delimiter=',', dtype=np.float32)
    # Convert data to tensor
    data = torch.from_numpy(data)
    # Reshape data
    data = data.reshape(10, 1, 28, 28)

    print(oodcls.classify(data))

    for i in range(data.size(0)):
        plt.subplot(2, 5, i + 1)
        plt.axis('off')
        plt.imshow(data[i].squeeze().numpy(), cmap='gray_r')
        plt.title(oodcls.classify(data)[i])
    plt.show()

