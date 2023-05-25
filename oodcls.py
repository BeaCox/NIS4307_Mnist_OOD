import torch
from sklearn.svm import OneClassSVM
import joblib
import torchvision.transforms as transforms
import model

class OodCls:
    # 加载模型
    def __init__(self, cnn_path, svm_path):
        self.cnn = model.CNN()
        self.cnn.load_state_dict(torch.load(cnn_path))
        self.cnn.eval()
        self.svm = joblib.load(svm_path)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    # 分类
    def classify(self, img):
        """
        img: tensor n*1*28*28
        preds: tensor n*1
        """
        # 先判断是否为OOD，img为tensor n*1*28*28
        # 提取img的中间层输出
        _, early_output = self.cnn(img)
        early_output = early_output.detach().numpy()

        # 判断是否为OOD
        preds = self.svm.predict(early_output)
        # 将preds转换为tensor
        preds = torch.from_numpy(preds)
        # 将preds的形状转换为n*1
        preds = preds.view(-1, 1)

        # 循环判断是否为OOD
        for i in range(preds.size(0)):
            # 如果为OOD，直接跳过
            if preds[i] == -1:
                continue
            # 如果不为OOD，进行分类
            else:
                output, _ = self.cnn(img[i].unsqueeze(0))
                output = torch.Tensor(output)
                output = torch.argmax(output, dim=1)
                preds[i] = output

        return preds