# -*- coding: utf-8 -*-
import torch
import torchvision.transforms as transforms
import model

class OodCls:
    # 加载模型
    def __init__(self, model_path):
        self.model = model.Model()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    # 分类
    def classify(self, img):
        """
        img: tensor n*1*28*28
        preds: tensor n*1
        """
        with torch.no_grad():
            output = self.model(img)
            preds = output.data.max(1, keepdim=True)[1]
        return preds
