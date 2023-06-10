# NIS4307_Mnist_OOD
SJTU NIS4307 课程大作业。

## 训练

 在[config.yml](./config.yml)中设置超参数。

## 接口调用

Example:

```python
# 导入oodcls.py
import oodcls

# 传入模型路径，实例化oodcls类
oodcls=oodcls.OodCls('./models/model.pth')

"""
调用oodcls类的classify函数
img: tensor n*1*28*28
preds: tensor n*1，-1代表ODD，0-9代表数字
"""
preds = oodcls.classify(img)
```

