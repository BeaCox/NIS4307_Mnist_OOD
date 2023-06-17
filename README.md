# NIS4307_Mnist_OOD
SJTU NIS4307 课程大作业。

## 安装依赖

```python
 pip install -r requirements.txt
```

## 训练

 在[config.yml](./config.yml)中设置超参数。
 运行train.py函数进行模型训练

## 接口调用

```python
# 导入oodcls.py
import oodcls

# 实例化oodcls类
oodcls=oodcls.OodCls()

"""
调用oodcls类的classify函数
img: tensor n*1*28*28
preds: tensor n*1，-1代表ODD，0-9代表数字
"""
preds = oodcls.classify(img)
```

Example:
```python
# 导入oodcls.py
import oodcls

oodcls = oodcls.OodCls()
data = np.genfromtxt('..\data.csv', delimiter=',', dtype=np.float32)
# Convert data to tensor
data = torch.from_numpy(data)
# Reshape data
data = data.reshape(60, 1, 28, 28)

preds = oodcls.classify(data)

for i in range(data.size(0)):
    plt.subplot(6, 10, i + 1)
    plt.axis('off')
    plt.imshow(data[i].squeeze().numpy(), cmap='gray_r')
    plt.title(preds[i])
plt.show()
```

## 代码含义详解

 详见 CCY_GENERATIVE-MODEL 和 YYS_EARLY_LAYER 分支中的README.md。
 main分支中使用的是 CCY_GENERATIVE-MODEL 中的模型。

