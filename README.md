## 基本思路

参考了[Detecting Out-of-Distribution Inputs in Deep Neural Networks Using an Early-Layer Output](https://arxiv.org/pdf/1910.10307.pdf)[^1]。论文的主要观点是：

> **假设存在一个潜在的空间可以将ID和OoD区分开，并且变换到这个空间的函数可以由网络的某一层良好估计。**

### 寻找*最佳OOD识别层*(OODL)

最后一层的特征对区分ID数据内部的分类很重要，但是不一定适合区分ID和OOD。由于我们的CNN模型一共只有4层，因此我们选择了第三层线性层的低维特征用于区分ID和OOD。

### 输入预处理

对数据加入一些微扰，能够增强ID的特征，可以强化ID数据分类功能的鲁棒性。

### OOD检测

+ $$Q: R^n\rightarrow[0, 1]^c: 深度网络模型$$
+ $$x \in R^n: 输入$$
+ $$Q_i: 网络对第i个类的概率输出 $$
+ $$X={x_1,…,x_m}: 训练集$$
+ $$L: 网络层数$$
+ $$q^l: 第l层的输出(q^0=x)$$
+ $$l_0: OODL层$$
+ $$Q^{l_0}: 第l_0层的概率输出$$
+ $$S^{l_0}: 基于特征Q^{l_0}训练的区分ID和OOD的分类器$$

为了不用OOD数据进行训练，将分类设定为单分类问题，使用`sklearn`库中的`OneClassSVM`来实现。

OOD检测机制如下：

$$
O_{l_0}(x;\delta)=\left\{
\begin{aligned}
&0, \delta\geq S^{l_0}(q^{l_0}(x)) \\
&1, otherwise\\
\end{aligned}
\right.
$$

其中$$O_{l_0}是检测函数，\delta是OOD阈值$$

## 核心代码分析

```python
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
np.random.seed(random_seed)
```

手动设置种子，使实验结果能够复现。

```python
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
```

对Mnist训练数据进行一定的变换，增强数据。

```
    # 前向传播
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 512)
        early_output = self.fc1(x)
        x = self.fc2(early_output)
        return x, early_output
```

在模型的定义中，返回值有两个，其中`x`为ID数据分类的`softmax`值，而`early_output`为中间层特征输出。

```python
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
```

该函数提取中间层输出作为特征向量`np.concatenate`将所有`early_output`连接在一起，形成一个特征向量矩阵，并将其返回。

```python
mnist_features = extract_early_output(model, device, train_loader)
svm = OneClassSVM()
svm.fit(mnist_features)
ood_threshold = np.percentile(svm.decision_function(mnist_features), ood_threshold_percentile)
```

提取MNIST的中间层输出作为特征向量，用于训练OneClassSVM。`decision_function`返回一个距离数组，表示每个样本到决策边界的有符号距离。这些距离值越小，表示样本越接近决策边界，越可能是异常样本；距离值越大，表示样本远离决策边界，越可能是正常样本。`np.percentile`函数根据指定的`ood_threshold_percentile`百分位数，计算OOD阈值。

### 实验结果

该模型在检测位于ID/OOD边界的数据（如基于Mnist生成的较优数据）时，效果不如基于生成器的方法好。

[^1]: Abdelzad V, Czarnecki K, Salay R, et al. Detecting out-of-distribution inputs in deep neural networks using an early-layer output[J]. arXiv preprint arXiv:1910.10307, 2019.
