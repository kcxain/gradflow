<p align="center">
    <br>
    <img src="https://github.com/kcxain/gradflow/raw/master/assets/logo.png" width="360"/>
    <br>
<p>

<div align="center">

[![PyPI](https://img.shields.io/pypi/v/gradflow)](https://pypi.org/project/gradflow/)
[![license](https://img.shields.io/github/license/kcxain/gradflow)](https://github.com/kcxain/gradflow/blob/master/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/kcxain/gradflow?logo=github)](https://github.com/kcxain/gradflow)
[![GitHub latest commit](https://badgen.net/github/last-commit/kcxain/gradflow)](https://github.com/kcxain/gradflow/commit/)

<h4 align="center">
    <p>
      <b>中文</b> |
       <a href="https://github.com/kcxain/gradflow/blob/master/README_en.md"><b>English</b></a>
    <p>
</h4>

</div>


## 简介

GradFlow 是一个简单、高效的深度学习框架。它实现了基于反向模式自动微分算法的自动求导机制，并提供了深度学习训练与推理的必备组件，如 optimizers, data loaders 和 modules 等。

## 安装
- 更新 pip
  ```bash
  pip install --upgrade pip
  ```
- 使用 pip 安装 GradFlow 的最新发布版本
  ```bash
  pip install gradflow
  ```
- 如果你在中国，可以使用如下命令设置 Pypi 镜像
  ```bash
  pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
  ```

## 快速上手

GradFlow 各组件 API 的设计参考了 Pytorch，所以可以像使用 Pytorch 一样使用它，上手非常容易！

### Tensor
- 从 Python 的 list 或 numpy 的 array 中创建 Tensor
  ```python
  import gradflow as gf
  import numpy as np

  a = gf.Tensor(np.array([1,2,3]))
  # gradflow.Tensor([1 2 3])
  b = gf.Tensor([2,2,3])
  # gradflow.Tensor([2 2 3])
  ```
- 我们对 Tensor 类重写了 Python 的所有运算操作符，可使用这些运算符直接作用于整个 Tensor
  ```python
  a + b
  # gradflow.Tensor([3 4 6])
  a * b
  # gradflow.Tensor([2 4 9])
  a ** 2
  # gradflow.Tensor([1 4 9])
  ```
- 我们还实现了一些常用的 Tensor 变换操作，如 reshape, sum, broadcast_to, transpose 等
  ```python
  a = df.Tensor(np.random.randn(3,5))
  a.shape
  # (3, 5)
  a.reshape((5, 3))
  # shape: (5, 3)
  c = a.sum(1)
  # gradflow.Tensor([-1.28751241  1.05285348 -0.64622878 -0.38683152  1.55657958])
  # shape: (1, 5)
  d = c.broadcast_to((6,5))
  # shape: (6, 5)
  d.broadcast_to((1,6,5)).transpose((0, 1))
  # shape: (6, 5) -> (1, 6, 5) -> (6, 1, 5)
  ```

### 自动求导
- GradFlow 默认将 Tensor 的 `requires_grad` 属性设置为 `True`，对 Tensor 进行的所有操作都将被记录在**计算图**中
  ```python
  w = gf.Tensor([1, 2, 3], dtype="float32"), v = gf.Tensor([2, 3, 4], dtype="float32")
  w.requires_grad
  # True
  u = w + v
  u.inputs
  # (gradflow.Tensor([1. 2. 3.]), gradflow.Tensor([2. 3. 4.]))
  u.op
  # <gradflow.ops.EWiseAdd at 0x7fbe964a8820>
  l = u * v
  l.inputs
  # (gradflow.Tensor([3. 5. 7.]), gradflow.Tensor([2. 3. 4.]))
  l.inputs[0].op
  # <gradflow.ops.EWiseAdd at 0x7fbe964a8820>
  l.op
  # <gradflow.ops.EWiseMul at 0x7fc2eb451c70>
  l.backward()
  l.grad
  # gradflow.Tensor([1. 1. 1.])
  u.grad
  # gradflow.Tensor([2. 3. 4.])
  ```
- 可以使用 `data` 属性得到从计算图中分离出来的 Tensor，它的值与原 Tensor 相同
  ```python
  a = gf.Tensor(np.random.randn(3,5))
  a.data.requires_grad
  # False
  ```

### 神经网络库
和 Pytorh 一样，你可以继承抽象类 `nn.Module` 来创建自己的模型，`optim.Optimizer` 来创建自己的优化器, 以及 `data.Dataset`, `data.DataLoader` 来加载自己的数据集。

当然，我们也实现了一些常见的模型，优化器等。如：
- 数据预处理：`RandomFlipHorizontal`, `RandomCrop`
- 初始化方法：`xavier_uniform`, `xavier_normal`, `kaiming_uniform`, `kaiming_normal`
- 优化器：`SGD`, `Adam`
- 常用模块：`Linear`, `Flatten`, `ReLU`, `Sequential`, `SoftmaxLoss`, `BatchNorm1d`, `LayerNorm1d`, `Dropout`, `Residual`

这些实现都按照 Pytorch 的 API 来设计，所以你可以在 [Pytorch Docs](https://pytorch.org/docs/stable/index.html) 中查看它们的用法。

### 训练
下面给出一个使用 GradFlow 训练深度学习模型的基本流程：

- 实现自己的 **Dataset** 类
  ```python
  class MyDataset(Dataset):

    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        # pre-process methoe
        super().__init__(transforms)
        self.images, self.labels = get_data(image_filesname=image_filename,
                                               label_filename=label_filename)

    def __getitem__(self, index) -> object:
        X, y = self.images[index], self.labels[index]
        if self.transforms:
            X_in = X.reshape((28, 28, -1))
            X_out = self.apply_transforms(X_in)
            return X_out.reshape(-1, 28 * 28), y
        else:
            return X, y

    def __len__(self) -> int:
        return self.labels.shape[0]
  ```
- 指定模型和优化器
  ```python
  def train(batch_size=100, epochs=10, optimizer=gf.optim.Adam,
            lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
      train_data = get_train_data()
      test_data = get_test_data()
      # Your DataLoader
      train_loader = gf.data.DataLoader(train_data, batch_size)
      test_loader = gf.data.DataLoader(test_data, batch_size)
      # Your Model
      model = MLPResNet(784, hidden_dim=hidden_dim)
      # Your Optimizer
      opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
      # each epoch
      for _ in range(epochs):
          train_acc, train_loss = epoch(train_loader, model, opt)
      test_acc, test_loss = epoch(test_loader, model)
      return (train_acc, train_loss, test_acc, test_loss)
  ```
- 在每个 epoch 中，可以像这样更新参数
  ```python
  def epoch(dataloader, model, opt=None):
    hit, total = 0, 0
    loss_func = nn.SoftmaxLoss()
    loss_all = 0
    if opt is not None:
        model.train()
        for idx, data in enumerate(dataloader):
            x, y = data
            output = model(x)
            opt.reset_grad()
            loss = loss_func(output, y)
            loss_all += loss.numpy()
            loss.backward()
            opt.step()
            hit += (y.numpy() == output.numpy().argmax(1)).sum()
            total += y.shape[0]
    else:
        model.eval()
        for idx, data in enumerate(dataloader):
            x, y = data
            output = model(x)
            loss = loss_func(output, y)
            loss_all += loss.numpy()
            hit += (y.numpy() == output.numpy().argmax(1)).sum()
            total += y.shape[0]
    acc = (total - hit) / total
    return acc, loss_all / (idx + 1)
  ```

### 更多例子
我们在 [examples](examples) 文件夹下提供了使用 GradFlow 进行模型训练或推理的更多例子。

## License
本项目使用 [Apache License (Version 2.0)](https://github.com/kcxain/gradflow/blob/master/LICENSE).