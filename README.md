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

</div>

# Introduction

GradFlow is a deep learning framework designed to be **simple**, **scalable** and **efficient** with **PyTorch-like API**. 

GradFlow provides all the necessary components for training a deep learning model, including initialization methods, optimizers, data loaders, and modules.

We use reverse **accumulation method** which involves calculating the gradient from the outermost operation inwards to implement auto gradient.

# Install GradFlow
- Upgrade pip
  ```bash
  pip install --upgrade pip
  ```
- To install latest release of GradFlow
  ```bash
  pip install gradflow
  ```
- If you are in China, you could run this to have pip download packages from domestic mirror of pypi:
  ```bash
  pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
  ```

# Usage

GradFlow uses **PyTorch-like API**, So you can use it just like you would with pytorch.

## Tensor
- You can create a tensor from either a list or an array of numpy.
  ```python
  import gradflow as gf
  import numpy as np

  a = gf.Tensor(np.array([1,2,3]))
  # gradflow.Tensor([1 2 3])
  b = gf.Tensor([2,2,3])
  # gradflow.Tensor([2 2 3])
  ```
- Because we overload Python's basic operators, you can use any operator to directly operate on tensors
  ```python
  a + b
  # gradflow.Tensor([3 4 6])
  a * b
  # gradflow.Tensor([2 4 9])
  a ** 2
  # gradflow.Tensor([1 4 9])
  ```
- You can also perform some special operations
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

## Autograd
- By default, we create gradflow Tensors that sets requires_grad to be true. Every operation you do on Tensor will be recorded in the **Computational Graph**.
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
- You can use the `data` attribute to detach tensor from the computation graph
  ```python
  a = df.Tensor(np.random.randn(3,5))
  a.data.requires_grad
  # False
  ```

## Neural Network Library
Just like Pytorch, you can use abstract class `nn.Module` to create your own model, `optim.Optimizer` to create your own optimizer, and `data.Dataset`, `data.DataLoader` to load your dataset.

And we also provide some common **Models** and **Algorithms**, such as
- Data Pre-processing: `RandomFlipHorizontal`, `RandomCrop`
- Init methods: `xavier_uniform`, `xavier_normal`, `kaiming_uniform`, `kaiming_normal`
- Optimizers: `SGD`, `Adam`
- Modules: `Linear`, `Flatten`, `ReLU`, `Sequential`, `SoftmaxLoss`, `BatchNorm1d`, `LayerNorm1d`, `Dropout`, `Residual`

You can read the [Pytorch Docs](https://pytorch.org/docs/stable/index.html) to see their usage or see the examples in this reposity.

## Training

You can use the following templates to train the model.

- Implement your **Dataset**
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
- Specify or customize the model and optimizer
  ```python
  def train(batch_size=100, epochs=10, optimizer=df.optim.Adam,
            lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
      train_data = get_train_data()
      test_data = get_test_data()
      # Your DataLoader
      train_loader = df.data.DataLoader(train_data, batch_size)
      test_loader = df.data.DataLoader(test_data, batch_size)
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
- In each epoch, the parameters can be updated like this
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

## Examples
In the [examples](examples) folder you can find several other samples.

# License
This project is licensed under the [Apache License (Version 2.0)](https://github.com/kcxain/gradflow/blob/master/LICENSE).