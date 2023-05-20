"""The module.
"""
from typing import List, Callable, Any
from typing_extensions import Required
from gradflow.autograd import Tensor
from gradflow import ops
import gradflow.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:

    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):

    def forward(self, x):
        return x


class Linear(Module):

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 device=None,
                 dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(
            init.kaiming_uniform(in_features, out_features, dtype=dtype))
        if bias:
            self.bias = Parameter(
                init.kaiming_uniform(out_features, 1, dtype=dtype).reshape(
                    (1, out_features)))
        else:
            self.bias = None

    def forward(self, X: Tensor) -> Tensor:

        X_out = X @ self.weight
        if self.bias:
            return X_out + self.bias.broadcast_to(X_out.shape)
        return X_out


class Flatten(Module):

    def forward(self, X):

        return X.reshape((X.shape[0], -1))


class ReLU(Module):

    def forward(self, x: Tensor) -> Tensor:

        return ops.relu(x)


class Sequential(Module):

    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:

        for module in self.modules:
            x = module(x)
        return x


class SoftmaxLoss(Module):

    def forward(self, logits: Tensor, y: Tensor):

        exp_sum = ops.logsumexp(logits, axes=(1, )).sum()
        z_y_sum = (logits * init.one_hot(logits.shape[1], y)).sum()
        return (exp_sum - z_y_sum) / logits.shape[0]


class BatchNorm1d(Module):

    def __init__(self,
                 dim,
                 eps=1e-5,
                 momentum=0.1,
                 device=None,
                 dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum

        self.weight = Parameter(init.ones(self.dim), requires_grad=True)
        self.bias = Parameter(init.zeros(self.dim), requires_grad=True)
        self.running_mean = init.zeros(self.dim)
        self.running_var = init.ones(self.dim)

    def forward(self, x: Tensor) -> Tensor:

        batch_size = x.shape[0]
        feature_size = x.shape[1]
        # running estimates
        mean = x.sum(axes=(0, )) / batch_size
        x_minus_mean = x - mean.broadcast_to(x.shape)
        print(mean)
        print(mean.broadcast_to(x.shape))
        var = (x_minus_mean**2).sum(axes=(0, )) / batch_size

        if self.training:
            self.running_mean = (
                1 -
                self.momentum) * self.running_mean + self.momentum * mean.data
            self.running_var = (1 - self.momentum
                                ) * self.running_var + self.momentum * var.data

            x_std = ((var + self.eps)**0.5).broadcast_to(x.shape)
            normed = x_minus_mean / x_std
            return normed * self.weight.broadcast_to(
                x.shape) + self.bias.broadcast_to(x.shape)
        else:
            normed = (x - self.running_mean) / (self.running_var +
                                                self.eps)**0.5
            return normed * self.weight.broadcast_to(
                x.shape) + self.bias.broadcast_to(x.shape)


class LayerNorm1d(Module):

    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps

        self.weight = Parameter(init.ones(dim), requires_grad=True)
        self.bias = Parameter(init.zeros(dim), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:

        batch_size = x.shape[0]
        feature_size = x.shape[1]
        mean = x.sum(axes=(1, )).reshape((batch_size, 1)) / feature_size
        x_minus_mean = x - mean.broadcast_to(x.shape)
        x_std = ((x_minus_mean**2).sum(axes=(1, )).reshape(
            (batch_size, 1)) / feature_size + self.eps)**0.5
        normed = x_minus_mean / x_std.broadcast_to(x.shape)
        return self.weight.broadcast_to(
            x.shape) * normed + self.bias.broadcast_to(x.shape)


class Dropout(Module):

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:

        if self.training:
            return x * (init.randb(*x.shape, p=(1 - self.p))) / (1 - self.p)
        else:
            return x


class Residual(Module):

    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:

        return x + self.fn(x)
