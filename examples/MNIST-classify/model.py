import struct
import gzip
import numpy as np
import sys
import gradflow as df
import gradflow.nn as nn


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    modules = nn.Sequential(nn.Linear(dim, hidden_dim), norm(hidden_dim),
                            nn.ReLU(), nn.Dropout(drop_prob),
                            nn.Linear(hidden_dim, dim), norm(dim))
    return nn.Sequential(nn.Residual(modules), nn.ReLU())


def MLPResNet(dim,
              hidden_dim=100,
              num_blocks=3,
              num_classes=10,
              norm=nn.BatchNorm1d,
              drop_prob=0.1):
    modules = [nn.Linear(dim, hidden_dim), nn.ReLU()]
    for i in range(num_blocks):
        modules.append(
            ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob))
    modules.append(nn.Linear(hidden_dim, num_classes))
    return nn.Sequential(*modules)


def softmax_loss(Z, y_one_hot):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (df.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (df.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (df.Tensor[np.float32])
    """
    m = Z.shape[0]
    Z1 = df.ops.summation(
        df.ops.log(df.ops.summation(df.ops.exp(Z), axes=(1, ))))
    Z2 = df.ops.summation(Z * y_one_hot)
    return (Z1 - Z2) / m


def loss_err(h, y):
    """ Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = df.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
