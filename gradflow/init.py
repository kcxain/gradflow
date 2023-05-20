import math
import gradflow as gf


def rand(*shape,
         low=0.0,
         high=1.0,
         device=None,
         dtype="float32",
         requires_grad=False):
    """ Generate random numbers uniform between low and high """
    device = gf.cpu() if device is None else device
    array = device.rand(*shape) * (high - low) + low
    return gf.Tensor(array,
                     device=device,
                     dtype=dtype,
                     requires_grad=requires_grad)


def randn(*shape,
          mean=0.0,
          std=1.0,
          device=None,
          dtype="float32",
          requires_grad=False):
    """ Generate random normal with specified mean and std deviation """
    device = gf.cpu() if device is None else device
    array = device.randn(*shape) * std + mean
    return gf.Tensor(array,
                     device=device,
                     dtype=dtype,
                     requires_grad=requires_grad)


def constant(*shape, c=1.0, device=None, dtype="float32", requires_grad=False):
    """ Generate constant Tensor """
    device = gf.cpu() if device is None else device
    array = device.ones(*shape, dtype=dtype) * c  # note: can change dtype
    return gf.Tensor(array,
                     device=device,
                     dtype=dtype,
                     requires_grad=requires_grad)


def ones(*shape, device=None, dtype="float32", requires_grad=False):
    """ Generate all-ones Tensor """
    return constant(*shape,
                    c=1.0,
                    device=device,
                    dtype=dtype,
                    requires_grad=requires_grad)


def zeros(*shape, device=None, dtype="float32", requires_grad=False):
    """ Generate all-zeros Tensor """
    return constant(*shape,
                    c=0.0,
                    device=device,
                    dtype=dtype,
                    requires_grad=requires_grad)


def randb(*shape, p=0.5, device=None, dtype="bool", requires_grad=False):
    """ Generate binary random Tensor """
    device = gf.cpu() if device is None else device
    array = device.rand(*shape) <= p
    return gf.Tensor(array,
                     device=device,
                     dtype=dtype,
                     requires_grad=requires_grad)


def one_hot(n, i, device=None, dtype="float32", requires_grad=False):
    """ Generate one-hot encoding Tensor """
    device = gf.cpu() if device is None else device
    return gf.Tensor(device.one_hot(n, i.numpy(), dtype=dtype),
                     device=device,
                     requires_grad=requires_grad)


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):

    a = gain * math.sqrt(6 / (fan_in + fan_out))
    return a * (2 * rand(fan_in, fan_out, **kwargs) - 1)


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):

    std = gain * math.sqrt(2 / (fan_in + fan_out))
    return std * randn(fan_in, fan_out, **kwargs)


def kaiming_uniform(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"

    bound = math.sqrt(2) * math.sqrt(3 / fan_in)
    return bound * (2 * rand(fan_in, fan_out, **kwargs) - 1)


def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"

    gain = math.sqrt(2)
    std = gain / math.sqrt(fan_in)
    return std * randn(fan_in, fan_out, **kwargs)
