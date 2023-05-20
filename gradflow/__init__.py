__version__ = "0.0.1"


from .autograd import Tensor, cpu, all_devices
from . import ops
from .ops import *
from . import init
from . import data
from . import nn
from . import optim
