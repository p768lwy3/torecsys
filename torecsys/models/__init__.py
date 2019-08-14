r"""torecsys.models is a sub module of building recommendation system model, including inputs embedding, implementation of algorithms
"""
import torch.nn as nn

class _Estimator(nn.Module):
    def __init__(self):
        super(_Estimator, self).__init__()

from .inputs import *
from .utils import *
