r"""torecsys.models is a sub module of building recommendation system model, including inputs embedding, implementation of algorithms
"""
import torch.nn as nn

class _Estimator(nn.Module):
    def __init__(self):
        super(_Estimator, self).__init__()

from .ctr import *
from .emb import *
from .inputs import *
from .ltr import *
from .utils import *
