r"""[In development] torecsys.estimators is a sub module of estimators, which can be called driectly with a fitted dataloader of inputs
"""

from logging import Logger
from os import path
from pathlib import Path
from texttable import Texttable
import torch
import torch.utils.data
from torecsys.trainer import Trainer
from torecsys.utils.logging import TqdmHandler
from typing import Dict
import warnings

# ignore import warnings of the below packages
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # from tensorboardX import SummaryWriter
    from torch.utils.tensorboard import SummaryWriter
    from tqdm.autonotebook import tqdm

class _Estimator(Trainer):
    r"""Base class of estimator provide several function would be used in training and 
    inference."""
    def __init__(self, **kwargs):
        super(_Estimator, self).__init__(**kwargs)

from .ctr import *
from .emb import *
from .ltr import *
