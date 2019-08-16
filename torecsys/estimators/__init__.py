r"""[In development] torecsys.estimators is a sub module of estimators, which can be called driectly with a fitted dataloader of inputs
"""

from torecsys.utils.logging import TqdmHandler
from logging import Logger
from texttable import Texttable
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from typing import Callable

class _Estimator(nn.Module):
    r"""Base class of estimator provide several function would be used in training and 
    inference."""
    def __init__(self, 
                 model    : Callable,
                 loss     : Callable,
                 optimizer: Callable, 
                 epochs   : int,
                 verbose  : int,
                 logdir   : logdir = None):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.epcohs = epochs
        self.verbose = verbose
        
        if verbose >= 1:
            self.logger = Logger()
            handler = TqdmHandler()
            self.logger.addHandler(handler)
            self.logger.setLevel("DEBUG")
        
        if verbose == 2:
            self.writer = SummaryWriter(logdir=logdir)

        super(_Estimator, self).__init__()
    
    def _describe(self):
        return

    def _add_graph(self):
        return

    def _iteration(self, batch_data):
        return 

    def fit(self, dataloader: torch.utils.data.DataLoader):
        for e in self.epochs:
            for i, (batch_x, batch_y) in enumerate(dataloader):
                break
    
    def predict(self, batch_data):
        return
    
    def save(self):
        return
    
    def load(self):
        return

from .ctr import *
from .emb import *
from .ltr import *
