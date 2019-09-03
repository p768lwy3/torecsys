r"""[In development] torecsys.estimators is a sub module of estimators, which can be called driectly with a fitted dataloader of inputs
"""

from torecsys.utils.logging import TqdmHandler
from logging import Logger
from os import path
from pathlib import Path
from texttable import Texttable
from typing import Callable, Dict
import warnings

# ignore import warnings of the below packages
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # from tensorboardX import SummaryWriter
    from torch.utils.tensorboard import SummaryWriter
    from tqdm.autonotebook import tqdm

class _Estimator(object):
    r"""Base class of estimator provide several function would be used in training and 
    inference."""
    def _init_logger(self, 
                     name    : str,
                     level   : str  = "DEBUG",
                     handler : list = [TqdmHandler()],
                     log_dir : str  = "./log"):
        r"""Initialize loggers' variables.
        
        Args:
            name (str): Name of logging.Logger
            level (str): Level of logging.Logger. Default to DEBUG.
            handler (list): List of logging handler. Default to [TqdmHandler()].
            log_dir (str): Directory to save the log of tensorboard summary writer. Default to ./log.
        """
        if self.verboses >= 1:
            # initialize logger of trainer
            self.logger = Logger(name)

            # set level to logger
            self.logger.setLevel(level)
            
            # set handler to logger
            for h in handler:
                self.logger.addHandler(h)
            
            # print statement
            self.logger.info("logger has been initialized.")
        
        if self.verboses >= 2:
            # initialize tensorboard
            # to be confirmed if this is correct
            self.log_dir = path.join(path.dirname(__file__), log_dir)

            # create the folder if log_dir is not exist
            Path(log_dir).mkdir(parents=True, exist_ok=True)

            # initialize tensorboard summary writer with the given log_dir
            self.writer = SummaryWriter(log_dir=log_dir)

            # print statement
            self.logger.info("tensorboard summary writter has been initialized and the log directory is set to %s." % (self.log_dir))
    
    def _describe(self):
        r"""Summary summary of estimator
        """
        return

    def fit(self, dataloader: torch.utils.data.DataLoader):
        for e in self.epochs:
            for i, (batch_x, batch_y) in enumerate(dataloader):
                break
    
    def save(self):
        return
    
    def load(self):
        return
    
    # required functions: _iterate, predict, evaluate, _add_embedding, _add_graph, to_cuda, to_jit

from .ctr import *
from .emb import *
from .ltr import *
