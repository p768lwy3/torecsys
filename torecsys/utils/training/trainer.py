from ..logging import TqdmHandler
from logging import Logger
from texttable import Texttable
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm


class Trainer(object):
    def __init__(self, 
                 embeddings, 
                 model, 
                 regularizer, 
                 loss, 
                 optimizer,
                 verbose  : int = 2):
        
        self.embeddings = embeddings
        self.model = model
        self.regularizer = regularizer
        self.loss = self.loss
        self.optimizer = self.optimizer
        
        if verbose >= 1:
            self.logger = Logger("trainer")
            self.logger.setLevel("DEBUG")
            handelr = TqdmHandler()
            self.logger.addHandler(handelr)
        
        if verbose >= 2:
            self.writer = SummaryWriter(logdir=logdir)
    
    def _add_graph(self, inputs):
        return 
    
    def _describe(self):
        return 
        
    def _iterate(self, batch):
        return 
    
    def fit(self, dataloader):
        return
    
    def predict(self, batch):
        return
    
    def save(self):
        return
    
    def load(self):
        return
    