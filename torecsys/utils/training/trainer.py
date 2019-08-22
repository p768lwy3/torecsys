from ..logging import TqdmHandler
from logging import Logger
from texttable import Texttable
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm


class Trainer(object):
    def __init__(self, 
                 inputs_wrapper : torecsys.inputs.InputsWrapper, 
                 model          : torecsys.models._Model, 
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
        
    def _iterate(self, batch_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # zero the parameter gradients
        self.optimizer.zero_grad()
        
        # calculate forward prediction
        embed_inputs = self.inputs_wrapper(batch_inputs)
        outputs = self.model(**embed_inputs)

        # apply regularizer

        # calculate backward and optimize
        loss = self.loss(outputs)
        loss.backward()
        optimizer.step()

        # return loss to log stream and tensorboard
        return loss
        
    
    def fit(self, dataloader):
        return
    
    def predict(self, batch):
        return
    
    def save(self):
        return
    
    def load(self):
        return
    