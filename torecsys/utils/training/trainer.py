from ..logging import TqdmHandler
from torecsys.functional.regularization import Regularizer
from torecsys.inputs import _Inputs
from torecsys.models import _Model
from logging import Logger
from os import path
from pathlib import Path
from texttable import Texttable
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
from typing import Dict
import warnings


class Trainer(object):
    def __init__(self,
                 inputs_wrapper : _Inputs,
                 model          : _Model,
                 regularizer    : Regularizer = Regularizer(0.1, 2),
                 loss           : type = nn.MSELoss,
                 optimizer      : type = optim.AdamW,
                 epochs         : int = 10,
                 verboses       : int = 2,
                 log_step       : int = 500,
                 log_dir        : str = "./logdir"):
        
        self.embeddings = embeddings
        self.model = model
        self.regularizer = regularizer
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs

        # streaming log in tqdm will be initialized 
        if verboses >= 1:
            # initialize logger of trainer
            self.logger = Logger("trainer")

            # set logger config, including level and handler
            self.logger.setLevel("DEBUG")
            handelr = TqdmHandler()
            self.logger.addHandler(handelr)

            self.logger.info("logger have been initialized.")
        
        # log in tensorboard will be initialized 
        if verboses >= 2:
            # store the path of log dir
            self.log_dir = path.join(path.dirname(__file__), log_dir)

            # create the folder if log_dir is not exist 
            Path(log_dir).mkdir(parents=True, exist_ok=True)

            # intitialize tensorboard summary writer with given log_dir
            self.writer = SummaryWriter(logdir=log_dir)

            # print the summary writer's location
            self.logger.info("tensorboard summary writter have been created and the log directory is set to %s." % (self.log_dir))

        self.verboses = verboses

    def _add_graph(self, 
                   samples_inputs : Dict[str, torch.Tensor], 
                   verboses       : bool = True):
        r"""Add graph data to summary.
        
        Args:
            samples_inputs (Dict[str, T]): A dictionary of variables to be fed.
            verboses (bool, optional): Whether to print graph structure in console. Defaults to True.
        """
        if self.verboses >= 2:
            self.writer.add_graph(self.model, samples_inputs, verboses=verboses)
        else:
            if self.verboses >= 1:
                self.logger.warn("_add_graph only can be called when self.verboses >= 2.")
            else:
                warnings.warn("_add_graph only can be called when self.verboses >= 2.")
        
    def _describe(self):
        return 
        
    def _iterate(self, batch_inputs: Dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        # zero the parameter gradients
        self.optimizer.zero_grad()
        
        # calculate forward prediction
        embed_inputs = self.inputs_wrapper(batch_inputs)
        outputs = self.model(**embed_inputs)

        # calculate loss and regularized loss
        loss = self.loss(outputs, labels)
        if self.regularizer is not None:
            reg_loss = self.regularizer(self.model.named_parameters)
            loss += reg_loss

        # calculate backward and optimize 
        loss.backward()
        optimizer.step()

        # return loss to log stream and tensorboard
        return loss
    
    def fit(self, dataloader: torch.utils.data.DataLoader):
        # initialize global_step = 0 for logging
        global_step = 0

        # loop through n epochs
        for epoch in self.epochs:
            # logging of the epoch
            if verboses >= 1:
                self.logger.info("Epoch %s / %s:" % (epoch + 1, self.epochs))
            
            # initialize progress bar of dataloader of this epoch
            pbar = tqdm(dataloader, desc="step loss : ??.????", ncols=100, ascii=True)

            for i, (batch_inputs, labels) in enumerate(pbar):
                # iteration of the batch
                loss = self._iterate(batch_inputs, labels)

                # set loss to the description of pbar
                pbar.set_description("step loss : %.4f" % (loss))

                # log for each y steps
                if global_step % log_dir == 0:
                    if self.verboses >= 1:
                
                global_step += 1


        
    
    def predict(self, batch):
        return
    
    def save(self):
        return
    
    def load(self):
        return
    