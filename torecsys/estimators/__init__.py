r"""[In development] torecsys.estimators is a sub module of estimators, which can be called driectly with a fitted dataloader of inputs
"""

from torecsys.utils.logging import TqdmHandler
from logging import Logger
from os import path
from pathlib import Path
from texttable import Texttable
import torch
import torch.utils.data
from typing import Dict
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
    
    def _describe(self, _vars: Dict[str, str]):
        """Show summary of estimator
        
        Args:
            _vars (Dict[str, str]): Dictionary of arguments and hyperparamters of the estimator
        """
        # initialize and configurate Texttable
        t = Texttable()
        t.set_deco(Texttable.BORDER)
        t.set_cols_align(["l", "l"])
        t.set_cols_valign(["t", "t"])

        # append data to texttable
        t.add_rows(
            [["Name: ", "Value: "]] + \
            [[k.capitalize(), v] for k, v in _vars.items() if v is not None]
        )

        return t.draw()

    def fit(self, dataloader: torch.utils.data.DataLoader):
        # initialize global_step = 0 for logging
        global_step = 0

        # number of batches
        num_batch = len(dataloader)

        # loop through n epochs
        for epoch in range(self.epochs):
            # initialize loss variables to store aggregated loss
            steps_loss = 0.0
            epoch_loss = 0.0

            # logging of the epoch
            if self.verboses >= 1:
                self.logger.info("Epoch %s / %s:" % (epoch + 1, self.epochs))
            
            # initialize progress bar of dataloader of this epoch
            pbar = tqdm(dataloader, desc="step loss : ??.????")
            
            for i, batch_data in enumerate(pbar):
                # iteration of the batch
                loss = self._iteratie(batch_data)

                # add step loss to steps_loss and epoch_loss
                loss_val = loss.item()
                steps_loss += loss_val
                epoch_loss += loss_val

                # set loss to the description of pbar
                pbar.set_description("step loss : %.4f" % (loss_val))

                # log for each y steps
                if global_step % self.log_step == 0:
                    if self.verboses >= 1:
                        self.logger.debug("step avg loss at step %d of epoch %d : %.4f" % (i, epoch, steps_loss / self.log_step))
                    if self.verboses >= 2:
                        self.writer.add_scalar("training/steps_avg_loss", steps_loss / self.log_step, global_step=global_step)    
                    steps_loss = 0.0

                global_step += 1

            # log for each epoch
            if self.verboses >= 1:
                self.logger.info("epoch avg loss : %.4f" % (epoch_loss / num_batch))
            
            if self.verboses >= 2:
                self.writer.add_scalar("training/epoch_avg_loss", epoch_loss / num_batch, global_step=epoch)
    
    def save(self, save_path: str, file_name: str):
        # make directory to save model if the directory is not exist
        Path(save_path).mkdir(parents=True, exist_ok=True)

        # save jit module if use_jit is True
        if self.use_jit:
            save_file = path.join(save_path, "%s.pt" % (file_name))
            torch.jit.save(self.sequential, save_file)
        # else, save module in the usual way
        else:
            save_file = path.join(save_path, "%s.tar" % (file_name))
            torch.save(self.sequential.state_dict(), save_path)
    
    def load(self, load_path: str, file_name: str):
        # load jit module if use_jit is True
        if self.use_jit:
            load_file = path.join(load_path, "%s.pt" % (file_name))
            self.sequential = torch.jit.load(load_file)
        # else, load module in the usual way
        else:
            load_file = path.join(load_path, "%s.tar" % (file_name))
            self.sequential.load_state_dict(torch.load(load_file))
    

from .ctr import *
from .emb import *
from .ltr import *
