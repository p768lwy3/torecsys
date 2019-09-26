from .trainer import Trainer
from torecsys.data.negsampling import _NegativeSampler
from torecsys.functional.regularization import Regularizer
from torecsys.inputs.base import _Inputs
from torecsys.losses.ltr import _RankingLoss
from torecsys.losses.ltr.pairwise_ranking_loss import BayesianPersonalizedRankingLoss
from torecsys.models import _Model
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from typing import Dict
import warnings

# ignore import warnings of the below packages
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from tqdm.autonotebook import tqdm

class RankingTrainer(Trainer):
    r"""Object for training a sequential of transformation and embedding of inputs and model of
    click through rate prediction with a negative sampler (i.e. in a pairwise way).
    """
    def __init__(self,
                 inputs_wrapper : _Inputs,
                 model          : _Model,
                 neg_sampler    : _NegativeSampler,
                 neg_number     : int,
                 regularizer    : Regularizer = Regularizer(0.1, 2),
                 loss           : _RankingLoss = BayesianPersonalizedRankingLoss(),
                 optimizer      : type = optim.AdamW,
                 epochs         : int = 10,
                 verboses       : int = 2,
                 log_step       : int = 500,
                 log_dir        : str = "logdir", 
                 use_jit        : bool = False,
                 **kwargs):
        r"""Initialize ranking trainer object.
        
        Args:
            inputs_wrapper (_Inputs): Defined object of trs.inputs.InputWrapper, where its outputs' \
                fields should be equal to the model forward's arguments.
            model (_Model): Object of nn.Module, which is allowed to calculate foward propagation 
                and gradients.
            neg_sampler (_NegativeSampler): Object of trs.data.negsampling._NegativeSampler to 
                generate negative samples to train the given model in pairwise ranking way, with 
                a callable function `generate`, which return Dict[str, T] by inputing pos_samples 
                which is a Dict[str, T] of positive samples.
            neg_number (int): An integer of number of negative samples to be drawn.
            regularizer (Regularizer, optional): Callable to calculate regularization. 
                Defaults to Regularizer(0.1, 2).
            loss (type, optional): Callable to calculate loss. 
                Defaults to nn.MSELoss.
            optimizer (type, optional):  Object to optimize the model. 
                Defaults to optim.AdamW.
            epochs (int, optional): Number of training epochs. 
                Defaults to 10.
            verboses (int, optional): Mode of logging. 0 = slient, 1 = progress bar, 2 = tensorboard.
                Defaults to 2.
            log_step (int, optional): Number of global steps for each log. 
                Defaults to 500.
            log_dir (str, optional): Directory to store the log of tensorboard. 
                Defaults to "logdir".
            use_jit (bool, optional): Whether jit.trace is applyed to the sequential or not.
                In experimental.
                Defaults to False.
        Kawrgs:
            example_inputs (Dict[str, T]): Example inputs for jit.trace to trace the sequential.
        
        Attributes:
            sequential (Union[nn.modules, jit.TopLevelTracedModule]): Sequential of inputs wrapper 
                and model, which can do the forward calculation directly with the batch inputs.
            neg_sampler (callable): Function to generate negative samples to train the given model \
                in pairwise ranking way.
            regularizer (callable): Callable to calculate regularization.
            loss (callable): Callable to calculate loss.
            parameters (List[nn.Paramter]): List of trainable tensors of parameters.
            optimizer (class): Object to optimize model.
            epochs (int): Number of training epochs.
            verboses (int): Mode of logging.
            log_step (int): Number of global steps for each log.
            use_jit (bool): Flag to show whether jit.trace is applyed to the sequential or not.
            num_params (int): Total number of trainable parameters in the sequential.
            logger (class): Object of logging.Logger to log the process.
            log_dir (str):  Directory to store the log of tensorboard.
            writer (class): Object of tensorboard.writer.SummaryWriter to log the process in tensorboard.
        """
        # refer to parent class
        super(RankingTrainer, self).__init__(
            inputs_wrapper = inputs_wrapper,
            model          = model,
            regularizer    = regularizer,
            labels_name    = None,
            loss           = loss,
            optimizer      = optimizer,
            epochs         = epochs,
            verboses       = verboses,
            log_step       = log_step,
            log_dir        = log_dir,
            use_jit        = use_jit,
            **kwargs
        )

        # bind neg_sampler to neg_sampler
        self.neg_sampler = neg_sampler
        self.neg_number = neg_number

    def _iterate(self, 
                 pos_inputs: Dict[str, torch.Tensor], 
                 neg_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        r"""Iteration for each batch of inputs.
        
        Args:
            pos_inputs (Dict[str, T]): Dictionary of tensors of positive samples, where its keys 
                are the name of inputs' fields in inputs wrapper, and its values are tensors of 
                those fields, with shape = (B, N, E).
            neg_inputs (Dict[str, T]): Dictionary of tensors of negative samples, where its keys 
                are the name of inputs' fields in inputs wrapper, and its values are tensors of 
                those fields, with shape = (B * Nneg, N, E).
        
        Returns:
            T: Loss of the model.
        """
        # zero the parameter gradients
        self.optimizer.zero_grad()

        # cat pos_inputs and neg_inputs into a dictionary of tensors - batch_inputs
        batch_inputs = dict()
        for inp_name in pos_inputs:
            # set batch size and number of negative samples for reshaping
            batch_size = pos_inputs[inp_name].size(0)
            num_samples = int(neg_inputs[inp_name].size(0) / batch_size)

            # reshape tensors and concat them into one tensor
            num_fields = neg_inputs[inp_name].size(1)
            pos_tensors = pos_inputs[inp_name].view(batch_size, 1, num_fields)
            neg_tensors = neg_inputs[inp_name].view(batch_size, num_samples, num_fields)
            batch_inputs[inp_name] = torch.cat((pos_tensors, neg_tensors), dim=1)
            batch_inputs[inp_name] = batch_inputs[inp_name].view(batch_size * (1 + num_samples), num_fields)
        
        # calculate forward prediction, where outputs' shape = (B * (1 + Nneg), 1)
        outputs = self.sequential(batch_inputs)

        # split outputs into pos_outputs and neg_outputs
        pos_outputs, neg_outputs = outputs.view(batch_size, -1, 1).split(
            (1, num_samples), dim=1)
        pos_outputs = pos_outputs.squeeze()
        neg_outputs = neg_outputs.squeeze()

        print(pos_outputs)
        print(neg_outputs)
        
        # calculate loss and regularized loss
        loss = self.loss(pos_outputs, neg_outputs)
        if self.regularizer is not None:
            named_params = list(self.sequential.named_parameters())
            reg_loss = self.regularizer(named_params)
            loss += reg_loss

        # calculate backward and optimize 
        loss.backward()
        self.optimizer.step()

        # return loss to log stream and tensorboard
        return loss
    
    def fit(self, dataloader: torch.utils.data.DataLoader):
        r"""Callable to fit the model with a dataloader
        
        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader, where its iteration is a dictionary 
                of batch of positive samples and negative samples
        """
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

            for i, pos_inputs in enumerate(pbar):
                # generate negative samples
                neg_inputs = self.neg_sampler(pos_inputs, self.neg_number)
                
                # iteration of the batch
                loss = self._iterate(pos_inputs, neg_inputs)
                
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

    def evalaute(self):
        return
        