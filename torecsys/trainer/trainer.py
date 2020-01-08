from logging import Logger
from os import path
from pathlib import Path
from texttable import Texttable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torecsys.inputs.base import _Inputs
from torecsys.layers.regularization import Regularizer
from torecsys.models import _Model
from torecsys.models.sequential import Sequential
from torecsys.utils.logging import TqdmHandler
from typing import Callable, Dict
import warnings

# ignore import warnings of the below packages
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from torch.utils.tensorboard import SummaryWriter
    from tqdm.autonotebook import tqdm

class Trainer(object):
    r"""Trainer object to train model, including trs.inputs.inputs_wrapper and trs.model.
    """
    def __init__(self,
                 inputs_wrapper : _Inputs,
                 model          : _Model,
                 labels_name    : str,
                 regularizer    : callable = Regularizer(0.1, 2),
                 loss           : callable = nn.MSELoss(),
                 optimizer      : type = optim.AdamW,
                 epochs         : int = 10,
                 verboses       : int = 2,
                 log_step       : int = 500,
                 log_dir        : str = "logdir", 
                 use_jit        : bool = False,
                 **kwargs):
        r"""Initialize Trainer.
        
        Args:
            inputs_wrapper (_Inputs): Input object. 
                Required: outputs' fields = model's input.
            model (_Model): Model object. 
            labels_name (str, optional): Label's name (i.e. target variable).
            regularizer (callable, optional): Regularization function. 
                Defaults to Regularizer(0.1, 2).
            loss (callable, optional): Loss function. 
                Defaults to nn.MSELoss.
            optimizer (type, optional): Optimization function. 
                Defaults to optim.AdamW.
            epochs (int, optional): Number of training epochs. 
                Defaults to 10.
            verboses (int, optional): Logging's mode, 
                where 0 = slient, 1 = progress bar, 2 = tensorboard.
                Defaults to 2.
            log_step (int, optional): Number of global steps for each log. 
                Defaults to 500.
            log_dir (str, optional): Directory to store the log of tensorboard. 
                Defaults to "logdir".
            use_jit (bool, optional): [In development] Boolean flag to enable torch.jit.trace.
                Defaults to False.
        
        Arguments:
            example_inputs (Dict[str, T]): Example inputs for jit.trace to trace the 
                sequential.
        
        Attributes:
            sequential (Union[nn.modules, jit.TopLevelTracedModule]): Sequential of inputs wrapper 
                and model, which can do the forward calculation directly with the batch inputs.
            labels_name (str): Label's name (i.e. target variable).
            regularizer (callable): Regularization function. 
            loss (callable): Loss function. 
            parameters (List[nn.Paramter]): List of trainable tensors of parameters.
            optimizer (class): Object to optimize model.
            epochs (int): Number of training epochs.
            verboses (int): Mode of logging.
            log_step (int): Number of global steps for each log.
            use_jit (bool): Flag to show whether jit.trace is applyed to the sequential or not.
            num_params (int): Total number of trainable parameters in the sequential.
            logger (class): Object of logging.Logger to log the process.
            log_dir (str):  Directory to store the log of tensorboard.
            writer (class): Object of tensorboard.writer.SummaryWriter to log the process in 
                tensorboard.
        """
        # initialize sequential by inputs wrapper and mdel
        if use_jit:
            warnings.warn("Using torch.jit.trace is in experimental in this package. There will be errors from torch.jit.")
            self.sequential = torch.jit.trace(Sequential(inputs_wrapper=inputs_wrapper, model=model), kwargs.get("example_inputs"))
        else:
            self.sequential = Sequential(inputs_wrapper=inputs_wrapper, model=model)

        # set regularizer, loss, optimizer and other things
        self.labels_name = labels_name
        self.regularizer = regularizer
        self.loss = loss
        self.parameters = list(self.sequential.parameters())
        self.optimizer = optimizer(self.parameters)
        self.epochs = epochs
        self.verboses = verboses
        self.log_step = log_step
        self.use_jit = use_jit

        # count number of parameters in model
        self.num_params = sum(p.numel() for p in self.parameters if p.requires_grad)

        # streaming log in tqdm will be initialized 
        if verboses >= 1:
            # initialize logger of trainer
            self.logger = Logger("trainer")

            # set logger config, including level and handler
            self.logger.setLevel("DEBUG")
            handler = TqdmHandler()
            self.logger.addHandler(handler)

            self.logger.info("logger has been initialized.")
        
        # log in tensorboard will be initialized 
        if verboses >= 2:
            # store the path of log dir
            self.log_dir = path.join(path.dirname(__file__), log_dir)

            # create the folder if log_dir is not exist 
            Path(log_dir).mkdir(parents=True, exist_ok=True)

            # intitialize tensorboard summary writer with the given log_dir
            self.writer = SummaryWriter(log_dir=log_dir)

            # print the summary writer's location
            self.logger.info("tensorboard summary writter has been initialized and the log directory is set to %s." % (self.log_dir))

        print(self._describe())
    
    def _to_jit(self, example_inputs: Dict[str, torch.Tensor]):
        r"""Trace sequential (i.e. stack of input wrappers and model) into torch.jit module for \
        a better computation performance.
        
        Args:
            example_inputs (Dict[str, T]): Example of batch inputs passed to model, where key = names of input field and value 
                = tensors of input field
        """
        warnings.warn("Using torch.jit.trace is in experimental in this package. There will be errors from torch.jit.")
        self.use_jit = True
        self.sequential = torch.jit.trace(self.sequential, example_inputs)
    
    def _add_embedding(self,
                       param_name  : str, 
                       metadata    : list = None,
                       label_img   : torch.Tensor = None,
                       global_step : int = None,
                       tag         : str = None):
        r"""Add embedding to summary writer of tensorboard.
        
        Args:
            param_name (str): Name of paramter in the sequential.
            metadata (list, optional): A list of labels, each element will be convert to string. 
                Defaults to None.
            label_img (T, optional): Tensors of images correspond to each data point. 
                Defaults to None.
            global_step (int, optional): Global step value to record. 
                Defaults to None.
            tag (str, optional): Name for the embedding. 
                Defaults to None.
        
        Raises:
            ValueError: when parameter of the given name cannot be found.
        """
        # get embedding from inputs_wrapper
        param_dict = dict(self.sequential.inputs_wrapper.named_parameters())
        embed_mat = param_dict.get(param_name)

        if embed_mat is not None:
            if self.verboses >= 2:
                tag = "%s.step.%d" % (param_name, global_step) if tag is None else tag
                self.writer.add_embedding(embed_mat, metadata=metadata, label_img=label_img, global_step=global_step, tag=tag)
            else:
                self.logger.warn("_add_embedding only can be called when self.verboses >= 2.")
        else:
            raise ValueError("parameter %s cannot found." % param_name)

    def _add_graph(self, 
                   example_inputs : Dict[str, torch.Tensor], 
                   verbose        : bool = False):
        r"""Add graph data to summary.
        
        Args:
            example_inputs (Dict[str, T]): Example inputs, which is a dictionary of tensors feed to inputs wrapper.
            verboses (bool, optional): Whether to print graph structure in console. Defaults to True.
        """
        raise ValueError("_add_graph is not work well now. \nFor reference: https://github.com/lanpa/tensorboardX/issues/483.")
        if self.verboses >= 2:
            self.writer.add_graph(self.sequential, example_inputs, verbose=verbose)
        else:
            if self.verboses >= 1:
                self.logger.warn("_add_graph only can be called when self.verboses >= 2.")
            else:
                warnings.warn("_add_graph only can be called when self.verboses >= 2.")
        
    def _describe(self):
        r"""Show summary of trainer
        """
        # getattr from self 
        inputs_name = self.sequential.inputs_wrapper.__class__.__name__ if getattr(self, "sequential", None) is not None else None
        model_name = self.sequential.model.__class__.__name__ if getattr(self, "sequential", None) is not None  else None
        loss_name = self.loss.__class__.__name__ if getattr(self, "loss", None) is not None  else None
        optim_name = self.optimizer.__class__.__name__ if getattr(self, "optimizer", None) is not None  else None
        regul_norm = self.regularizer.norm if getattr(self, "regularizer", None) is not None  else None
        regul_lambda = self.regularizer.weight_decay if getattr(self, "regularizer", None) is not None  else None
        epochs = self.epochs if getattr(self, "epochs", None) is not None  else None
        logdir = self.log_dir if getattr(self, "log_dir", None) is not None  else None
        
        # initialize _vars of parameters 
        _vars = {
            "inputs"        : inputs_name,
            "model"         : model_name,
            "loss"          : loss_name,
            "optimizer"     : optim_name,
            "reg norm"      : regul_norm,
            "reg lambda"    : regul_lambda,
            "num of epochs" : epochs,
            "log directory" : logdir
        }
        
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
        
    def _iterate(self, batch_inputs: Dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        r"""Iteration for each batch of inputs.
        
        Args:
            batch_inputs (Dict[str, T]): Dictionary of tensors, where its keys are the name of inputs' \
                fields in inputs wrapper, and its values are tensors of those fields.
            labels (T): Tensors of the targeted values.
        
        Returns:
            T: Loss of the model.
        """
        # zero the parameter gradients
        self.optimizer.zero_grad()
        
        # calculate forward prediction
        outputs = self.sequential(batch_inputs)

        # calculate loss and regularized loss
        loss = self.loss(outputs, labels)
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
                of batch of inputs and labels
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

            for i, batch_data in enumerate(pbar):
                # iteration of the batch
                labels = batch_data.pop(self.labels_name)
                loss = self._iterate(batch_data, labels)
                
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

    def evalaute(self, 
                 dataloader  : torch.utils.data.DataLoader,
                 evaluate_fn : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> float:
        r"""evaulate trained model's performance with a dataset in dataloader, by giving evaluate_fn
        
        Args:
            dataloader (torch.utils.data.DataLoader): dataloader of dataset for evaluation
            evaluate_fn (Callable[[T, T], T]): function of evaluation by inputing true values and predicted values
        
        Returns:
            float: scores of evaulation
        """
        # initialize scores to save the aggregation
        scores = 0.0
        for i, batch_data in enumerate(dataloader):
            # pop labels from batch data
            labels = batch_data.pop(self.labels_name)
            
            # calculate forward calculation
            yhat = self.predict(batch_data)
            
            # calculate evaulate scores, where score's shape = (1, )
            score = evaluate_fn(yhat, labels)

            # add score to scores
            scores += score.item()

        avg_scores = float(scores / i)
        return avg_scores
    
    def predict(self, batch_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        r"""Get prediction from the model
        
        Args:
            batch_inputs (Dict[str, T]): batch inputs of model, where key = names of input field and value = tensors of input field
        
        Returns:
            T, shape = (batch size, 1), dtype = torch.float: prediction of the model
        """
        # set no gradient to the computation
        with torch.no_grad():
            # calculate forward calculation
            outputs = self.sequential(batch_inputs)
        return outputs
    
    def summary(self):
        r"""Summary model to a report
        """
        return
    
    def save(self, save_path: str, file_name: str):
        r"""Save the state dict of model.
        
        Args:
            save_path (str): Directory to save the model.
            file_name (str): Name of the saved model.
        """
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
        r"""Load the state dict of model.
        
        Args:
            load_path (str): Directory to load the model.
            file_name (str): Name of the model to be loaded.
        """
        # load jit module if use_jit is True
        if self.use_jit:
            load_file = path.join(load_path, "%s.pt" % (file_name))
            self.sequential = torch.jit.load(load_file)
        # else, load module in the usual way
        else:
            load_file = path.join(load_path, "%s.tar" % (file_name))
            self.sequential.load_state_dict(torch.load(load_file))

    def to(self, device: str):
        """Set all attributes of torch to the given device.
        
        Args:
            device (str): [description]
        
        Raises:
            NotImplementedError: not yet implemented.
        """
        raise NotImplementedError("to be implemented.")
