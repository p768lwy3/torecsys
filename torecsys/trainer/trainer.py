from logging import Logger
from os import path
from pathlib import Path
from texttable import Texttable
import torch
import torch.nn as nn
import torch.nn.modules.loss as nn_loss
import torch.nn.parallel as nn_parallel
import torch.optim as optim
import torch.utils.data
from torecsys.data.negsampling import _NegativeSampler
from torecsys.inputs.base import _Inputs
from torecsys.layers.regularization import Regularizer
import torecsys.models as trs_model
from torecsys.models.sequential import Sequential
from torecsys.utils.logging import TqdmHandler
from typing import Callable, Dict, List, Tuple, TypeVar, Union
import warnings

# ignore import warnings of the below packages
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from torch.utils.tensorboard import SummaryWriter
    from tqdm.autonotebook import tqdm

obj_dict = {
    "ctr": "clickthroughrate",
    "emb": "embedding",
    "ltr": "learningtorank"
}

class DevTrainer(object):
    r"""Trainer object to train model, including trs.inputs.inputs_wrapper and trs.model.
    """
    def __init__(self):
        r"""Initialize Trainer.
        """
        # Initialize private variables
        # Model Core
        self._objective = "clickthroughrate"
        self._inputs = None
        self._model  = None
        self._sequential = None

        self._regularizer = None
        self._criterion = None
        self._optimizer = None

        # GPU-related
        self._use_cuda = False
        self._devices = None
        self._base_device_ordinal = None
        self._dtype = "float"

        # JIT-related
        self._use_jit = False

        # Dataloader-related
        self._loaders = {}
        self._loader_iters = {}
        self._loader_specs = {}

        # Targets field name
        self._targets_name = None

        # Negative-sampler
        self._negative_sampler = None
        self._negative_size = None

        # Mode
        self._current_mode = "train"

        # Logging related
        self._logger = None
        self._last_logged = {}
        self._log_directory = {}
        self._num_log_step = None

        # Iteration and epoch
        self._global_step_cnt = 0
        self._iteration_cnt = 0
        self._epoch_cnt = 0
        self._max_num_epochs = 0
        self._max_num_iterations = None

        # TODO
        # Validation
        # Metrics
        # Checkpoint
        # Early Stopping
        # Callback
    
    @property
    def objective(self) -> str:
        r"""Get the objective of trainer.
        
        Returns:
            str: objective of the trainer.
        """
        return self._objective

    @property
    def inputs(self) -> _Inputs:
        r"""Get the inputs.
        
        Returns:
            _Inputs: Inputs of the trainer.
        """
        return self._inputs
    
    @property
    def model(self) -> nn.Module:
        r"""Get the model.
        
        Returns:
            torch.nn.Module: Model of the trainer.
        """
        return self._model
    
    @property
    def sequential(self) -> nn.Sequential:
        r"""Get the sequential.
        
        Returns:
            torch.nn.Sequential: Sequential of the trainer.
        """
        return self._sequential
    
    @property
    def regularizer(self) -> Regularizer:
        r"""Get the regularizer.
        
        Returns:
            torecsys.layers.regularization.Regularizer: Regularizer of the trainer.
        """
        return self._regularizer
    
    @property
    def criterion(self) -> nn_loss._Loss:
        r"""Get the criterion.
        
        Returns:
            torch.nn.modules.loss._Loss: Criterion of the trainer.
        """
        return self._criterion
    
    @property
    def optimizer(self) -> optim:
        r"""Get the optimizer
        
        Returns:
            torch.optim: Optimizer of the trainer.
        """
        return self._optimizer
    
    @property
    def train_loader(self) -> torch.utils.data.DataLoader:
        r"""Get the data loader for training.
        
        Raises:
            AssertionError: when data loader for training is not found.
        
        Returns:
            torch.utils.data.DataLoader: Data loader for training of the trainer.
        """
        if self._loaders.get("train") is None:
            raise AssertionError("Data loader for training not setted.")
        return self._loaders.get("train")
    
    @property
    def validate_loader(self) -> torch.utils.data.DataLoader:
        r"""Get the data loader for validation.
        
        Raises:
            AssertionError: when data loader for validation is not found.
        
        Returns:
            torch.utils.data.DataLoader: Data loader for validation of the trainer.
        """

        if self._loaders.get("validate") is None:
            raise AssertionError("Data loader for validation not setted.")
        return self._loaders.get("validate")
    
    @property
    def targets_name(self) -> str:
        r"""Get the target field name.
        
        Returns:
            str: Name of target field.
        """
        return self._targets_name
    
    @property
    def negative_sampler(self) -> _NegativeSampler:
        r"""Get the negative sampler used for negative sampling in learning-to-rank.
        
        Returns:
            _NegativeSampler: Negative sampler of the trainer.
        """
        return self._negative_sampler
    
    @property
    def negative_size(self) -> int:
        r"""Get the size of sampling to generate negative samples.
        
        Returns:
            int: Size of sampling.
        """
        return self._negative_size
    
    @property
    def dtype(self) -> str:
        r"""Get the data type of model.
        
        Returns:
            str: Data type of model.
        """
        return self._dtype
    
    @property
    def logger(self) -> Logger:
        r"""Get the logger.
        
        Returns:
            logging.Logger: Logger of the trainer
        """
        return self._logger
    
    @property
    def log_directory(self) -> str:
        r"""Get the logging directory.
        
        Returns:
            str: Logging directory of the trainer.
        """
        return self._log_directory
    
    @property
    def max_num_epochs(self) -> int:
        r"""Get the maximum number of training epochs.
        
        Returns:
            int: Maximum number of training epochs.
        """
        return self._max_num_epochs
    
    @property
    def max_num_iterations(self) -> int:
        r"""Get the maximum number of training iterations.
        
        Returns:
            int: Maximum number of training iterations.
        """
        return self._max_num_iterations

    @property
    def has_inputs(self) -> bool:
        r"""Return whether inputs is binded to the trainer.
        
        Returns:
            bool: True if inputs is binded else False.
        """
        return self._inputs is not None
    
    @property
    def has_model(self) -> bool:
        r"""Return whether model is binded to the trainer.
        
        Returns:
            bool: True if model is binded else False.
        """
        return self._model is not None
    
    @property
    def has_regularizer(self) -> bool:
        r"""Return whether regularizer is binded to the trainer.
        
        Returns:
            bool: True if regularizer is binded else False.
        """
        return self._regularizer is not None
    
    @property
    def has_criterion(self) -> bool:
        r"""Return whether criterion is binded to the trainer.
        
        Returns:
            bool: True if criterion is binded else False.
        """
        return self._criterion is not None

    @property
    def has_optimizer(self) -> optim.Optimizer:
        r"""Return optimizer of the trainer if trainer._optimizer is not None.

        Raises:
            ValueError: when optimizer is not bined to the trainer.
        
        Returns:
            torch.optim.Optimizer: optimizer of the trainer.
        """
        if self._optimizer is None:
            raise ValueError("optimizer is not set yet.")
        return self._optimizer
    
    @property
    def has_regularizer(self) -> bool:
        r"""Return whether regularizer is binded to the trainer.
        
        Returns:
            bool: True if regularizer is binded else False.
        """
        return self._regularizer is not None
    
    @property
    def has_max_num_epochs(self) -> int:
        """Return max_num_epochs setted to the trainer.
        
        Raises:
            ValueError: when max_num_epochs is not setted to the trainer.
        
        Returns:
            int: max_num_epochs of the trainer.
        """
        if self._max_num_epochs is None:
            raise ValueError("max_num_epochs is not set yet.")
        return self._max_num_epochs
    
    @property
    def has_max_num_iterations(self) -> bool:
        r"""Return whether max_num_iterations is setted to the trainer.
        
        Returns:
            bool: True if max_num_iterations is setted else False.
        """
        return self._max_num_iterations is not None
    
    def is_cuda(self) -> bool:
        r"""Return whether using GPU for training.
        
        Returns:
            bool: True if cuda is enable else False.
        """
        return self._use_cuda
    
    def is_jit(self) -> bool:
        r"""Return whether using jit for training.
        
        Returns:
            bool: True if jit is enabled else False.
        """
        return self._use_jit
    
    @objective.setter
    def objective(self, objective: str):
        r"""Set objective to the trainer.
        
        Args:
            objective (str): objective to be setted to the trainer.
        """
        self.bind_objective(objective)
    
    @inputs.setter
    def inputs(self, inputs: _Inputs):
        r"""Set Inputs to the trainer.
        
        Args:
            inputs (torecsys.inputs.base._Inputs): Inputs to be setted to the trainer.
        """
        # Bind inputs to _inputs
        self.bind_inputs(inputs)
    
    @model.setter
    def model(self, model: nn.Module):
        r"""Set model to the trainer.
        
        Args:
            model (torch.nn.Module): Model to be setted to the trainer.
        """
        # Bind model to _model
        self.build_model(model)
    
    @regularizer.setter
    def regularizer(self, regularizer: Regularizer):
        r"""Set regularizer to the trainer.
        
        Args:
            regularizer (torecsys.layers.regularization.Regularizer): Regularizer to be setted to the trainer.
        """
        # Bind regularizer to _regularizer
        self.build_regularizer(regularizer)
    
    @criterion.setter
    def criterion(self, value: Union[str, Callable[[torch.Tensor], torch.Tensor], dict]):
        r"""Set loss to the trainer.
        
        Args:
            value (Union[str, Callable[[T], T], dict]): value to build criterion.

        Raises:
            TypeError: when type of value is not allowed.
        """
        # Bind criterion to _criterion
        if isinstance(value, str) or callable(value):
            self.build_criterion(value)
        elif isinstance(value, dict):
            self.build_criterion(value)
        else:
            raise TypeError(f"{type(value).__name__} not allowed.")
    
    @optimizer.setter
    def optimizer(self, value: Union[str, Callable[[torch.Tensor], torch.Tensor], dict]):
        r"""Set optimizer to the trainer.
        
        Args:
            value (Union[str, Callable[[T], T], dict]): value to build optimizer.
        
        Raises:
            TypeError: when type of value is not allowed.
        """
        # Bind optimizer to _optimizer
        if isinstance(value, str) or callable(value):
            self.build_optimizer(value)
        elif isinstance(value, dict):
            self.build_optimizer(**value)
        else:
            raise TypeError(f"{type(value).__name__} not allowed.")
    
    @train_loader.setter
    def train_loader(self, dataloader: torch.utils.data.DataLoader):
        r"""Set data loader for training to the trainer.
        
        Args:
            dataloader (torch.utils.data.DataLoader): data loader for training of trainer.
        
        Raises:
            TypeError: when type of dataloader is not allowed
        """
        if isinstance(dataloader, torch.utils.data.DataLoader):
            raise TypeError("dataloader must be a torch.utils.data.DataLoader object.")
        self._loaders.update({"train" : dataloader})
    
    @validate_loader.setter
    def validate_loader(self, dataloader: torch.utils.data.DataLoader):
        r"""Set data loader for training to the trainer.
        
        Args:
            dataloader (torch.utils.data.DataLoader): data loader for validation of trainer.
        
        Raises:
            TypeError: when type of dataloader is not allowed
        """
        if isinstance(dataloader, torch.utils.data.DataLoader):
            raise TypeError("dataloader must be a torch.utils.data.DataLoader object.")
        self._loaders.update({"validate" : dataloader})
    
    @targets_name.setter
    def targets_name(self, targets_name: str):
        """Set targets_name of the trainer.
        
        Args:
            targets_name (str): targets name to be setted for getting targets field in batch.
        """
        self.set_targets_name(targets_name)
    
    @dtype.setter
    def dtype(self, dtype: str):
        r"""Set dtype of the trainer.
        
        Args:
            dtype (str): data type of trainer to be setted.
        """
        # Bind dtype to _dtype
        self.set_dtype(dtype)
    
    @max_num_epochs.setter
    def max_num_epochs(self, max_num_epochs: int):
        r"""Set maximum number of training epochs to the trainer
        
        Args:
            max_num_epochs (int): maximum number of training epochs.
        """
        self.set_max_num_epochs(max_num_epochs)
    
    @max_num_iterations.setter
    def max_num_iterations(self, max_num_iterations: int):
        r"""Set maximum number of training iterations to the trainer
        
        Args:
            max_num_iterations (int): maximum number of training iterations.
        """
        self.set_max_num_iterations(max_num_iterations)
    
    def bind_objective(self, objective: str) -> TypeVar("Trainer"):

        if not isinstance(objective, str):
            raise TypeError("objective must be a str.")
        
        objective = objective.lower()
        objective = obj_dict.get(objective) if objective in obj_dict else objective
        
        if objective not in ["clickthroughrate", "embedding", "learningtorank"]:
            raise AssertionError(f"{objective} not allowed.")
        
        self._objective = objective

        return self
    
    def bind_inputs(self, inputs: _Inputs) -> TypeVar("Trainer"):
        r"""Bind inputs to the trainer.inputs
        
        Args:
            inputs (torecsys.inputs.base._Inputs): Inputs to be binded to the trainer.inputs.
        
        Raises:
            TypeError: whether type of inputs is not allowed to be setted.
        
        Returns:
            torecsys.trainer.Trainer: self
        """
        # Check if the type of inputs is allowed
        if not isinstance(inputs, _Inputs):
            raise TypeError("inputs must be a torecsys.inputs.base._Inputs object.")
        
        # Bind inputs to _inputs
        self._inputs = inputs

        # Apply cuda to inputs if is_cuda is True
        if self.is_cuda():
            self._inputs.cuda()

        return self

    def build_model(self, 
                    method: Union[str, Callable[[torch.Tensor], torch.Tensor], nn.Module],
                    **kwargs) -> TypeVar("Trainer"):
        r"""Build model of the trainer
        
        Args:
            method (Union[str, Callable[[torch.Tensor], torch.Tensor], nn.Module]): 
                Model to be binded to the trainer._model.
        
        Raises:
            TypeError: whether type of model is not allowed to be setted.
        
        Returns:
            torecsys.trainer.Trainer: self
        """
        # Check if the type of model is allowed
        if isinstance(method, str):
            model_method = getattr(trs_model, method, None)
            if model_method is None:
                raise AssertionError(f"{method} not found.")
        elif callable(method) and isinstance(method, type):
            model_method = method
        elif not isinstance(method, nn.Module):
            self._model = method
            return self
        else:
            raise TypeError("{type(method).__name__} not allowed.")
        
        # Bind model to _model
        self._model = model_method(**kwargs)

        # Apply cuda to model if is_cuda is True
        if self.is_cuda():
            self._model.cuda()

        return self
    
    def build_regularizer(self, 
                          regularizer: Regularizer = None,
                          **kwargs) -> TypeVar("Trainer"):
        r"""Build regularizer of the trainer
        
        Args:
            regularizer (torecsys.layers.regularization, optional): \
                regularizer to be binded to the trainer._regularizer.
        
        Raises:
            TypeError: whether type of regularizer is not allowed to be setted.
        
        Returns:
            torecsys.trainer.Trainer: self
        """
        # Check if the type of regularizer is allowed
        if regularizer is None:
            regularizer = Regularizer(**kwargs)
        else:
            if isinstance(model, Regularizer):
                pass
            else:
                raise TypeError("regularizer must be a torecsys.layers.regularization.Regularizer.")
        
        # Bind regularizer to _regularizer
        self._regularizer = regularizer
        
        # Apply cuda to regularizer if is_cuda is True
        if self.is_cuda():
            self._regularizer.cuda()
        
        return self
    
    def build_sequential(self, 
                         inputs : _Inputs = None,
                         model  : nn.Module = None,
                         **kwargs) -> TypeVar("Trainer"):
        r"""Build sequential with inputs and model.

        Args:
            inputs (_Inputs): Inputs object.
                Required: output fields' names = model inputs' names.
            model (_Model): Model object.
        
        Raises:
            AssertionError: whether inputs or model is not found.
        
        Returns:
            torecsys.trainer.Trainer: self
        """
        # Set new values to private variables
        if inputs is not None:
            self._inputs = inputs

        if model is not None:
            self._model = model
        
        # Check if private variables _inputs or _model is None
        if not self.has_inputs or not self.has_model:
            raise AssertionError("inputs or model not found")
        
        # Build sequential with _inputs and model
        self._sequential = Sequential(inputs_wrapper=self._inputs, model=self._model)

        # Enable GPU process if is_cuda is True
        if self.is_cuda():
            self.cuda()
        
        # Enable jit process if is_jit is True
        if self.is_jit():
            self._to_jit(_sequentail, kwargs.get("sample_inputs"))

        return self
    
    def build_criterion(self,
                        method  : Union[str, Callable[[torch.Tensor], torch.Tensor], optim.Optimizer],
                        **kwargs) -> TypeVar("Trainer"):
        r"""Build criterion.
        
        Args:
            method (Union[str, Callable[[torch.Tensor], torch.Tensor], optim.Optimizer]): method of criterion.
        
        Raises:
            AssertionError: when criterion_method is not found.
            TypeError: when type of criterion_method is not allowed.
        
        Returns:
            torecsys.trainer.Trainer: self
        """
        # criterion to _criterion 
        if isinstance(method, str):
            criterion_method = getattr(nn_loss, method, None)
            if criterion_method is None:
                raise AssertionError(f"{criterion_method} not found.")
        elif callable(method) and isinstance(method, type):
            criterion_method = method
        elif isinstance(method, nn_loss._Loss):
            self._criterion = method
            return self
        else:
            raise TypeError(f"{type(method).__name__} not allowed.")
        
        self._criterion = criterion_method(**kwargs)

        # Enable GPU process if data parallelism is allowed
        if hasattr(self, "_base_device_ordinal"):
            base_device_ordinal = self._base_device_ordinal
        else:
            base_device_ordinal = None

        if self.is_cuda() and base_device_ordinal != -1:
            self._criterion.cuda()
        
        return self
    
    def build_optimizer(self, 
                        method     : Union[str, Callable[[torch.Tensor], torch.Tensor], optim.Optimizer], 
                        parameters : nn.parameter.Parameter = None, 
                        **kwargs) -> TypeVar("Trainer"):
        r"""Build optimizer.

        Args:
            method (Union[str, Callable[[torch.Tensor], torch.Tensor], optim.Optimizer]): method of optimizer.
            parameters (torch.nn.parameter.Parameter, optional): parameters to be optimized.
                Defaults to None.
        
        Raises:
            AssertionError: when optimizer_method is not found.
            TypeError: when type of optimizer_method is not allowed.
        
        Returns:
            torecsys.trainer.Trainer: self
        """
        if isinstance(method, str):
            optimizer_method = getattr(optim, method, None)
            if optimizer_method is None:
                raise AssertionError(f"{optimizer_method} not found.")
        elif callable(method) and isinstance(method, type):
            optimizer_method = method
        elif isinstance(method, optim.Optimizer):
            self._optimizer = method
            return self
        else:
            raise TypeError(f"{type(optimizer_method).__name__} not allowed.")
        
        # Get parameters from self._sequential and set them to self._optimizer
        parameters = self._sequential.parameters() if parameters is None else parameters
        self._optimizer = optimizer_method(parameters, **kwargs)
        return self
    
    def set_loader(self, 
                   name   : str, 
                   loader : torch.utils.data.DataLoader) -> TypeVar("Trainer"):
        r"""Set data loader to the trainer.
        
        Args:
            name (str): name of loader. allows: ["train", "validate", "test"]
            loader (torch.utils.data.DataLoader): dataloader object.
        
        Raises:
            AssertionError: when name is not in ["train", "validate", "test"].
            TypeError: when type of loader is not allowed.

        Returns:
            torecsys.trainer.Trainer: self
        """
        if name not in ["train", "validate", "test"]:
            raise AssertionError(f"name must be in [\"train\", \"validate\", \"test\"], got {name} instead.")
        
        if not isinstance(loader, torch.utils.data.DataLoader):
            raise TypeError(f"{type(loader).__name__} not allowed.")
        
        # Check if the dataloader is new
        is_new_loader = loader is not self._loaders.get(name)
        self._loaders.update({ name : loader })

        # Remove cached of DataLoaderIter
        if is_new_loader and name in self._loader_iters:
            del self._loader_iters[name]
        
        return self
    
    def set_targets_name(self, targets_name: str) -> TypeVar("Trainer"):
        r"""Set targets_name of the trainer.
        
        Args:
            targets_name (str): targets name to be setted for getting targets field in batch.
        
        Raises:
            TypeError: when type of targets_name is not allowed.
        
        Returns:
            torecsys.trainer.Trainer: self
        """
        if not isinstance(targets_name, str):
            raise TypeError(f"{type(targets_name).__name__} not allowed.")

        self._targets_name = targets_name
        return self
    
    def set_dtype(self, dtype: str) -> TypeVar("Trainer"):
        r"""Set data type of trainer.
        
        Args:
            dtype (str): data type of trainer to be setted.
        
        Raises:
            AssertionError: when dtype is not allowed.
        
        Returns:
            torecsys.trainer.Trainer: self
        """
        # Check if dtype is allowed
        if dtype not in ["double", "float", "half"]:
            raise AssertionError(f"{dtype} not found.")
        
        # Bind dtype to _dtype
        self._dtype = dtype

        # Call methods of _inputs and _model to applied data type changes
        self._inputs = getattr(self._inputs, dtype)()
        self._model = getattr(self._model, dtype)()

        return self
    
    def set_max_num_epochs(self, max_num_epochs: int) -> TypeVar("Trainer"):
        r"""Set maximum number of training epochs to the trainer
        
        Args:
            max_num_epochs (int): maximum number of training epochs.
        
        Raises:
            TypeError: when type of max_num_epochs is not allowed.
        
        Returns:
            torecsys.trainer.Trainer: self
        """
        if not isinstance(max_num_epochs, int):
            raise TypeError("max_num_epochs must be int.")
        
        self._max_num_epochs = max_num_epochs

        return self
    
    def set_max_num_iterations(self, max_num_iterations: int) -> TypeVar("Trainer"):
        r"""Set maximum number of training iterations to the trainer
        
        Args:
            max_num_iterations (int): maximum number of training iterations.
        
        Raises:
            TypeError: when type of max_num_iterations is not allowed.
        
        Returns:
            torecsys.trainer.Trainer: self
        """
        if not isinstance(max_num_iterations, int):
            raise TypeError("max_num_iterations must be int.")
        
        self._max_num_iterations = max_num_iterations

        return self
    
    def cast(self, objects: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]) \
        -> Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]:
        r"""Cast objects to the trainer's data type.
        
        Args:
            objects (Union[T, List[T], Tuple[torch.Tensor]]): object to be casted.
        
        Returns:
            Union[T, List[T], Tuple[torch.Tensor]]: object which is casted.
        """
        if isinstance(objects, (list, tuple)):
            # Cast on list of objects
            return type(objects)([self.cast(_object) for _object in objects])
        else:
            # Cast on object
            if objects.__class__.__name__ in ["HalfTensor", "FloatTensor", "DoubleTensor"]:
                cast_fn = getattr(objects, self._dtype, None)
            else:
                cast_fn = None
            
            if cast_fn is not None:
                return cast_fn()
            else:
                return objects
    
    def to(self, device: Union[str, torch.device]) -> TypeVar("Trainer"):
        r"""Set device of trainer.
        
        Args:
            device (str): device to be used for training.
        
        Raises:
            TypeError: when device is not allowed
        
        Returns:
            torecsys.trainer.Trainer: self
        """
        if device == "cuda":
            return self.cuda()
        elif device == "cpu":
            return self.cpu()
        elif isinstance(device, torch.device):
            self.to(device.type)
        else:
            raise TypeError(f"{type(device).__name__} not allowed.")
    
    def cuda(self, 
             devices      : Union[int, list, tuple] = None, 
             base_devices : str = None) -> TypeVar("Trainer"):
        r"""Enable GPU computation of trainer.

        Args:
            devices (Union[int, list, tuple]): Specify the devices to use for data parallel training.
            base_devices (str): when using data parallel training, specify where the results tensors are collected.
                If "cuda", the results are collected in `devices[0]`.
        
        Raises:
            ValueError: when base_device is not cpu or cuda.
            ValueError: when no data parallelism but base_device is cpu.
        
        Returns:
            torecsys.trainer.Trainer: self
        """
        # check if base_devices is allowed
        if base_devices not in [None, "cpu", "cuda"]:
            raise ValueError(f"base_device must either be \"cpu\" or \"cuda\", got {base_devices} instead.")
        
        if isinstance(devices, int) or (isinstance(devices, (list, tuple)) and len(devices) == 1):
            # no data-parallelism, make sure base_device is not CPU
            raise ValueError("base_device cannot be \"cpu\" if data parallelism is not allowed.")

        self._base_device_ordinal = {None: None, "cpu": -1, "cuda": None}.get(base_devices)

        # Enable GPU in inputs, model and criterion
        if self.has_inputs:
            self._inputs.cuda()
        
        if self.has_model:
            self._model.cuda()
        
        if self.has_criterion and self._base_device_ordinal != -1:
            self.criterion.cuda()
        elif self.has_criterion and self._base_device_ordinal == -1:
            self.criterion.cpu()

        self._use_cuda = True
        self._devices = devices

        return self
    
    def cpu(self) -> TypeVar("Trainer"):
        r"""Disable GPU computation of trainer.
        
        Returns:
            torecsys.trainer.Trainer: self
        """
        if self.has_inputs:
            self._inputs.cpu()

        if self.has_model:
            self._model.cpu()
        
        if self.has_criterion:
            self.criterion.cpu()
        
        self._use_cuda = False
        self._devices = None
        
        return self
    
    def train(self) -> TypeVar("Trainer"):
        r"""Set trainer to train mode.
        
        Returns:
            torecsys.trainer.Trainer: self
        """
        self._current_mode = "train"
        self._sequential.train()

        if self.has_criterion and isinstance(self.criterion, nn.Module):
            self.criterion.train()
        
        if self.has_metrics and isinstance(self.metric, nn.Module):
            self.metric.train()
        
        return self
    
    def eval(self) -> TypeVar("Trainer"):
        r"""Set trainer to eval mode.
        
        Returns:
            torecsys.trainer.Trainer: self
        """
        self._current_mode = "eval"
        self._sequential.eval()

        if self.has_criterion and isinstance(self.criterion, nn.Module):
            self.criterion.eval()
        
        if self.has_metrics and isinstance(self.metric, nn.Module):
            self.metric.eval()
        
        return self
    
    def jit(self) -> TypeVar("Trainer"):
        r"""To enable jit in the trainer.
        
        Args:
            self (Trainer): self.
        """
        raise NotImplementedError(".jit() is not implemented.")
    
    def fit(self, 
            mode  : str = None, 
            reset : bool = True):
        r"""Method to train the model
        
        Args:
            mode (str, optional): Mode of trainer. 
                Defaults to None.
            reset (bool, optional): Boolean flag to reset counter in trainer. 
                Defaults to True.
        """
        if reset:
            self._global_step_cnt = 0
            self._iteration_cnt = 0
            self._epoch_cnt = 0
        
        # Get current mode from self
        if mode is None:
            mode = self._current_mode
        
        # Get dataloader of current mode
        dataloader = self._loaders[mode]
        num_batch = len(dataloader)

        epochs = self.has_max_num_epochs

        for epoch in range(epochs):
            # Iterate dataloader
            loader_iter = iter(dataloader)

            _steps_loss = 0.0
            _epoch_loss = 0.0

            # Logging and Callback

            # Generate progress bar of this epoch from dataloader
            if self.has_max_num_iterations:
                num_batch_epochs = min(self.max_num_iterations, num_batch)
            pbar = tqdm(range(num_batch_epochs), desc="Step Loss: ?")

            # Iterate through the progress bar
            for i in pbar:
                # Get next batch from dataloader
                batch_values = next(dataloader)
                
                # Calculate prediction and loss of the batch
                prediction, loss = self._iterate(batch_values, backward=True)

                # Convert loss from torch.Tensor to float
                loss_value = loss.cpu().item()
                _steps_loss += loss_value
                _epoch_loss += loss_value

                # Update progress bar description
                pbar.set_description(f"Step Loss: {loss_value:.4f}")

                # Logging for each y steps
                # if (self._global_step_cnt) % self._log_step == 0:
                # ...
                # _steps_loss = 0.0

                # Take validate
            
                self._global_step_cnt += 1
                self._iteration_cnt += 1
            
            self._epoch_cnt += 1
            _epoch_loss = 0.0

            # Logging and Callback
    
    def train_for(self):
        return
    
    def validate_for(self):
        return
    
    def apply_model(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        r"""Apply model forward.

        Args:
            inputs (Dict[str, T]): Dictionary to input to sequential.
        
        Returns:
            torch.Tensor: output of sequential.
        """
        if hasattr(self, "_base_device_ordinal"):
            base_device_ordinal = self._base_device_ordinal
        else:
            base_device_ordinal = None
        
        if self._devices is not None:
            return nn_parallel.data_parallel(self._sequential, inputs, list(self._devices), output_device=base_device_ordinal)
        else:
            return self._sequential(inputs)
    
    def _iterate(self, 
                 batch_values : Dict[str, torch.Tensor],
                 mode         : str = None,
                 backward     : bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Iteration for each batch of inputs.
        
        Args:
            batch_values (Dict[str, T]): Dictionary.
            mode (str): Mode of trainer.
            backward (bool): Boolean flag to do backpropagation
        
        Returns:
            Tuple[T, T]: Prediction and loss of the model.
        """
        if mode is None:
            mode = self._current_mode

        # Zero the parameter gradients
        self.optimizer.zero_grad()

        # Get batch_data with batch_values
        batch_data = self._get_batch(batch_values)

        # Calculate forward prediction
        batch_inputs = batch_data.pop("batch_inputs")
        prediction = self.apply_model(batch_inputs)

        # Calculate loss and regularized loss
        loss = self._get_loss(prediction, **batch_data)
        
        if self.has_regularizer:
            # Apply regularization to loss if required
            named_params = list(self._sequential.named_parameters())
            reg_loss = self.regularizer(named_params)
            loss += reg_loss
        
        if backward:
            # Take backpropagation of loss if required 
            loss.backward()
        
        return prediction, loss
    
    def _get_batch(self, 
                   batch_values : Dict[str, torch.Tensor],
                   objective    : str = None) -> Dict[str, torch.Tensor]:
        r"""Method to convert batch_values to batch_data to calculate forward and loss.
        
        Args:
            batch_values (Dict[str, T]): Dictionary of inputs' tensor.
            objective (str): Objective of trainer.
        
        Raises:
            AssertionError: when objective is not in ["clickthroughrate", "embedding", "learningtorank"].
        
        Returns:
            Dict[str, T]: Dictionary of batch data.
        """
        # Get current objective of trainer
        if objective is None:
            objective = self._objective

        # Initialize empty dictionary to store batch data
        batch_data = dict()

        if self._objective is "clickthroughrate":
            # Get targets field values from batch_values
            targets = batch_values.pop(self.targets_name)
            batch_data["batch_inputs"] = batch_values
            batch_data["batch_targets"] = targets

        elif self._objective is "embedding":
            pass

        elif self._objective is "learningtorank":
            # Initialize empty dictionary to store batch inputs
            batch_inputs = dict()

            # Generative negative samples from negative sampler
            neg_inputs = self.negative_sampler(batch_values)

            # Stack pos_inputs and neg_inputs into a batch of inputs,
            # with shape = (B * (1 + num_neg), ...) for each tensor
            for inp_field in batch_values:
                # Get tensors of positive samples and negative samples
                pos_tensors = batch_values[inp_field]
                neg_tensors = neg_inputs[inp_field]

                # Get batch_size of positive samples and number of negative samples
                batch_size = pos_tensors.size(0)
                num_neg = int(neg_tensors.size(0) / batch_size)

                # Reshape tensors for stacking
                # inputs: pos_tensors, shape = (B, N)
                # inputs: neg_tensors, shape = (B * num_neg, N)
                # output: pos_tensors, shape = (B, 1, N)
                # output: neg_tensors, shape = (B, num_neg, N)
                # TODO: handle pos_tensors with dim > 2
                field_size = pos_tensors.size(1)
                pos_tensors = pos_tensors.view(batch_size, 1, field_size)
                neg_tensors = neg_tensors.view(batch_size, num_neg, field_size)

                # Stack tensors and store it to batch_inputs
                # inputs: pos_tensors, shape = (B, 1, N)
                # inputs: neg_tensors, shape = (B, num_neg, N)
                # output: b_tensors, shape = (B * (1 + num_neg), N)
                b_tensors = torch.cat([pos_tensors, neg_tensors], dim = 1)
                b_tensors = b_tensors.view(batch_size * (1 + num_neg), field_size)
                batch_inputs[inp_field] = b_tensors
            
            # Store batch_inputs to batch_data
            batch_data["batch_inputs"] = batch_inputs
            batch_data["batch_size"] = batch_size

        else:
            raise AssertionError(f"{self._objective} not allowed.")

        return batch_data
    
    def _get_loss(self,
                  prediction : torch.Tensor,
                  objective  : str = None,
                  mode       : str = None,
                  **kwargs) -> torch.Tensor:
        r"""Method to calculate loss of trainer.
        
        Args:
            prediction (T): Predicted values of model in trainer.
            objective (str, optional): Objective of trainer. 
                Defaults to None.
            mode (str, optional): Mode of trainer. 
                Defaults to None.
        
        Raises:
            AssertionError: when mode is not in ["train", "eval"].
            AssertionError: when objective is not ["clickthroughrate", "embedding", "learningtorank"].
        
        Returns:
            T, dtype = torch.float, shape = (B, 1): loss.
        """
        # Get current objective and mode of trainer
        if objective is None:
            objective = self._objective

        if mode is None:
            mode = self._current_mode

        # Get criterion by mode
        if mode is "train":
            criterion = self.criterion
        elif mode is "eval":
            criterion = self.validate_criterion
        else:
            raise AssertionError(f"{mode} not allowed.")
        
        # Calculate loss by criterion
        if objective is "clickthroughrate":
            # Get targets from kwargs
            # output: targets, shape = (B, ...)
            targets = kwargs.pop("batch_targets")
            
            # Calculate loss with prediction and targets
            # inputs: prediction, shape = (B, 1)
            # inputs: targets, shape = (B, ...)
            # output: loss, shape = (B, 1)
            loss = criterion(prediction, targets)

        elif objective is "embedding":
            pass

        elif objective is "learningtorank":
            # Get batch size from kwargs
            batch_size = kwargs.pop("batch_size")
            num_neg = int(prediction.size(0) / batch_size) - 1

            # Split prediction to pos_out and neg_out
            # inputs: prediction, shape = (B * (1 + num_neg), 1)
            # output: pos_out, shape = (B, 1)
            # output: neg_out, shape = (B * num_neg, 1)
            pos_out, neg_out = prediction.view(batch_size, -1, 1).split((1, num_neg), dim=1)
            pos_out = pos_out.squeeze(-1)
            neg_out = neg_out.squeeze(-1)

            # Calculate loss
            # output: loss, shape = (B, 1)
            loss = criterion(pos_out, neg_out)

        else:
            raise AssertionError(f"{self._objective} not allowed.")
        
        return loss
    
    def predict(self):
        return
    
    def save_for(self):
        return
    
    def save(self):
        return
    
    def load(self):
        return
    
    def summary(self,                
                deco        : int = Texttable.BORDER,
                cols_align  : List[str] = ["l", "l"],
                cols_valign : List[str] = ["t", "t"]) -> TypeVar("Trainer"):
        # Get attributes from self and initialize _vars of parameters 
        _vars = {
            "inputs"        : self._inputs.__class__.__name__ if self.has_inputs else None,
            "model"         : self._model.__class__.__name__ if self.has_model else None,
            "loss"          : self.criterion.__class__.__name__ if self.has_criterion else None,
            "optimizer"     : self._optimizer.__class__.__name__ \
                if getattr(self, "_optimizer", None) is not None else None,
            "reg norm"      : self.regularizer.norm if self.has_regularizer else None,
            "reg lambda"    : self.regularizer.weight_decay if self.has_regularizer else None,
            "num of epochs" : self._max_num_epochs,
            "log directory" : self._log_directory
        }
        
        # Create and configurate Texttable
        t = Texttable()
        t.set_deco(deco)
        t.set_cols_align(cols_align)
        t.set_cols_valign(cols_valign)

        # Append data to texttable
        t.add_rows(
            [["Name: ", "Value: "]] + \
            [[k.capitalize(), v] for k, v in _vars.items() if v is not None]
        )

        # Print summary with texttable
        print(t.draw())

        return self
    
    def generate_report(self) -> str:
        r"""Summarize report of experiment of model.

        TODO:

        #. data per hyperparameter (e.g. dropout size, hidden size)

        #. training loss per epochs

        #. metrics, including AUC, logloss

        #. matplotlib line chart of data
        
        Returns:
            str: string of result of texttable.draw
        """
        raise NotImplementedError("not implemented.")

        # initialize and configurate Texttable
        t = Texttable()
        t.set_deco(Texttable.BORDER)
        t.set_cols_align(["l", "l"])
        t.set_cols_valign(["t", "t"])

        # append data to texttable

        return t.draw()
    
    @classmethod
    def build(cls, **trainer_config):
        """Factory method to build the trainer.
        """
        

class Trainer(object):
    r"""Trainer object to train model, including trs.inputs.inputs_wrapper and trs.model.
    """
    def __init__(self,
                 inputs_wrapper : _Inputs,
                 model          : trs_model._Model,
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
            AssertionError: when parameter of the given name cannot be found.
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
            raise AssertionError("parameter %s cannot found." % param_name)

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
        
    def _describe(self) -> str:
        r"""Show summary of trainer
        
        Returns:
            str: string of result of texttable.draw
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
                if (global_step + 1) % self.log_step == 0:
                    if self.verboses >= 1:
                        self.logger.debug("step avg loss at step %d of epoch %d : %.4f" % (i, epoch + 1, steps_loss / self.log_step))
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
            torch.save(self.sequential.state_dict(), save_file)
    
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
