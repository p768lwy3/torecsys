from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from pytorch_lightning import LightningModule
from pytorch_metric_learning.miners import BaseMiner
from texttable import Texttable
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torecsys.losses
import torecsys.models as trs_model
from torecsys.inputs import Inputs
from torecsys.inputs.base import BaseInput
from torecsys.layers.regularization import Regularizer
from torecsys.models.sequential import Sequential


class TorecsysModule(LightningModule):
    """
    ...
    """

    TorecsysModule = TypeVar('TorecsysModule')

    MODULE_TYPE_CTR = 'click_through_rate'
    MODULE_TYPE_EMB = 'embedding'
    MODULE_TYPE_LTR = 'learning_to_rank'
    MODULE_TYPE_ENUM = {
        'ctr': MODULE_TYPE_CTR,
        'emb': MODULE_TYPE_EMB,
        'ltr': MODULE_TYPE_LTR
    }
    MODULE_TYPES = [MODULE_TYPE_CTR, MODULE_TYPE_EMB, MODULE_TYPE_LTR]

    def __init__(self):
        """
        Initialize TorecsysModule
        """
        super().__init__()

        self._objective: str = self.MODULE_TYPE_CTR
        self._inputs: Optional[BaseInput] = None
        self._model: Optional[nn.Module] = None
        self._sequential: Optional[nn.Sequential] = None
        self._targets_name: Optional[str] = None
        self._miner: Optional[BaseMiner] = None
        self._criterion: Optional[nn.Module] = None
        self._regularizer: Optional[nn.Module] = None
        self._optimizer: Optional[nn.Module] = None
        self._scheduler: Optional[ReduceLROnPlateau] = None

    @property
    def objective(self) -> str:
        """
        Get the objective of trainer
        
        Returns:
            str: objective of the trainer
        """
        return self._objective

    @objective.setter
    def objective(self, objective: str):
        """
        Set objective to the trainer

        Args:
            objective (str): objective to be set to the trainer
        """
        _ = self.set_objective(objective)

    def set_objective(self, objective: str) -> TorecsysModule:
        """
        Set objective to the self.objective

        Args:
            objective (str): objective to be set to self.objective

        Raises:
            TypeError: whether type of objective is not allowed to be set

        Returns:
            torecsys.trainer.TorecsysModule: self
        """
        if not isinstance(objective, str):
            raise TypeError(f'objective must be a str, given {type(objective)}')

        objective = objective.lower()
        objective = self.MODULE_TYPE_ENUM.get(objective) if objective in self.MODULE_TYPE_ENUM else objective

        if objective not in self.MODULE_TYPES:
            raise AssertionError(f'value of objective is not allowed, given {objective}, required {self.MODULE_TYPES}')

        self._objective = objective

        return self

    @property
    def inputs(self) -> BaseInput:
        """
        Get the inputs of the trainer
        
        Returns:
            BaseInput: inputs of the trainer
        """
        return self._inputs

    @property
    def has_inputs(self) -> bool:
        """
        Return whether the inputs of the trainer is exists

        Returns:
            bool: True if self._inputs is not None else False
        """
        return self._inputs is not None

    @inputs.setter
    def inputs(self, inputs: BaseInput):
        """
        Set Input to the trainer

        Args:
            inputs (torecsys.inputs.base._Inputs): Input to be set to the trainer
        """
        _ = self.set_inputs(inputs)

    def set_inputs(self, inputs: Optional[BaseInput] = None, **kwargs) -> TorecsysModule:
        """
        Set inputs to the self.inputs

        Args:
            inputs (torecsys.inputs.BaseInput, optional): inputs to be set to self.inputs. Defaults to None

        Raises:
            TypeError: whether type of inputs is not allowed to be set

        Returns:
            torecsys.trainer.TorecsysModule: self
        """
        if inputs is not None:
            if isinstance(inputs, BaseInput):
                self._inputs = inputs
            else:
                raise TypeError(f'inputs must be a torecsys.inputs.base._Inputs object, given {type(inputs)}')
        else:
            self._inputs = Inputs(schema=kwargs)

        return self

    @property
    def model(self) -> nn.Module:
        """
        Get the model of the trainer
        
        Returns:
            torch.nn.Module: model of the trainer
        """
        return self._model

    @property
    def has_model(self) -> bool:
        """
        Return whether the model of the trainer is exists

        Returns:
            bool: True if self._model is not None else False
        """
        return self._model is not None

    def set_model(self, method: Union[str, Dict[str, Any], nn.Module], **kwargs) -> TorecsysModule:
        """
        Set model to the trainer

        Args:
            method (Union[str, nn.Module]): model to be set to the trainer._model

        Raises:
            AssertionError: whether no model is found with the given string of method
            TypeError: whether type of model is not allowed to be set

        Returns:
            torecsys.trainer.TorecsysModule: self
        """
        if isinstance(method, str):
            model_method = getattr(trs_model, method, None)
            if model_method is None:
                raise AssertionError(f'given {method} is not found')
        elif isinstance(method, nn.Module):
            self._model = method
            return self
        elif callable(method) and isinstance(method, type):
            model_method = method
        else:
            raise TypeError(f'type of method not allowed, given {type(method).__name__}, '
                            f'required: [str, Dict[str, Any], nn.Module]')

        self._model = model_method(**kwargs)

        return self

    @model.setter
    def model(self, model: nn.Module):
        """
        Set model to the trainer

        Args:
            model (torch.nn.Module): model to be set to the trainer
        """
        self.set_model(model)

    @property
    def sequential(self) -> nn.Sequential:
        """
        Get the sequential of the trainer, i.e. stack of the self._inputs and self._model
        
        Returns:
            torch.nn.Sequential: sequential of the trainer
        """
        return self._sequential

    def set_sequential(self,
                       inputs: BaseInput = None,
                       model: nn.Module = None) -> TorecsysModule:
        """
        Set sequential with inputs and model to the trainer

        Args:
            inputs (BaseInput): inputs object. required: output fields' names = model inputs' names
            model (BaseModel): model object.

        Raises:
            AssertionError: whether inputs or model is not found

        Returns:
            torecsys.trainer.TorecsysModule: self
        """
        # Set new values to private variables
        if inputs is not None:
            self._inputs = inputs

        if model is not None:
            self._model = model

        if not self.has_inputs or not self.has_model:
            raise AssertionError('inputs or model is not found')

        self._sequential = Sequential(inputs=self._inputs, model=self._model)

        return self

    @property
    def regularizer(self) -> nn.Module:
        """
        Get the regularizer of the trainer

        Returns:
            nn.Module: regularizer of the trainer
        """
        return self._regularizer

    @property
    def has_regularizer(self) -> bool:
        """
        Return whether the regularizer of the trainer is exists

        Returns:
            bool: True if self._regularizer is not None else False
        """
        return self._regularizer is not None

    @regularizer.setter
    def regularizer(self, regularizer: nn.Module):
        """
        Set regularizer to the trainer.

        Args:
            regularizer (torecsys.layers.regularization.Regularizer): Regularizer to be set to the trainer.
        """
        self.set_regularizer(regularizer)

    def set_regularizer(self,
                        regularizer: Optional[nn.Module] = None,
                        **kwargs) -> TorecsysModule:
        """
        Set regularizer of the trainer

        Args:
            regularizer (torecsys.layers.regularization, optional): regularizer to be bind to the trainer._regularizer

        Raises:
            TypeError: whether type of regularizer is not allowed to be set

        Returns:
            torecsys.trainer.TorecsysModule: self
        """
        if regularizer is None:
            regularizer = Regularizer(**kwargs)
        else:
            if isinstance(model, Regularizer):
                pass
            else:
                raise TypeError('regularizer must be a torecsys.layers.regularization.Regularizer.')

        self._regularizer = regularizer

        return self

    @property
    def miner(self) -> BaseMiner:
        """
        Get the miner of the trainer

        Returns:
            pytorch_metric_learning.miners.BaseMiner: miner (sampler) of the trainer
        """
        return self._miner

    @property
    def has_miner(self) -> bool:
        """
        Return whether the miner of the trainer is exists

        Returns:
            bool: True if self._miner is not None else False
        """
        return self._miner is not None

    @miner.setter
    def miner(self, value: Union[str, Callable[[torch.Tensor], torch.Tensor], dict]):
        """
        Set negative sampler to the trainer

        Args:
            value (Union[str, Callable[[T], T], dict]): value to build negative sampler

        Raises:
            TypeError: when type of value is not allowed.
        """
        if isinstance(value, str) or callable(value):
            self.build_negative_sampler(value)
        elif isinstance(value, dict):
            self.build_negative_sampler(**value)
        else:
            raise TypeError(f'{type(value).__name__} not allowed.')

    @property
    def criterion(self) -> Callable:
        """
        Get the criterion (loss function) of the trainer

        Returns:
            Callable: criterion of the trainer
        """
        return self._criterion

    @property
    def has_criterion(self) -> bool:
        """
        Return whether the criterion of the trainer is exists

        Returns:
            bool: True if self._criterion is not None else False
        """
        return self._criterion is not None

    @criterion.setter
    def criterion(self, value: Union[str, nn.Module, dict]):
        """
        Set loss to the trainer

        Args:
            value (Union[str, nn.Module, dict]): value to build criterion.

        Raises:
            TypeError: when type of value is not allowed.
        """
        # Bind criterion to _criterion
        if isinstance(value, str) or isinstance(value, nn.Module):
            self.set_criterion(value)
        elif isinstance(value, dict):
            self.set_criterion(**value)
        else:
            raise TypeError(f"{type(value).__name__} not allowed.")

    def set_criterion(self,
                      method: Union[str, nn.Module],
                      **kwargs) -> TorecsysModule:
        """
        Set criterion

        Args:
            method (Union[str, nn.Module]): method of criterion

        Raises:
            AssertionError: when method is not found
            TypeError: when type of method is not allowed

        Returns:
            torecsys.trainer.TorecsysModule: self
        """
        if isinstance(method, str):
            criterion_method = getattr(torecsys.losses, method, None)
            if criterion_method is None:
                criterion_method = getattr(nn.modules.loss, method, None)
                if criterion_method is None:
                    raise AssertionError(f'{method} not found')
        elif callable(method) and isinstance(method, type):
            criterion_method = method
        else:
            raise TypeError(f'{type(method).__name__} not allowed')

        self._criterion = criterion_method(**kwargs)

        return self

    @property
    def optimizer(self) -> nn.Module:
        """
        Return optimizer of the trainer if self._optimizer is exists

        Raises:
            AssertionError: when optimizer is not exists in the trainer

        Returns:
            nn.Module: optimizer of the trainer
        """
        if self._optimizer is None:
            raise AssertionError('self._optimizer is None')
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value: Union[str, nn.Module, dict]):
        """
        Set optimizer to the trainer

        Args:
            value (Union[str, Callable[[T], T], dict]): value to build optimizer

        Raises:
            TypeError: when type of value is not allowed
        """
        if isinstance(value, str) or isinstance(value, nn.Module):
            self.set_optimizer(value)
        elif isinstance(value, dict):
            self.set_optimizer(**value)
        else:
            raise TypeError(f'{type(value).__name__} not allowed.')

    def set_optimizer(self,
                      method: Union[str, nn.Module],
                      parameters: nn.parameter.Parameter = None,
                      **kwargs) -> TorecsysModule:
        """
        Set optimizer

        Args:
            method (Union[str, nn.Module]): method of optimizer
            parameters (torch.nn.parameter.Parameter, optional): parameters to be optimized. Defaults to None

        Raises:
            AssertionError: when method is not found
            TypeError: when type of method is not allowed

        Returns:
            torecsys.trainer.TorecsysModule: self
        """
        if isinstance(method, str):
            optimizer_method = getattr(optim, method, None)
            if optimizer_method is None:
                raise AssertionError(f'{method} not found')
        elif callable(method) and isinstance(method, type):
            optimizer_method = method
        elif isinstance(method, optim.Optimizer):
            self._optimizer = method
            return self
        else:
            raise TypeError(f'{type(method).__name__} not allowed')

        if self.has_inputs and self.has_model:
            self.set_sequential()
            parameters = self._sequential.parameters() if parameters is None else parameters
            self._optimizer = optimizer_method(parameters, **kwargs)
        else:
            raise ValueError('missing inputs and model for the trainer to initiate optimizer')

        return self

    def configure_optimizers(self) -> Tuple[dict]:
        """
        Define optimizers and schedulers of pytorch_lightning.LightningModule

        Returns:

        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return (
            {
                'optimizer': optimizer,
                # 'lr_scheduler': {
                #     'scheduler': ReduceLROnPlateau(optimizer, ...),
                #     'monitor': 'metric_to_track',
                # }
            },
        )

    @property
    def targets_name(self) -> str:
        """
        Get the target field name of the trainer
        
        Returns:
            str: name of target field of the trainer
        """
        return self._targets_name

    @targets_name.setter
    def targets_name(self, targets_name: str):
        """
        Set targets_name of the trainer

        Args:
            targets_name (str): targets name to be set for getting targets field in batch
        """
        self.set_targets_name(targets_name)

    def set_targets_name(self, targets_name: str) -> TorecsysModule:
        """
        Set targets_name of the trainer.

        Args:
            targets_name (str): targets name to be set for getting targets field in batch

        Raises:
            TypeError: when type of targets_name is not allowed

        Returns:
            torecsys.trainer.TorecsysModule: self
        """
        if not isinstance(targets_name, str):
            raise TypeError(f'{type(targets_name).__name__} not allowed')

        self._targets_name = targets_name

        return self

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Training step of pytorch_lightning.LightningModule

        Args:
            batch (Dict[str, torch.Tensor]): output of the dataloader
            batch_idx (int): integer displaying index of this batch

        Returns:
            Union[T, Dict[str, Any]]: loss tensor
        """
        batch_data = self._parse_batch(batch)
        batch_inputs = batch_data.pop('batch_inputs')

        prediction = self._apply_model(batch_inputs)

        loss = self._get_loss(prediction, **batch_data)
        loss_val = loss.get('loss') if isinstance(loss, dict) else loss

        if self.has_regularizer:
            named_params = list(self._sequential.named_parameters())
            reg_loss = self.regularizer(named_params)
            loss_val += reg_loss
            loss['loss'] = loss_val

        # logs metrics for each training_step, and the average across the epoch, to the progress bar and logger
        self.log('train_loss', loss_val, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    # def validation_step(self):
    #     """
    #     Validation step of pytorch_lightning.LightningModule
    #
    #     Returns:
    #
    #     """
    #     return

    # def test_step(self):
    #     """
    #     Test step of pytorch_lightning.LightningModule
    #
    #     Returns:
    #
    #     """
    #     return

    def forward(self, *args, **kwargs) -> Any:
        """

        Args:
            *args:
            **kwargs:

        Returns:

        """
        # Calculate forward prediction
        batch_data = self._parse_batch(kwargs.get('batch'))
        batch_inputs = batch_data.pop('batch_inputs')
        prediction = self._apply_model(batch_inputs)
        return prediction

    def _parse_batch(self,
                     batch_values: Dict[str, torch.Tensor],
                     objective: str = None) -> Dict[str, torch.Tensor]:
        """
        Method to convert batch_values to batch_data to calculate forward and loss

        Args:
            batch_values (Dict[str, T]): dictionary of inputs' tensor
            objective (str): objective of trainer

        Raises:
            AssertionError: when objective is not in ["clickthroughrate", "embedding", "learningtorank"]

        Returns:
            Dict[str, T]: Dictionary of batch data
        """
        batch_data = {}

        if self._objective is None:
            if objective is None:
                raise ValueError('missing objective for the trainer')

            self._objective = objective

        if self._objective == self.MODULE_TYPE_CTR:
            targets = batch_values.pop(self.targets_name)
            batch_data['batch_inputs'] = batch_values
            batch_data['batch_targets'] = targets
        elif self._objective == self.MODULE_TYPE_EMB:
            pass
        elif self._objective == self.MODULE_TYPE_LTR:
            batch_inputs = {}

            # Stack pos_inputs and neg_inputs into a batch of inputs, with shape = (B * (1 + num_neg), ...)
            for inp_field in batch_values:
                # Get tensors of positive samples and negative samples
                pos_tensors = batch_values[inp_field]

                # Get batch_size of positive samples and number of negative samples
                batch_size = pos_tensors.size(0)
                num_neg = int(neg_tensors.size(0) / batch_size)

                # Reshape tensors for stacking
                # inputs: pos_tensors, shape = (B, N)
                # inputs: neg_tensors, shape = (B * num_neg, N)
                # output: pos_tensors, shape = (B, 1, N)
                # output: neg_tensors, shape = (B, num_neg, N)
                field_size = pos_tensors.size(1)
                pos_tensors = pos_tensors.view(batch_size, 1, field_size)
                neg_tensors = neg_tensors.view(batch_size, num_neg, field_size)

                # Stack tensors and store it to batch_inputs
                # inputs: pos_tensors, shape = (B, 1, N)
                # inputs: neg_tensors, shape = (B, num_neg, N)
                # output: b_tensors, shape = (B * (1 + num_neg), N)
                b_tensors = torch.cat([pos_tensors, neg_tensors], dim=1)
                b_tensors = b_tensors.view(batch_size * (1 + num_neg), field_size)
                batch_inputs[inp_field] = b_tensors

                # Store batch_inputs to batch_data
                batch_data['batch_inputs'] = batch_inputs
                batch_data['batch_size'] = batch_size
        else:
            raise AssertionError(f'{self._objective} not allowed')

        return batch_data

    def _apply_model(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Apply model forward

        Args:
            inputs (Dict[str, T]): dictionary to input to sequential
        
        Returns:
            torch.Tensor: output of sequential
        """
        return self._sequential(inputs)

    def _get_loss(self, prediction: torch.Tensor, objective: Optional[str] = None, **kwargs) \
            -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Method to calculate loss of trainer
        
        Args:
            prediction (T): predicted values of model in trainer
            objective (str, optional): objective of trainer. Defaults to None
        
        Raises:
            AssertionError: when mode is not in ["train", "eval"]
            AssertionError: when objective is not ["clickthroughrate", "embedding", "learningtorank"]
        
        Returns:
            Union[T, Dict[str, Any]], data_type = torch.float, shape = (B, 1): loss
        """
        if objective is None:
            objective = self._objective

        if objective == self.MODULE_TYPE_CTR:
            # Get targets from kwargs, output: targets, shape = (B, ...)
            targets = kwargs.pop('batch_targets')

            # Calculate loss with prediction and targets
            # inputs: prediction, shape = (B, 1)
            # inputs: targets, shape = (B, ...)
            # output: loss, shape = (B, 1)
            loss = self.criterion(prediction, targets)
        elif objective == self.MODULE_TYPE_EMB:
            loss = 0
        elif objective == self.MODULE_TYPE_LTR:
            batch_size = kwargs.pop('batch_size')
            num_neg = int(prediction.size(0) / batch_size) - 1

            # Split prediction to pos_out and neg_out
            # inputs: prediction, shape = (B * (1 + num_neg), 1)
            # output: pos_out, shape = (B, 1)
            # output: neg_out, shape = (B * num_neg, 1)
            pos_out, neg_out = prediction.view(batch_size, -1, 1).split((1, num_neg), dim=1)
            pos_out = pos_out.squeeze(-1)
            neg_out = neg_out.squeeze(-1)

            # Calculate loss, output: loss, shape = (B, 1)
            loss = self.criterion(pos_out, neg_out)
        else:
            raise AssertionError(f'{self._objective} not allowed')

        return {
            'loss': loss
        }

    def summary(self,
                deco: int = Texttable.BORDER,
                cols_align: Optional[List[str]] = None,
                cols_valign: Optional[List[str]] = None
                ) -> str:
        """
        Summarize training module

        Args:
            deco:
            cols_align:
            cols_valign:

        Returns:
            str: texttable string of summarization
        """
        cols_align = ['l', 'l'] if cols_align is None else cols_align
        cols_valign = ['t', 't'] if cols_valign is None else cols_valign

        objective = ' '.join([w.capitalize() for w in self._objective.split('_')])
        _vars = {
            'objective': objective,
            'inputs': self._inputs.__class__.__name__ if self.has_inputs else None,
            'model': self._model.__class__.__name__ if self.has_model else None,
            'reg norm': self._regularizer.norm if self.has_regularizer else None,
            'reg lambda': self._regularizer.weight_decay if self.has_regularizer else None,
            'loss': self._criterion.__class__.__name__ if self.has_criterion else None,
            'optimizer': self._optimizer.__class__.__name__ if getattr(self, "_optimizer", None) is not None else None
        }

        if self._objective == self.MODULE_TYPE_CTR:
            _vars.update({
                'target field name': self._targets_name
            })
        elif self._objective == self.MODULE_TYPE_EMB:
            pass
        elif self._objective == self.MODULE_TYPE_LTR:
            _vars.update({
                'miner': self._miner.__class__.__name__ if self.has_miner else None
            })

        t = Texttable()
        t.set_deco(deco)
        t.set_cols_align(cols_align)
        t.set_cols_valign(cols_valign)
        t.add_rows([['Name', 'Value']] + [[k.capitalize(), v] for k, v in _vars.items()])

        return t.draw()

    @classmethod
    def build(cls, **trainer_config):
        """
        Factory method to build the trainer

        Args:
            **trainer_config: See below.

        Keyword Args:
            load_from (str): load full config from a file path
            objective (str): objective to be set on trainer
            inputs_config (Dict[str, Any]): dictionary to build inputs
            model_config (Dict[str, Any]): dictionary to build model
            regularizer_config (Dict[str, Any]): dictionary to build regularizer
            criterion_config (Dict[str, Any]): dictionary to build criterion
            optimizer_config (Dict[str, Any]): dictionary to build optimizer
            targets_name (str): targets field name to be set on trainer
        """
        trainer = cls()

        if trainer_config.get('load_from'):
            trainer = trainer.load(**trainer_config.get('load_from'))
        else:
            if trainer_config.get('objective'):
                trainer.set_objective(trainer_config.get('objective'))

            if trainer_config.get('inputs_config'):
                trainer.set_inputs(**trainer_config.get('inputs_config'))

            if trainer_config.get('model_config'):
                trainer.set_model(**trainer_config.get('model_config'))

            if trainer_config.get('regularizer_config'):
                trainer.set_regularizer(**trainer_config.get('regularizer_config'))

            if trainer_config.get('criterion_config'):
                trainer.set_criterion(**trainer_config.get('criterion_config'))

            if trainer_config.get('optimizer_config'):
                trainer.set_optimizer(**trainer_config.get('optimizer_config'))

            if trainer_config.get('targets_name'):
                trainer.set_targets_name(trainer_config.get('targets_name'))

        return trainer
