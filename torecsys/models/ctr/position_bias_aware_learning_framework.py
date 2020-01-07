from . import _CtrModel
import torch
import torch.nn as nn
from torecsys.layers import DNNLayer, PALLayer
from torecsys.utils.decorator import jit_experimental, no_jit_experimental_by_namedtensor
from typing import Dict

class PositionBiasAwareLearningFrameworkModel(_CtrModel):
    r""""Model class of Positon-bias aware learning framework (PAL).
    
    :Reference:

    #. `Huifeng Guo et al, 2019. PAL: a position-bias aware learning framework for CTR prediction in live recommender systems <https://dl.acm.org/citation.cfm?id=3347033&dl=ACM&coll=DL>`_.
    
    """
    @no_jit_experimental_by_namedtensor
    def __init__(self,
                 pctr_model       : nn.Module,
                 pos_model        : nn.Module = None,
                 output_size      : int = None,
                 max_num_position : int = None,
                 **kwargs):
        r"""Initialize PositionBiasAwareLearningFrameworkModel
        
        Args:
            pctr_model (nn.Module): model of CTR prediction
            pos_model (nn.Module, optional): Model of position-bias.
                Defaults to None.
            output_size (int, optional): Size of output tensor of click-through-rate model.
                Defaults to None.
            max_num_position (int, optional): Maximum length of list, i.e. Maximum number of postion.
                Defaults to None.
        
        Arguments:
            layer_sizes (List[int]): Layer sizes of DNNLayer.
            dropout_p (List[float]): Probability of Dropout in DNNLayer.
            activation (Callable[[T], T]): Activation function in DNNLayer.
        
        Attributes:
            pctr_model (nn.Module): model of CTR prediction.
            pos_model (nn.Module): Model of position-bias.

        """
        # refer to parent class
        super(PositionBiasAwareLearningFrameworkModel, self).__init__()

        # Initialize pCTR module
        self.pctr_model = pctr_model

        # Initialize position module
        if pos_model is not None:
            # Bind a nn.Module to pos_model
            self.pos_model = pos_model
        else:
            # Initialize sequential to store the module of positional embedding
            self.pos_model = nn.Sequential()

            # Initialize positional embedding layer
            self.pos_model.add_module("PosEmbedding", PALLayer(
                input_size       = output_size,
                max_num_position = max_num_position
            ))

            # Initialize dense layer after apply positional embedding bias
            self.pos_model.add_module("Dense", DNNLayer(
                inputs_size = output_size, 
                output_size = 1, 
                layer_sizes = kwargs.get("layer_sizes"), 
                dropout_p   = kwargs.get("dropout_p"), 
                activation  = kwargs.get("activation")
            ))

            # Initialize sigmoid layer to transform outputs
            self.pos_model.add_module("Sigmoid", nn.Sigmoid())
        
    def forward(self, inputs: Dict[str, torch.Tensor], pos_inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward calculation of PositionBiasAwareLearningFrameworkModel
        
        Args:
            inputs (Dict[str, T]): Input arguments of pctr_model.
            pos_inputs (T, shape = (B, ), dtype = torch.int): position of inputs.
        
        Returns:
            T, shape = (B, O), dtype = torch.float: Output of PositionBiasAwareLearningFrameworkModel
        """
        # Calculate with pctr model forwardly
        # inputs: inputs, dictionary of inputs of self.pctr_model
        # output: pctr_output, shape = (B, O = 1)
        pctr_out = self.pctr_model(**inputs)

        # Calculate with pos model forwardly
        # inputs: pos_inputs, shape = (B, N = 1)
        # output: pos_output, shape = (B, O = 1)
        output  = self.pos_model((pctr_out, pos_inputs))
        output.names = ("B", "O")

        # Drop names of outputs, since autograd doesn't support NamedTensor yet.
        output = output.rename(None)

        return output
        
    def predict(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Prediction of PositionBiasAwareLearningFrameworkModel
        
        Args:
            inputs (Dict[str, T]): Input arguments of pctr_model.
        
        Returns:
            T, shape = (B, O), dtype = torch.float: Prediction of PositionBiasAwareLearningFrameworkModel
        """
        # Calculate with pctr model forwardly
        # inputs: inputs, dictionary of inputs of self.pctr_model
        # output: pctr_output, shape = (B, O = 1)
        pctr_out = self.pctr_model(**inputs)
        pctr_out.names = ("B", "O")

        # Drop names of outputs, since autograd doesn't support NamedTensor yet.
        pctr_out = pctr_out.rename(None)

        return pctr_out
