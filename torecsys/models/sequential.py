import torch
import torch.nn as nn
from torecsys.inputs.inputs_wrapper import InputsWrapper
from torecsys.models import _Model
from typing import Dict

class Sequential(nn.Module):
    r"""Sequential container, where the module of embeddings and model will be added to it 
    in the order they are passed in the constructor. 
    """
    def __init__(self, 
                 inputs_wrapper : InputsWrapper, 
                 model          : _Model):
        r"""Initialize Sequential container.
        
        Args:
            inputs_wrapper (InputsWrapper): Inputs wrapper where the return is a dictionary 
                of inputs' tensors which are passed to the model directly.
            model (_Model): Model class to be trained and used in prediction.

        Attributes:
            inputs_wrapper (InputsWrapper): Inputs wrapper where the return is a dictionary 
                of inputs' tensors which are passed to the model directly.
            model (_Model): Model class to be trained and used in prediction.
        """
        # refer to parent class
        super(Sequential, self).__init__()

        # bind inputs and model to inputs and model
        self.inputs_wrapper = inputs_wrapper
        self.model = model
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        r"""Forward calculation of Sequential.
        
        Args:
            inputs (Dict[str, T]): Dictionary of inputs, where key is name of input fields, 
                and value is tensor pass to Input class.
        
        Returns:
            T: Output of model.
        """
        # transform and embed inputs
        inputs = self.inputs_wrapper(inputs)

        # calculate forward propagation of model
        outputs = self.model(**inputs)
        
        return outputs
        