import torecsys.inputs.InputsWrapper
import torecsys.models._Model
import torch
import torch.nn as nn


class Sequential(nn.Module):
    r"""Sequential container, where the module of embeddings and model will be added to it 
    in the order they are passed in the constructor. 
    """
    def __init__(self, 
                 inputs : torecsys.inputs.InputsWrapper, 
                 model  : torecsys.models._Model):
        r"""Initialize Sequential,
        
        Args:
            inputs (torecsys.inputs.InputsWrapper): Inputs wrapper where the return is a dictionary 
                of inputs' tensors which are passed to the model directly.
            model (torecsys.models._Model): Model class to be trained and used in prediction.

        Attributes:
            inputs (torecsys.inputs.InputsWrapper): Inputs wrapper where the return is a dictionary 
                of inputs' tensors which are passed to the model directly.
            model (torecsys.models._Model): Model class to be trained and used in prediction.
        """
        # refer to parent class
        super(Sequential, self).__init__()

        # bind inputs and model to inputs and model
        self.inputs = inputs
        self.model = model

        # add module in inputs and model to the Module
        # for name, param in self.inputs.named_parameters():
        #     self.add_module(name, param)
        # for name, param in self.model.named_parameters():
        #     self.add_module(name, param)
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        r"""Forward calculation of Sequential.
        
        Args:
            inputs (Dict[str, T]): Dictionary of inputs, where key is name of input fields, and value is 
                tensor pass to Input class.
        
        Returns:
            T: Output of model.
        """
        # transform and embed inputs
        inputs = self.inputs(inputs)

        # calculate forward propagation of model
        outputs = self.model(**inputs)
        return outputs
        