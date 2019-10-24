from . import _CtrModel
from torecsys.layers import WideLayer
from torecsys.utils.decorator import jit_experimental, no_jit_experimental_by_namedtensor
import torch
import torch.nn as nn

class LogisticRegressionModel(_CtrModel):
    r"""Model class of Logistic Regression (LR).
    
    Logistic Regression is a model to predict click through rate with a simple logistic 
    regression, i.e. a linear layer plus a sigmoid transformation to make the outcome between 
    0 and 1, which is to represent the probability of the input is true.

    """
    @no_jit_experimental_by_namedtensor
    def __init__(self, 
                 inputs_size: int,
                 output_size: int = 1):
        r"""Initialize LogisticRegressionModel
        
        Args:
            inputs_size (int): Inputs size of logistic regression, 
                i.e. number of fields * embedding size.
            output_size (int, optional): Output size of model. 
                Defaults to 1.
        
        Attributes:
            linear (nn.Module): Module of linear layer.
            sigmoid (nn.Module): Module of sigmoid layer.
        """
        # refer to parent class
        super(LogisticRegressionModel, self).__init__()

        # initialize linear layer
        self.linear = nn.Linear(inputs_size, output_size)

        # initialize sigmoid layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, feat_inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward calculation of LogisticRegressionModel
        
        Args:
            feat_inputs (T), shape = (B, N, E), dtype = torch.float: Linear Features tensors.
        
        Returns:
            T, shape = (B, O), dtype = torch.float: Output of LogisticRegressionModel
        """
        # Calculate linear projection
        # inputs: feat_inputs, shape = (B, N, E)
        # output: outputs, shape = (B, O)
        feat_inputs = feat_inputs.flatten(["N", "E"], "O")
        outputs = self.linear(feat_inputs.rename(None))
        outputs.names = ("B", "O")

        # Transform with sigmoid function
        # inputs: outputs, shape = (B, O)
        # output: outputs, shape = (B, O)
        outputs = self.sigmoid(outputs)

        # Drop names of outputs, since autograd doesn't support NamedTensor yet.
        outputs = outputs.rename(None)

        return outputs
