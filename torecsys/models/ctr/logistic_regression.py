from . import _CtrModel
from torecsys.layers import WideLayer
from torecsys.utils.decorator import jit_experimental
import torch
import torch.nn as nn

class LogisticRegressionModel(_CtrModel):
    r"""LogisticRegressionModel is a model to predict click through rate with a simple 
    logistic regression, i.e. a linear layer plus a sigmoid transformation to make the 
    outcome between 0 and 1, which is to represent the probability of the input is true.
    """
    def __init__(self, 
                 inputs_size: int,
                 output_size: int = 1):
        r"""initialize logistic regression model
        
        Args:
            inputs_size (int): size of inputs' vector
            output_size (int, optional): size of output's vector. Defaults to 1.
        """
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(inputs_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feat_inputs: torch.Tensor) -> torch.Tensor:
        r"""feed forward of logistic regression
        
        Args:
            feat_inputs (T), shape = (B, N, 1), dtype = torch.float: first order outputs, i.e. outputs from nn.Embedding(V, 1)
        
        Returns:
            T, shape = (B, O), dtype = torch.float: output tensor of logistic regression
        """
        outputs = self.linear(feat_inputs.squeeze())
        outputs = self.sigmoid(outputs)
        return outputs
