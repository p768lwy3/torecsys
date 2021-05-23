from typing import Optional

import torch
import torch.nn as nn

from torecsys.models.ctr import CtrBaseModel


class LogisticRegressionModel(CtrBaseModel):
    """
    Model class of Logistic Regression (LR).
    
    Logistic Regression is a model to predict click through rate with a simple logistic regression,
    i.e. a linear layer plus a sigmoid transformation to make the outcome between 0 and 1,
    which is to represent the probability of the input is true.

    """

    def __init__(self,
                 inputs_size: int,
                 output_size: Optional[int] = 1):
        """
        Initialize LogisticRegressionModel
        
        Args:
            inputs_size (int): inputs size of logistic regression, i.e. number of fields * embedding size
            output_size (int, optional): output size of model. Defaults to 1
        """
        super().__init__()

        self.linear = nn.Linear(inputs_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, feat_inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward calculation of LogisticRegressionModel
        
        Args:
            feat_inputs (T), shape = (B, N, E), data_type = torch.float: linear Features tensors
        
        Returns:
            T, shape = (B, O), data_type = torch.float: output of LogisticRegressionModel
        """
        # Name the inputs tensor for flatten
        feat_inputs.names = ('B', 'N', 'E',)

        # Calculate linear projection
        # inputs: feat_inputs, shape = (B, N, E)
        # output: outputs, shape = (B, O)
        feat_inputs = feat_inputs.flatten(('N', 'E',), 'O')
        outputs = self.linear(feat_inputs.rename(None))
        outputs.names = ('B', 'O',)

        # Transform with sigmoid function
        # inputs: outputs, shape = (B, O)
        # output: outputs, shape = (B, O)
        outputs = self.sigmoid(outputs)

        # Drop names of outputs, since autograd doesn't support NamedTensor yet.
        outputs = outputs.rename(None)

        return outputs
