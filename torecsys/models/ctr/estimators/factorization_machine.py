from . import _CtrEstimator
from ..layers import FactorizationMachineLayer
from functools import partial
import torch
import torch.nn as nn
from typing import Dict

class FactorizationMachine(_CtrEstimator):
    r"""FactoizationMachine is an estimator of Factorization Machine which calculate interactions 
    between fields by the following equation:
    :math:`\^{y}(x) := b_{0} + \sum_{i=1}^{n} w_{i} x_{i} + \sum_{i=1}^{n} \sum_{j=1+1}^{n} <v_{i},v_{j}> x_{i} x_{j}` .

    :Reference:

    #. `Steffen Rendle, 2010. Factorization Machine <https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf>`_.

    """
    def __init__(self, 
                 embed_size    : int,
                 num_fields    : int,
                 dropout_p     : float = 0.1,
                 output_method : str = "concatenate",
                 output_size   : int = 1):
        r"""initialize Factorization Machine Estimator
        
        Args:
            embed_size (int): embedding size
            num_fields (int): number of fields in inputs
            dropout_p (float, optional): dropout probability after factorization machine. Defaults to 0.1.
            output_method (str, optional): output method, Allows: ["concatenate", "sum"]. Defaults to "concatenate".
            output_size (int, optional): ONLY apply on output_method == "concatenate", output size after concatenate. Defaults to 1.
        """
        super(FactorizationMachine, self).__init__()
        if bias:
            # output size = fm output size + linear output size + bias
            self.bias = nn.Parameter(torch.zeros(1))
            nn.init.xavier_uniform_(self.bias.data)
            output_size = embed_size + num_fields + 1
        else:
            # output size = fm output size + linear output size
            self.bias = None
            output_size = embed_size + num_fields
        
        self.fm = FactorizationMachineLayer(dropout_p)
        
        if output_method == "concatenate":
            self.fc = nn.Linear(self.output_size, output_size)
        elif output_method == "sum":
            self.fc = partial(torch.sum, dim=1, keepdim=True)
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        r"""feed forward of Factorization Machine Model 
        
        Args:
            inputs (Dict[str, torch.Tensor]): Dictionary of inputs torch.Tensor
        
        Key-Values:
            first_order, shape = (batch size, num_fields, 1): first order outputs, i.e. outputs from nn.Embedding(vocab_size, 1)
            second_order, shape = (batch size, num_fields, embed_size): second order outputs of one-hot encoding, i.e. outputs from nn.Embedding(vocab_size, embed_size)
        
        Returns:
            torch.Tensor, shape = (batch size, 1 OR output size), dtype = torch.float -- outputs of Factorization Machine
        """
        # get batch size
        batch_size = inputs["first_order"].size(0)

        # first_order's shape = (batch size, number of fields, 1)
        # which could be get by nn.Embedding(vocab size, 1)
        linear_out = inputs["first_order"]

        # second_order's shape = (batch size, number of fields, embed size)
        # which could be get by nn.Embedding(vocab size, embed size)
        fm_out = self.fm(inputs["second_order"])
            
        # reshape fm_out to (batch size, embed size)
        fm_out = fm_out.view(batch_size, -1)

        # reshape linear_out to (batch size, number of fields)
        linear_out = linear_out.view(batch_size, -1)

        # repeat and reshape bias to shape = (batch size, 1)
        if self.bias is not None:
            bias = self.bias.repeat(batch_size).view(batch_size, 1)

        # cat in dim = 1
        # shape = (batch size, number of fields * embed size + number of fields + 1)
        outputs = torch.cat([fm_out, linear_out, self.bias], dim=1)
            
        # fully-connected dense layer for output, return (batch size, output size)
        # or sum with second dimension, return (batch size, 1)
        outputs = self.fc(outputs)

        return outputs
