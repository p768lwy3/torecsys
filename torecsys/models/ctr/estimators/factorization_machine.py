from . import _CtrEstimator
from torecsys.models.ctr.modules import FactorizationMachineModule
from torecsys.models.inputs import MultipleIndexEmbedding
import torch
import torch.nn as nn
from typing import List


class FactorizationMachineEstimator(_CtrEstimator):
    r"""FactorizationMachineEstimator is an estimator class which is to feed the fields' indicies
    into the model and return predicted values, with the following calculation:
    :math:`\^{y}(x) := b_{0} + \sum_{i=1}^{n} w_{i} x_{i} + \sum_{i=1}^{n} \sum_{j=1+1}^{n} <v_{i},v_{j}> x_{i} x_{j}` .

    :Reference:

    #. `Steffen Rendle, 2010. Factorization Machine <https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf>`_.

    """
    def __init__(self, 
                 embed_size    : int,
                 field_sizes   : List[int],
                 dropout_p     : float = 0.1,
                 bias          : bool = True,
                 output_method : str = "concatenate",
                 output_size   : int = 1):
        r"""initialize Factorization Machine Estimator
        
        Args:
            embed_size (int): embedding size
            field_sizes (List[int]): list of fields' size, which will also be the offset during lookup
            dropout_p (float, optional): dropout probability after factorization machine. Defaults to 0.1.
            bias (bool, optional): use bias in factorization machine module. Defaults to True.
            output_method (str, optional): output method, Allows: ["concatenate", "sum"]. Defaults to "concatenate".
            output_size (int, optional): ONLY apply on output_method == "concatenate", output size after concatenate. Defaults to 1.
        """
        super(FactorizationMachineEstimator, self).__init__()

        self.model = nn.ModuleDict()
        self.model["first_order"] = MultipleIndexEmbedding(1, field_sizes)
        self.model["second_order"] = MultipleIndexEmbedding(embed_size, field_sizes)
        self.model["fm"] = FactorizationMachineModule(embed_size, len(field_sizes), dropout_p, bias, output_method, output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""feed forward to factorization machine with lists of fields indicies
        
        Args:
            inputs (torch.Tensor), shape = (batch size, number of fields), dtype = torch.long: inputs of fields' indices for each sample as a row
        
        Returns:
            torch.Tensor, shape = (output_size, ), dtype = torch.float: prediction of the model
        """
        emb_inputs = dict()
        emb_inputs["first_order"] = self.model["first_order"](inputs)
        emb_inputs["second_order"] = self.model["second_order"](inputs)

        outputs = self.model["fm"](emb_inputs)
        return outputs
