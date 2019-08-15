from . import _CtrEstimator
from torecsys.models.ctr.modules import FactorizationMachine
from torecsys.models.inputs import MultipleIndexEmbedding
import torch
import torch.nn as nn


class FactorizationMachineEstimator(_CtrEstimator):
    r"""[summary]
    """
    def __init__(self, 
                 embed_size    : int,
                 field_sizes   : List[int],
                 dropout_p     : float = 0.1,
                 bias          : bool = True,
                 output_method : str = "concatenate",
                 output_size   : int = 1):
        r"""[summary]
        
        Args:
            embed_size (int): [description]
            field_sizes (List[int]): [description]
            dropout_p (float, optional): [description]. Defaults to 0.1.
            bias (bool, optional): [description]. Defaults to True.
            output_method (str, optional): [description]. Defaults to "concatenate".
            output_size (int, optional): [description]. Defaults to 1**kwargs.
        """
        super(FactorizationMachineEstimator, self).__init__()

        self.model = nn.ModuleDict()
        self.model["first_order"] = MultipleIndexEmbedding(embed_size, field_sizes)
        self.model["second_order"] = MultipleIndexEmbedding(1, field_sizes)
        self.model["fm"] = FactorizationMachine(embed_size, len(field_sizes), dropout_p, bias, output_method, output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""[summary]
        
        Args:
            inputs (torch.Tensor): [description]
        
        Returns:
            torch.Tensor: [description]
        """
        emb_inputs = dict()
        emb_inputs["first_order"] = self.model["first_order"](inputs)
        emb_inputs["second_order"] = self.model["second_order"](inputs)

        outputs = self.model["fm"](emb_inputs)
        return outputs
