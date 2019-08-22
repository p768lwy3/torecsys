from . import _Inputs
import torch
from typing import List


class ConcatInputs(_Inputs):
    r"""
    """
    def __init__(self, schema: List[tuple]):
        
        super(ConcatInputs, self).__init__()
        # store the schema to self
        self.schema = schema
    
    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:

        outputs = list()

        for args_tuple in self.schema:
            # get basic args
            embedding = args_tuple[0]
            inp_name = args_tuple[1]
            