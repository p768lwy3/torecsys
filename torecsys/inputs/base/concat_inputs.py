from . import _Inputs
from torecsys.utils.decorator import jit_experimental
import torch
from typing import Dict, List


class ConcatInputs(_Inputs):
    r"""Base Inputs class for concatenation of list of Base Inputs class in rowwise. The shape of output 
    is :math:`(B, 1, E_{1} + ... + E_{k})`, where :math:`E_{i}` is embedding size of :math:`i-th` field. 
    """
    @jit_experimental
    def __init__(self, schema: List[tuple]):
        r"""Initialize ConcatInputs.
        
        Args:
            schema (List[tuple]): Schema of ConcatInputs. List of Tuple of Inputs class (i.e. class in 
                trs.inputs.base) and list of string of input fields. e.g. 
                
                .. code-block:: python

                    schema = [
                        (trs.inputs.base.SingleIndexEmbedding(4, 10), ["userId"]),
                        (trs.inputs.base.SingleIndexEmbedding(4, 10), ["movieId"])
                    ]
        
        Attributes:
            schema (List[tuple]): Schema of ConcatInputs.
            length (int): Sum of length of inputs (i.e. number of fields of inputs, or embedding size of 
                embedding) in schema.
        """
        # refer to parent class
        super(ConcatInputs, self).__init__()

        # bind schema to schema
        self.schema = schema

        # add modules in schema to the Module
        for i, tup in enumerate(schema):
            self.add_module("embedding_%d" % i, tup[0])

        # bind length to sum of lengths of inputs (i.e. number of fields of inputs, or embedding 
        # size of embedding) 
        self.length = sum([len(tup[0]) for tup in self.schema])
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        r"""Foward calculation of ConcatInputs.
        
        Args:
            inputs (Dict[str, T]): Dictionary of inputs, where key is name of input fields, and value is 
                tensor pass to Input class. Remark: key should exist in schema.
        
        Returns:
            T, shape = (B, 1, E_{sum}), dtype = torch.float: Output of ConcatInputs, where the values are
                concatenated in the third dimension.
        """
        # initialize list to store tensors temporarily 
        outputs = list()

        # loop through schema 
        for args_tuple in self.schema:
            # get basic args from tuple in schema
            embedding = args_tuple[0]
            inp_names = args_tuple[1]
            
            # convert list of inputs to tensor, with shape = (B, N, *)
            inp_val = [inputs[i] for i in inp_names]
            inp_val = torch.cat(inp_val, dim=1)
            args = [inp_val]
            
            # set args for specific input
            if embedding.__class__.__name__ == "SequenceIndexEmbedding":
                arg_name = args_tuple[2][0]
                args.append(inputs[arg_name])
            
            # append tensor to outputs
            outputs.append(embedding(*args))

        # concat in the third dimension, and the shape of output = (B, 1, sum(E))
        outputs = torch.cat(outputs, dim=2)
        
        return outputs
        