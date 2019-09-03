from .base import _Inputs
import torch
from typing import Dict, List


class InputsWrapper(_Inputs):
    r"""Inputs class for wrapping a number of Base Inputs class into a dictionary. The output is a 
    dictionary, which its keys are names of model's inputs and values are tensor of model's inputs.
    """
    def __init__(self, 
                 schema: Dict[str, tuple]):
        r"""Initialize InputsWrapper.
        
        Args:
            schema (Dict[str, tuple]): Schema of InputsWrapper. Dictionary, which keys are names of 
                inputs' fields and values are tensor of those fields. e.g. 
                
                .. code-block:: python
                
                    schema = {
                        "user"  : (trs.inputs.base.SingleIndexEmbedding(4, 10), ["userId"]),
                        "movie" : (trs.inputs.base.SingleIndexEmbedding(4, 10), ["movieId"]),
                        "pair"  : (trs.inputs.base.FieldAwareMultipleIndexEmbedding(4, [10, 10]), ["userId", "movieId"]),
                        "seq"   : (trs.inputs.base.SequenceIndexEmbedding(4, 10), ["seqId"], ["seqLength"])
                    }
        
        Attributes:
            schema (Dict[str, tuple]): Schema of InputsWrapper.
            length (int): None.
        """
        # refer to parent class
        super(InputsWrapper, self).__init__()
        
        # bind schema to schema
        self.schema = schema

        # add modules in schema to the Module
        for k, tup in schema.items():
            self.add_module(k, tup[0])

        # set length to None
        self.length = None
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        r"""Forward calculation of InputsWrapper.
        
        Args:
            inputs (Dict[str, T]): Dictionary of inputs, where key is name of input fields, and value is \
                tensor pass to Input class. Remark: key should exist in schema.
            
        Returns:
            Dict[str, T], dtype = torch.float: Output of InputsWrapper, which is a dictionary where keys \
                are names of model's inputs and values are tensor of model's inputs.
        """
        # initialize dictionary to store tensors
        outputs = dict()

        # loop through schema
        for out_name, args_tuple in self.schema.items():
            # get basic args from tuple in schema
            embedding = args_tuple[0]
            inp_names = args_tuple[1]
            
            # create inputs in different format if the inputs class is ConcatInputs or StackedInputs
            if embedding.__class__.__name__ in ["ConcatInputs", "StackedInputs"]:
                # create dictionary of concat inputs
                args_dict = { i : inputs[i] for i in inp_names }

                # create list variable to be passed 
                args = [args_dict]
            else:
                # convert list of inputs to tensor, with shape = (B, N, *)
                inp_val = [inputs[i] for i in inp_names]
                inp_val = torch.cat(inp_val, dim=1)
                args = [inp_val]
            
                # set args for specific input
                if embedding.__class__.__name__ == "SequenceIndexEmbedding":
                    arg_name = args_tuple[2][0]
                    args.append(inputs[arg_name])

            # set out_name in outputs to transformed tensors or embedded tensors
            outputs[out_name] = embedding(*args)

        return outputs
    