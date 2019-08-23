from .base import _Inputs
import torch
from typing import Dict, List


class InputsWrapper(_Inputs):
    r"""InputsWrapper is a class to wrap up inputs class into a dictionary, which can be 
    passed to models directly (with correct schema).
    """
    def __init__(self, 
                 schema: Dict[str, tuple]):
        r"""InputsWrapper is a wrapper to concatenate numbers of _Inputs of different fields
        
        Args:
            schema (Dict[str, tuple]): dictionary of schema, where keys are names of output fields and values are tuples of embeddings function and inputs arguments,
            e.g. ```python
            schema = {
                "user"  : (trs.inputs.base.SingleIndexEmbedding(4, 10), ["userId"]),
                "movie" : (trs.inputs.base.SingleIndexEmbedding(4, 10), ["movieId"]),
                "pair"  : (trs.inputs.base.FieldAwareMultipleIndexEmbedding(4, [10, 10]), ["userId", "movieId"]),
                "seq"   : (trs.inputs.base.SequenceIndexEmbedding(4, 10), ["seqId"], ["seqLength"])
            }```
        """
        super(InputsWrapper, self).__init__()
        
        # store the schema to self
        self.schema = schema
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        r"""forward process of InputsWrapper to wrap the inputs into a dictionary
        
        Args:
            inputs (Dict[str, T]): inputs values or index to be passed to function in trs.inputs.base, where keys should be existed in schema defined in __init__.
        
        Returns:
            Dict[str, T]: a dictionary of outputs, where keys are output field name and values are (embedding) values.
        """
        # init outputs dictionary to store outputs' tensor
        outputs = dict()

        # loop through schema for each output field
        for out_name, args_tuple in self.schema.items():
            # get basic args
            embedding = args_tuple[0]
            inp_name = args_tuple[1]
            
            # get tensor of inputs field from inputs
            inp_val = [inputs[i] for i in inp_name]
            
            # cat the tensors of inputs
            inp_val = torch.cat(inp_val, dim=1)
            args = [inp_val]
            
            # get args if type of embedding is the required type
            if embedding.__class__.__name__ == "SequenceIndexEmbedding":
                arg_name = args_tuple[2][0]
                args.append(inputs[arg_name])
            
            elif embedding.__class__.__name__ == "StackedInputs":
                raise ValueError("")

            elif embedding.__class__.__name__ == "ConcatInputs":
                raise ValueError("")
            
            # embedding
            outputs[out_name] = embedding(*args)

        return outputs
    