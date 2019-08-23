from . import _Inputs
import torch
from typing import List


class ConcatInputs(_Inputs):
    r"""ConcatInputs is a field of inputs for concatenating several inputs fields on row-wise,
    where the output's shape of ConcatInputs will be :math:`( B, 1, E_{1} + ... + E_{k} )` .
    """
    def __init__(self, schema: List[tuple]):
        r"""initialize a concat inputs field
        
        Args:
            schema (List[tuple]): a list of tuple of embeddings function and inputs arguments,
            e.g. ```python
            schema = [
                (trs.inputs.base.SingleIndexEmbedding(4, 10), ["userId"]),
                (trs.inputs.base.SingleIndexEmbedding(4, 10), ["movieId"])
            ]```
        """
        super(ConcatInputs, self).__init__()

        # store the schema to self
        self.schema = schema

        # set length to sum of lengths of inputs
        self.length = sum([len(tup[0]) for tup in self.schema])
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        r"""forward process of ConcatInputs
        
        Args:
            inputs (Dict[str, T]): inputs values or index to be passed to function in trs.inputs.base, where keys should be existed in schema defined in __init__.
        
        Returns:
            T, shape = (B, 1, E_{sum}), dtype = torch.float: concatenated outputs of values and embeddings values.
        """
        outputs = list()

        for args_tuple in self.schema:
            # get basic args
            embedding = args_tuple[0]
            inp_names = args_tuple[1]
            
            inp_val = [inputs[i] for i in inp_names]
            inp_val = torch.cat(inp_val, dim=1)
            args = [inp_val]
            
            if embedding.__class__.__name__ == "SequenceIndexEmbedding":
                arg_name = args_tuple[2][0]
                args.append(inputs[arg_name])
            
            outputs.append(embedding(*args))

        # concat in the third dimension, hence output's shape = (B, 1, sum(E))
        outputs = torch.cat(outputs, dim=2)
        
        return outputs
        