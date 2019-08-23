from . import _Inputs
import torch
from typing import Dict, List


class StackedInputs(_Inputs):
    r"""StackedInputs is a field of inputs for stacking multiple inputs fields into a (B, N, E) matrix.
    """
    def __init__(self, schema: List[tuple]):
        r"""initialize a stacked inputs field
        
        Args:
            schema (List[tuple]): a list of tuple of embeddings function and inputs arguments, where the lengths of embeddings must be same
            e.g. ```python
            schema = [
                (trs.inputs.base.SingleIndexEmbedding(4, 10), ["userId"]),
                (trs.inputs.base.SingleIndexEmbedding(4, 10), ["movieId"])
            ]```
        """
        super(StackedInputs, self).__init__()
        
        # check whether the lengths of embeddings are equal
        self.length = len(schema[0][0])
        if not all(len(tup[0]) == self.length for tup in schema):
            raise ValueError("all inputs lenght (i.e. embed_size) must be same.")

        # store the schema to self
        self.schema = schema

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        r"""forward process of StackedInputs
        
        Args:
            inputs (Dict[str, T]): inputs values or index to be passed to function in trs.inputs.base, where keys should be existed in schema defined in __init__.
        
        Returns:
            T, shape = (B, N, E): 2nd dimensional stacked features vectors
        """
        outputs = list()

        for args_tuple in self.schema:
            # get basic args
            embedding = args_tuple[0]
            inp_names = args_tuple[1]

            if embedding.__class__.__name__ == "ConcatInputs":
                # create dictionary of concat inputs
                args_dict = { i : inputs[i] for i in inp_names }

                # create list variable to be passed 
                args = [args_dict]
            else:
                # universal inputs' field
                inp_val = [inputs[i] for i in inp_names]
                inp_val = torch.cat(inp_val, dim=1)
                args = [inp_val]

                # append tensor of length to args if SequenceIndexEmbedding
                if embedding.__class__.__name__ == "SequenceIndexEmbedding":
                    arg_name = args_tuple[2][0]
                    args.append(inputs[arg_name])
            
            outputs.append(embedding(*args))

        # stack in the second dimension, hence output's shape = (B, sum(N), E)
        outputs = torch.cat(outputs, dim=1)

        return outputs
