from . import _Inputs
import torch
from typing import Dict, List


class StackedInputs(_Inputs):
    r"""Base Inputs class for stacking of list of Base Inputs class in columnwise. The shape of output 
    is :math:`(B, N_{1} + ... + N_{k}, E)` where :math:`N_{i}` is number of fields of inputs class i.
    """
    def __init__(self, schema: List[tuple]):
        r"""Initialize StackedInputs
        
        Args:
            schema (List[tuple]): Schema of StackedInputs. List of Tuple of Inputs class (i.e. class in \
                trs.inputs.base) and list of string of input fields. e.g. 
                
                .. code-block:: python
                
                    schema = [
                        (trs.inputs.base.SingleIndexEmbedding(4, 10), ["userId"]),
                        (trs.inputs.base.SingleIndexEmbedding(4, 10), ["movieId"])
                    ]
        
        Attributes:
            schema (List[tuple]): Schema of ConcatInputs.
            length (int): Size of embedding tensor.
        
        Raise:
            ValueError: when lengths of inputs are not equal.
        """
        # refer to parent class
        super(StackedInputs, self).__init__()
        
        # bind length to length of the first inputs class in schema
        self.length = len(schema[0][0])

        # check whether lengths of inputs are equal
        if not all(len(tup[0]) == self.length for tup in schema):
            raise ValueError("all inputs lenght (i.e. embed_size) must be same.")

        # bind schema to schema
        self.schema = schema

        # add modules in schema to the Module
        for i, tup in enumerate(schema):
            self.add_module("embedding_%d" % i, tup[0])

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        r"""Foward calculation of StackedInputs
        
        Args:
            inputs (Dict[str, T]): Dictionary of inputs, where key is name of input fields, and value is \
                tensor pass to Input class. Remark: key should exist in schema.
            
        Returns:
            T, shape = (B, N_{sum}, E), dtype = torch.float: Output of StackedInputs, where the values are \
                stacked in the seconnd dimension.
        """
        # initialize list to store tensors temporarily 
        outputs = list()

        # loop through schema 
        for args_tuple in self.schema:
            # get basic args from tuple in schema
            embedding = args_tuple[0]
            inp_names = args_tuple[1]

            # create inputs in different format if the inputs class is ConcatInputs
            if embedding.__class__.__name__ == "ConcatInputs":
                # create dictionary of concat inputs
                args_dict = { i : inputs[i] for i in inp_names }

                # create list variable to be passed 
                args = [args_dict]

            # else, use the same approch for other inputs class
            else:
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

        # stack in the second dimension, and the shape of output = (B, sum(N), E)
        outputs = torch.cat(outputs, dim=1)

        return outputs
