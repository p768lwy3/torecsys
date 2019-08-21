from . import _Inputs
from typing import Dict, List, Tuple

class InputsWrapper(_Inputs):
    def __init__(self, 
                 schema: Dict[str, Tuple[str, _Inputs]]):
        r"""InputsWrapper is a wrapper to concatenate numbers of _Inputs of different fields
        
        Args:
            schema (Dict[str, Tuple[str, _Inputs]]): schema of wrapper
        
        Key-Values:
            output name: tuple of schema, where the 1st element is input name and the 2nd element is the type of _Inputs in torecsys.inputs.base
        """
        super(InputsWrapper, self).__init__()
        # store the schema to stack
        self.schema = schema

        # store the inverse dict to get the output's field name by input's field name
        # raise : one-to-one?
        self.dict_inv = {v[0]: k for k, v in schema.items()}
    
    def forward(self, inputs: Dict[str, Tuple[torch.Tensor]]):
        # init output list to store the tensors
        outputs = []
        for inp_fname, tensor_tup in inputs.item():
            # get the output field name by input field name
            out_fname = self.dict_inv[inp_fname]

            # get the embedding function
            embedding = self.schema[out_fname][1]

            # append the tensors to outputs
            outputs.append(embedding(*tensor_tup))

        return 