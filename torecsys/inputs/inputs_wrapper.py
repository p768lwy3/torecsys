from .base import _Inputs
import torch
from typing import Dict, List


class InputsWrapper(_Inputs):
    r"""Inputs class for wrapping a number of Base Inputs class into a dictionary. The output 
    is a dictionary, which its keys are names of model's inputs and values are tensor of model's 
    inputs.
    """
    def __init__(self, 
                 schema: Dict[str, _Inputs]):
        r"""Initialize InputsWrapper.
        
        Args:
            schema (Dict[str, _Inputs]): Schema of InputsWrapper. Dictionary, 
                where keys are names of inputs' fields, and values are tensor of fields. e.g. 
                
                .. code-block:: python
                    
                    import torecsys as trs

                    # initialize embedding layers used in InputsWrapper
                    single_index_emb_0 = trs.inputs.base.SingleIndexEmbedding(2, 8)
                    single_index_emb_1 = trs.inputs.base.SingleIndexEmbedding(2, 8)

                    # set schema, including field names etc
                    single_index_emb_0.set_schema(["userId"])
                    single_index_emb_1.set_schema(["movieId"])

                    # create InputsWrapper
                    schema = {
                        "user"  : single_index_emb_0,
                        "movie" : single_index_emb_1
                    }
                    inputs_wrapper = trs.inputs.InputWrapper(schema=schema)
        
        Attributes:
            schema (Dict[str, _Inputs]): Schema of InputsWrapper.
            length (int): None.
        """
        # refer to parent class
        super(InputsWrapper, self).__init__()
        
        # bind schema to schema
        self.schema = schema

        # add modules in schema to the Module
        for k, inp in schema.items():
            self.add_module(k, inp)

        # set length to None
        self.length = None
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        r"""Forward calculation of InputsWrapper.
        
        Args:
            inputs (Dict[str, T]): Dictionary of inputs, where key is name of input fields, 
                and value is tensor pass to Input class.
            
        Returns:
            Dict[str, T], dtype = torch.float: Output of InputsWrapper, which is a dictionary 
                where keys are names of model's inputs and values are tensor of model's inputs.
        """
        # initialize dictionary to store tensors
        outputs = dict()

        # loop through schema
        for out_name, out_inp in self.schema.items():
            # create inputs in different format if it is ConcatInputs or StackedInputs
            if out_inp.__class__.__name__ in ["ConcatInputs", "StackedInputs"]:
                # create dictionary of concat inputs
                inp_dict = { i : inputs[i] for i in out_inp.schema.inputs }

                # create list variable to be passed 
                inp_args = [inp_dict]
            else:
                # convert list of inputs to tensor, with shape = (B, N, *)
                inp_val = [inputs[i] for i in out_inp.schema.inputs]
                inp_val = torch.cat(inp_val, dim=1)
                inp_args = [inp_val]
            
                # set args for specific input
                if out_inp.__class__.__name__ == "SequenceIndexEmbedding":
                    inp_args.append(inputs[out_inp.schema.lengths])
            
            # calculate embedding values
            output = out_inp(*inp_args)

            # set out_name in outputs to transformed tensors or embedded tensors
            outputs[out_name] = output

        return outputs
    