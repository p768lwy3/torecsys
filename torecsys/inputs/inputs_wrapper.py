from .base import _Inputs
import torch
import torch.nn as nn
from typing import Dict, List, TypeVar, Union

class InputsWrapper(_Inputs):
    r"""Inputs class for wrapping a number of Base Inputs class into a dictionary. The output 
    is a dictionary, which its keys are names of model's inputs and values are tensor of model's 
    inputs.
    """
    def __init__(self, 
                 schema: Union[Dict[str, _Inputs], None]):
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
        if schema is None:
            self.schema = dict()
        else:
            self.schema = schema

        # add modules in schema to the Module
        for k, inp in self.schema.items():
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
        # Initialize dictionary to store tensors
        outputs = dict()

        # Loop through schema
        for out_name, out_inp in self.schema.items():
            # Create inputs in different format if it is ConcatInputs or StackedInputs
            if out_inp.__class__.__name__ in ["ConcatInputs", "StackedInputs"]:
                # Create dictionary of concat inputs
                inp_dict = { i : inputs[i] for i in out_inp.schema.inputs }

                # Create list variable to be passed 
                inp_args = [inp_dict]
            else:
                # Convert list of inputs to tensor, with shape = (B, N, ...)
                inp_val = [inputs[i] for i in out_inp.schema.inputs]
                inp_val = torch.cat(inp_val, dim=1)
                inp_args = [inp_val]
            
                # Set args for specific input
                if out_inp.__class__.__name__ == "SequenceIndexEmbedding":
                    inp_args.append(inputs[out_inp.schema.lengths])
            
            # Calculate forwardly with module
            output = out_inp(*inp_args)

            # set out_name in outputs to transformed tensors or embedded tensors
            outputs[out_name] = output

        return outputs
    
    def add_inputs(self,
                   name: str = None, 
                   module: nn.Module = None,
                   schema: Dict[str, nn.Module] = None) -> TypeVar("InputsWrapper"):
        r"""[summary]
        
        Args:
            name (str, optional): [description]
            module (nn.Module, optional): [description]
            schema (Dict[str, nn.Module], optional): [description]
        
        Raises:
            TypeError: [description]
            TypeError: [description]
            TypeError: [description]
            AssertionError: [description]
        
        Returns:
            torecsys.inputs.InputsWrapper: self
        """
        if schema is not None:
            if not isinstance(schema, dict):
                raise TypeError(f"{type(schema).__name__} not allowed for schema.")
            
            for name, module in schema.items():
                self.add_inputs(name=name, module=module)
        
        else:
            if not isinstance(name, str):
                raise TypeError(f"{type(name).__name__} not allowed for name.")
        
            if not isinstance(module, nn.Module):
                raise TypeError(f"{type(module).__name__} not allowed for module.")

            if name in self.schema:
                raise AssertionError(f"{name} is defined in the schema.")
        
            self.schema.update({ name : module })
            self.add_module(name, module)

        return self
