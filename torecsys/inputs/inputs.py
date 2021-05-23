from typing import Dict, Optional, TypeVar, Union

import torch
import torch.nn as nn

from torecsys.inputs.base import BaseInput


class Inputs(BaseInput):
    """
    Inputs class for wrapping a number of Base Input class into a dictionary,
    the output is a dictionary, where keys are string and values are torch.tensor.
    """
    Inputs = TypeVar('Inputs')

    def __init__(self, schema: Union[Dict[str, nn.Module], None]):
        """
        Initialize Inputs
        
        Args:
            schema (Dict[str, nn.Module]): schema of Input. Dictionary,
                where keys are names of inputs' fields, and values are tensor of fields

                e.g.
                .. code-block:: python
                    
                    import torecsys as trs

                    # initialize embedding layers used in Inputs
                    single_index_emb_0 = trs.inputs.base.SingleIndexEmbedding(2, 8)
                    single_index_emb_1 = trs.inputs.base.SingleIndexEmbedding(2, 8)

                    # set schema, including field names etc
                    single_index_emb_0.set_schema(["userId"])
                    single_index_emb_1.set_schema(["movieId"])

                    # create Inputs
                    schema = {
                        'user'  : single_index_emb_0,
                        'movie' : single_index_emb_1
                    }
                    inputs = trs.inputs.Inputs(schema=schema)
        
        Attributes:
            schema (Dict[str, nn.Module]): schema of Inputs
        """
        super().__init__()

        self.schema = schema if schema is not None else {}

        for k, emb_fn in self.schema.items():
            self.add_module(k, emb_fn)

        self.length = None

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward calculation of Input
        
        Args:
            inputs (Dict[str, T]): dictionary of inputs, where keys are string and values are torch.tensor.
            
        Returns:
            Dict[str, T], data_type = torch.float: Output of Input, which is a dictionary
                where keys are string and values are torch.tensor.
        """
        outputs = {}

        for k, emb_fn in self.schema.items():
            if emb_fn.__class__.__name__ in ['ConcatInput', 'StackedInput']:
                inp_val = {i: inputs[i] for i in emb_fn.schema.inputs}
                inp_args = [inp_val]
            else:
                inp_val = []

                for emb_k in emb_fn.schema.inputs:
                    v = inputs[emb_k]
                    v = v.unsqueeze(-1) if v.dim() == 1 else v
                    inp_val.append(v)

                inp_val = torch.cat(inp_val, dim=1)
                inp_args = [inp_val]

                if emb_fn.__class__.__name__ == 'SequenceIndexEmbedding':
                    inp_args.append(inputs[emb_fn.schema.lengths])

            outputs[k] = emb_fn(*inp_args)

        return outputs

    def add_inputs(self,
                   name: Optional[str] = None,
                   model: Optional[nn.Module] = None,
                   schema: Optional[Dict[str, nn.Module]] = None) -> Inputs:
        """
        Add new input field to the schema
        
        Args:
            name (str, optional): Name of the input field
            model (nn.Module, optional): torch.nn.Module of the inputs for the input field
            schema (Dict[str, nn.Module], optional): Schema for the inputs of the input field
        
        Raises:
            TypeError: given non-allowed type of schema
            TypeError: given non-allowed type of name
            TypeError: given non-allowed type of model
            AssertionError: given the name of input field is declared in the schema
        
        Returns:
            torecsys.inputs.Inputs: self
        """
        if schema is not None:
            if not isinstance(schema, dict):
                raise TypeError(f'type of schema is not allowed, given {type(schema).__name__}')

            for name, model in schema.items():
                self.add_inputs(name=name, model=model)

        else:
            if not isinstance(name, str):
                raise TypeError(f'type of name is not allowed, given {type(name).__name__}')

            if name in self.schema:
                raise AssertionError(f'Given {name} is defined in the schema.')

            if not isinstance(model, nn.Module):
                raise TypeError(f'type of model is not not allowed, given {type(model).__name__}')

            self.schema.update([(name, model)])
            self.add_module(name, model)

        return self
