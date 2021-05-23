from typing import Dict, List, Union

import torch
import torch.nn as nn

from torecsys.inputs.base import BaseInput


class ConcatInput(BaseInput):
    """
    Base Input class for concatenation of list of Base Input class in row-wise.
    The shape of output is :math:`(B, 1, E_{1} + ... + E_{k})`, where :math:`E_{i}` 
    is embedding size of :math:`i-th` field. 
    """

    def __init__(self, inputs: List[BaseInput]):
        """
        Initialize ConcatInput
        
        Args:
            inputs (List[_Inputs]): List of input's layers (trs.inputs.base._Inputs),
                i.e. class of trs.inputs.base

                e.g.
                .. code-block:: python
                    
                    import torecsys as trs

                    # initialize embedding layers used in ConcatInput
                    single_index_emb_0 = trs.inputs.base.SingleIndexEmbedding(2, 8)
                    single_index_emb_1 = trs.inputs.base.SingleIndexEmbedding(2, 8)

                    # set schema, including field names etc
                    single_index_emb_0.set_schema(['userId'])
                    single_index_emb_1.set_schema(['movieId'])

                    # create ConcatInput embedding layer
                    inputs = [single_index_emb_0, single_index_emb_1]
                    concat_emb = trs.inputs.base.ConcatInput(inputs=inputs)
        """
        super().__init__()

        self.inputs = inputs
        self.length = sum([len(inp) for inp in self.inputs])

        inputs = []
        for idx, inp in enumerate(self.inputs):
            self.add_module(f'input_{idx}', inp)

            schema = inp.schema

            for arguments in schema:
                if isinstance(arguments, list):
                    inputs.extend(arguments)
                elif isinstance(arguments, str):
                    inputs.append(arguments)

        self.set_schema(inputs=list(set(inputs)))

    def __getitem__(self, idx: Union[int, slice, str]) -> Union[nn.Module, List[nn.Module]]:
        """
        Get Embedding Layer by index from inputs
        
        Args:
            idx (Union[int, slice, str]): index to get embedding layer from the schema
        
        Returns:
            Union[nn.Module, List[nn.Module]]: embedding layer(s) of the given index
        """
        if isinstance(idx, int):
            emb_layers = self.inputs[idx]
        elif isinstance(idx, slice):
            start = idx.start if idx.start else 0
            stop = idx.stop if idx.stop else len(self.inputs)
            step = idx.step if idx.step else 1

            emb_layers = []
            for i in range(start, stop, step):
                emb_layers.append(self.inputs[i])

        elif isinstance(idx, str):
            emb_layers = []
            for inp in self.inputs:
                if idx in inp.schema.inputs:
                    emb_layers.append(inp)
        else:
            raise ValueError('__getitem__ only accept int, slice, and str.')

        return emb_layers

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward calculation of ConcatInput
        
        Args:
            inputs (Dict[str, T]): dictionary of inputs, where key is name of input fields,
                and value is tensor pass to Input class
        
        Returns:
            T, shape = (B, 1, E_{sum}), data_type = torch.float: output of ConcatInput, where
                the values are concatenated in the third dimension
        """
        outputs = []

        for inp in self.inputs:
            inp_val = []
            for k in inp.schema.inputs:
                v = inputs[k]
                v = v.unsqueeze(-1) if v.dim() == 1 else v
                inp_val.append(v)
            inp_val = torch.cat(inp_val, dim=1)
            inp_args = [inp_val]

            if inp.__class__.__name__ == 'SequenceIndexEmbedding':
                inp_args.append(inputs[inp.schema.lengths])

            output = inp(*inp_args)

            if output.dim() < 3:
                output = output.unflatten('E', (('N', 1,), ('E', output.size('E')),), )

            outputs.append(output)

        # concat in the third dimension, and the shape of output = (B, 1, sum(E))
        outputs = torch.cat(outputs, dim='E')

        return outputs
