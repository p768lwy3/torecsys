from typing import Dict, List, Union

import torch
import torch.nn as nn

from torecsys.inputs.base import BaseInput


class StackedInput(BaseInput):
    r"""
    Base Input class for stacking of list of Base Input class in column-wise. The shape of output is
        :math:`(B, N_{1} + ... + N_{k}, E)`,
        where :math:`N_{i}` is number of fields of inputs class i.
    """

    def __init__(self, inputs: List[BaseInput]):
        """
        Initialize StackedInputs
        
        Args:
            inputs (List[BaseInput]): list of input's layers (trs.inputs.Inputs).

                e.g.
                .. code-block:: python
                    
                    import torecsys as trs

                    # initialize embedding layers used in StackedInputs
                    single_index_emb_0 = trs.inputs.base.SingleIndexEmbedding(2, 8)
                    single_index_emb_1 = trs.inputs.base.SingleIndexEmbedding(2, 8)

                    # set schema, including field names etc
                    single_index_emb_0.set_schema(['userId'])
                    single_index_emb_1.set_schema(['movieId'])

                    # create StackedInputs embedding layer
                    inputs = [single_index_emb_0, single_index_emb_1]
                    stacked_emb = trs.inputs.base.StackedInputs(inputs=inputs)
        
        Raise:
            ValueError: when lengths of inputs are not equal.
        """
        super().__init__()

        self.length = len(inputs[0])

        if not all(len(inp) == self.length for inp in inputs):
            raise ValueError('Lengths of inputs, i.e. number of fields or embedding size, must be equal.')

        self.inputs = inputs

        inputs = []
        for idx, inp in enumerate(self.inputs):
            self.add_module(f'Input_{idx}', inp)
            schema = inp.schema
            for arguments in schema:
                if isinstance(arguments, list):
                    inputs.extend(arguments)
                elif isinstance(arguments, str):
                    inputs.append(arguments)

        self.set_schema(inputs=list(set(inputs)))

    def __getitem__(self, idx: Union[int, slice, str]) -> Union[nn.Module, List[nn.Module]]:
        """
        Get Embedding Layer by index of the schema
        
        Args:
            idx (Union[int, slice, str]): index to get embedding layer from the schema
        
        Returns:
            Union[nn.Module, List[nn.Module]]: embedding layer(s) of the given index
        """
        if isinstance(idx, int):
            emb_layers = self.inputs[idx]
        elif isinstance(idx, slice):
            start = idx.start if idx.start is not None else 0
            stop = idx.stop if idx.stop is not None else len(self.schema)
            step = idx.step if idx.step is not None else 1

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
        Forward calculation of StackedInput
        
        Args:
            inputs (Dict[str, T]): dictionary of inputs, where key is name of input fields,
                and value is tensor pass to Input class. Remark: key should exist in schema.
            
        Returns:
            T, shape = (B, N_{sum}, E), data_type = torch.float: output of StackedInput,
                where the values are stacked in the second dimension.
        """
        outputs = []

        for inp in self.inputs:
            if inp.__class__.__name__ == 'ConcatInput':
                inp_dict = {i: inputs[i] for i in inp.schema.inputs}
                inp_args = [inp_dict]
            else:
                # convert list of inputs to tensor, with shape = (B, N, *)
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
                output = output.unflatten('E', (('N', 1,), ('E', output.size('E'),),))

            outputs.append(output)

        # stack in the second dimension
        # the shape of output = (B, sum(N), E)
        outputs = torch.cat(outputs, dim='N')

        return outputs
