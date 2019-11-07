from . import _Inputs
import torch
import torch.nn as nn
from torecsys.utils.decorator import jit_experimental, no_jit_experimental_by_namedtensor
from typing import Dict, List, Union

class ConcatInputs(_Inputs):
    r"""Base Inputs class for concatenation of list of Base Inputs class in rowwise. 
    The shape of output is :math:`(B, 1, E_{1} + ... + E_{k})`, where :math:`E_{i}` 
    is embedding size of :math:`i-th` field. 
    """
    @no_jit_experimental_by_namedtensor
    def __init__(self, inputs: List[_Inputs]):
        r"""Initialize ConcatInputs.
        
        Args:
            inputs (List[_Inputs]): List of input's layers (trs.inputs.base._Inputs), 
                i.e. class of trs.inputs.base. e.g. 
                
                .. code-block:: python
                    
                    import torecsys as trs

                    # initialize embedding layers used in ConcatInputs
                    single_index_emb_0 = trs.inputs.base.SingleIndexEmbedding(2, 8)
                    single_index_emb_1 = trs.inputs.base.SingleIndexEmbedding(2, 8)

                    # set schema, including field names etc
                    single_index_emb_0.set_schema(["userId"])
                    single_index_emb_1.set_schema(["movieId"])

                    # create ConcatInputs embedding layer
                    inputs = [single_index_emb_0, single_index_emb_1]
                    concat_emb = trs.inputs.base.ConcatInputs(inputs=inputs)
        
        Attributes:
            inputs (List[_Inputs]): List of input's layers.
            length (int): Sum of length of input's layers, 
                i.e. number of fields of inputs, or embedding size of embedding.
        """
        # refer to parent class
        super(ConcatInputs, self).__init__()

        # bind inputs to inputs
        self.inputs = inputs

        # add schemas and modules from inputs to this module
        inputs = []
        for idx, inp in enumerate(self.inputs):
            # add module 
            self.add_module("input_%d" % idx, inp)

            # append fields name to the list `inputs`
            schema = inp.schema
            for arguments in schema:
                if isinstance(arguments, list):
                    inputs.extend(arguments)
                elif isinstance(arguments, str):
                    inputs.append(arguments)
        
        self.set_schema(inputs=list(set(inputs)))

        # bind length to sum of lengths of inputs,
        # i.e. number of fields of inputs, or embedding size of embedding.
        self.length = sum([len(inp) for inp in self.inputs])
    
    def __getitem__(self, idx: Union[int, slice, str]) -> Union[nn.Module, List[nn.Module]]:
        """Get Embedding Layer by index from inputs.
        
        Args:
            idx (Union[int, slice, str]): index to get embedding layer from the schema.
        
        Returns:
            Union[nn.Module, List[nn.Module]]: Embedding layer(s) of the given index
        """
        if isinstance(idx, int):
            emb_layers = self.inputs[idx]

        elif isinstance(idx, slice):
            emb_layers = []
            
            # parse the slice object into integers used in range()
            start = idx.start if idx.start is not None else 0
            stop = idx.stop if idx.stop is not None else len(self.inputs)
            step = idx.step if idx.step is not None else 1

            for i in range(start, stop, step):
                emb_layers.append(self.inputs[i])

        elif isinstance(idx, str):
            emb_layers = []
            for inp in self.inputs:
                if idx in inp.schema.inputs:
                    emb_layers.append(inp)
        
        else:
            raise ValueError("getitem only accept int, slice, and str.")
        
        return emb_layers
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        r"""Foward calculation of ConcatInputs.
        
        Args:
            inputs (Dict[str, T]): Dictionary of inputs, where key is name of input fields, 
                and value is tensor pass to Input class.
        
        Returns:
            T, shape = (B, 1, E_{sum}), dtype = torch.float: Output of ConcatInputs, where 
                the values are concatenated in the third dimension.
        """
        # initialize list to store tensors temporarily 
        outputs = list()

        # loop through inputs 
        for inp in self.inputs:
            # convert list of inputs to tensor, with shape = (B, N, *)
            inp_val = [inputs[i] for i in inp.schema.inputs]
            inp_val = torch.cat(inp_val, dim="N")
            inp_args = [inp_val]
            
            # set args for specific input
            if inp.__class__.__name__ == "SequenceIndexEmbedding":
                inp_args.append(inputs[inp.schema.lengths])
            
            # calculate embedding values
            output = inp(*inp_args)

            # check if output dimension is less than 3, then .unsqueeze(1)
            if output.dim() < 3:
                output = output.unflatten("E", [("N", 1), ("E", output.size("E"))])

            # append tensor to outputs
            outputs.append(output)

        # concat in the third dimension, and the shape of output = (B, 1, sum(E))
        outputs = torch.cat(outputs, dim="E")
        
        return outputs
        