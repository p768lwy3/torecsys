from . import _Inputs
import torch
from torecsys.utils.decorator import jit_experimental, no_jit_experimental_by_namedtensor
from typing import Dict, List


class StackedInputs(_Inputs):
    r"""Base Inputs class for stacking of list of Base Inputs class in columnwise. The shape of output 
    is :math:`(B, N_{1} + ... + N_{k}, E)` where :math:`N_{i}` is number of fields of inputs class i.
    """
    @no_jit_experimental_by_namedtensor
    def __init__(self, inputs: List[_Inputs]):
        r"""Initialize StackedInputs
        
        Args:
            inputs (List[_Inputs]): List of input's layers (trs.inputs.base._Inputs), 
                i.e. class of trs.inputs.base. e.g. 
                
                .. code-block:: python
                    import torecsys as trs

                    # initialize embedding layers used in StackedInputs
                    single_index_emb_0 = trs.inputs.base.SingleIndexEmbedding(2, 8)
                    single_index_emb_1 = trs.inputs.base.SingleIndexEmbedding(2, 8)

                    # set schema, including field names etc
                    single_index_emb_0.set_schema(["userId"])
                    single_index_emb_1.set_schema(["movieId"])

                    # create StackedInputs embedding layer
                    inputs = [single_index_emb_0, single_index_emb_1]
                    stacked_emb = trs.inputs.base.StackedInputs(inputs=inputs)
        
        Attributes:
            inputs (List[_Inputs]): List of input's layers.
            length (int): Size of embedding tensor.
        
        Raise:
            ValueError: when lengths of inputs are not equal.
        """
        # refer to parent class
        super(StackedInputs, self).__init__()
        
        # bind length to length of the first input's class in inputs,
        # i.e. number of fields of inputs, or embedding size of embedding.
        self.length = len(inputs[0])

        # check whether lengths of inputs are equal
        if not all(len(inp) == self.length for inp in inputs):
            raise ValueError("Lengths of inputs, " + 
                "i.e. number of fields or embeding size, must be equal.")

        # bind inputs to inputs
        self.inputs = inputs

        # add modules from inputs to this module
        inputs = []
        for idx, inp in enumerate(self.inputs):
            # add module
            self.add_module("Input_%d" % idx, inp)

            # append fields name to the list `inputs`
            schema = inp.schema
            for arguments in schema:
                if isinstance(arguments, list):
                    inputs.extend(arguments)
                elif isinstance(arguments, str):
                    inputs.append(arguments)

        self.set_schema(inputs=list(set(inputs)))

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

        # loop through inputs 
        for inp in self.inputs:
            # get schema, i.e. input's field names, from input in list
            inp_names = inp.schema.inputs

            # create inputs in different format if the inputs class is ConcatInputs
            if inp.__class__.__name__ == "ConcatInputs":
                # create dictionary of concat inputs
                inp_dict = { i : inputs[i] for i in inp_names }

                # create list variable to be passed 
                inp_args = [inp_dict]

            # else, use the same approch for other inputs class
            else:
                # convert list of inputs to tensor, with shape = (B, N, *)
                inp_val = [inputs[i] for i in inp_names]
                inp_val = torch.cat(inp_val, dim=1)
                inp_args = [inp_val]

                # set args for specific input
                if inp.__class__.__name__ == "SequenceIndexEmbedding":
                    inp_names = inp.schema.lengths
                    inp_args.append(inputs[inp_names])
            
            # calculate embedding values
            output = inp(*inp_args)

            # check if output dimension is less than 3, then .unsqueeze(1)
            if output.dim() < 3:
                output = output.unflatten("E", [("N", 1), ("E", output.size("E"))])
            
            # append tensor to outputs
            outputs.append(output)

        # stack in the second dimension, and the shape of output = (B, sum(N), E)
        outputs = torch.cat(outputs, dim=1)

        return outputs
