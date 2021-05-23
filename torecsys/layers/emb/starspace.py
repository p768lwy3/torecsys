from typing import Callable, Dict, Tuple

import torch

from torecsys.layers import BaseLayer


class StarSpaceLayer(BaseLayer):
    """
    Layer class of Starspace.
    
    StarSpace by Ledell Wu et al, 2017 is proposed by Facebook in 2017.

    It was implemented in C++ originally for a general purpose to embed different kinds of 
    relations between different pairs, like (word, tag), (user-group) etc. Starspace is 
    calculated in the following way: 

    #. calculate similarity between context and positive samples or negative samples 
    
    #. calculate margin ranking loss between similarity of positive samples and those of negative 
    samples 
    
    :Reference:

    #. `Ledell Wu et al, 2017 StarSpace: Embed All The Things! <https://arxiv.org/abs/1709.03856>`_.

    """

    @property
    def inputs_size(self) -> Dict[str, Tuple[str, ...]]:
        return {
            'inputs': ('B', '2', 'E',)
        }

    @property
    def outputs_size(self) -> Dict[str, Tuple[str, ...]]:
        return {
            'outputs': ('B', 'E',)
        }

    def __init__(self,
                 similarity: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        """
        Initialize StarSpaceLayer
        
        Args:
            similarity (Callable[[T, T], T]): function of similarity between two tensors.
                e.g. torch.nn.functional.cosine_similarity
        """
        super().__init__()

        self.similarity = similarity

    def extra_repr(self) -> str:
        """
        Return information in print-statement of layer.
        
        Returns:
            str: information of print-statement of layer.
        """
        return f'similarity={self.similarity.__qualname__.split(".")[-1].lower()}'

    def forward(self, samples_inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward calculation of StarSpaceLayer
        
        Args:
            samples_inputs (T), shape = (B, N = 2, E), data_type = torch.float: embedded features tensors of context
                and target
        
        Returns:
            T, shape = (B, E), data_type = torch.float: output of StarSpaceLayer
        """
        # Name the inputs tensor for alignment
        samples_inputs.names = ('B', 'N', 'E',)

        # Get and reshape feature tensors of context
        # inputs: samples_inputs, shape = (B, N = 2, E)
        # output: context, shape = (B, N = 1, E)
        context = samples_inputs[:, 0, :]
        context = context.unflatten('E', (('N', 1,), ('E', context.size('E'),),))

        # Get and reshape feature tensors of target
        # inputs: samples_inputs, shape = (B, N = 2, E)
        # output: target, shape = (B, N = 1, E)
        target = samples_inputs[:, 1, :]
        target = target.unflatten("E", [("N", 1), ("E", context.size("E"))])

        # Calculate similarity between context and target
        # inputs: context, shape = (B, N = 1, E)
        # inputs: target, shape = (B, N = 1, E)
        # output: outputs, shape = (B, O = E)
        context = context.rename(None)
        target = target.rename(None)
        outputs = self.similarity(context, target)
        outputs.names = ("B", "O")

        return outputs
