import torch
import torch.nn as nn
from torecsys.utils.decorator import jit_experimental, no_jit_experimental_by_namedtensor
from typing import Callable


class StarSpaceLayer(nn.Module):
    r"""Layer class of Starspace :title:`Ledell Wu et al, 2017`[1], proposed by Facebook in 2017. 
    It was implemented in C++ originally for a general purpose to embed different kinds of 
    relations between different pairs, like (word, tag), (user-group) etc. Starspace is 
    calculated in the following way: 

    #. calculate similarity between context and positive samples or negative samples 
    
    #. calculate margin ranking loss between similarity of positive samples and those of negative samples 
    
    :Reference:

    #. `Ledell Wu et al, 2017 StarSpace: Embed All The Things! <https://arxiv.org/abs/1709.03856>`

    """
    @no_jit_experimental_by_namedtensor
    def __init__(self, 
                 similarity: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        r"""Initialize StarSpaceLayer
        
        Args:
            similarity (Callable[[T, T], T]): Function of similarity between two tensors. 
                e.g. torch.nn.functional.cosine_similarity.
        
        Attributes:
            similarity (Callable[[T, T], T]): Function of similarity between two tensors. 
        """
        # refer to parent class
        super(StarSpaceLayer, self).__init__()
        
        self.similarity = similarity
    
    def extra_repr(self) -> str:
        """Return information in print-statement of layer.
        
        Returns:
            str: Information of print-statement of layer.
        """
        return 'similarity={}'.format(
            self.similarity.__class__.__name__.lower()
        )
    
    def forward(self, samples_inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward calculation of StarSpaceLayer
        
        Args:
            samples_inputs (T), shape = (B, 2, E), dtype = torch.float: Embedded context and target tensors.
        
        Returns:
            T, shape = (B, S), dtype = torch.float: Output of StarSpaceLayer.
        """
        # get context in index 0 of second dimension, where the output's shape = (B, E)
        ## context = samples_inputs[:, 0, :].unsqueeze(1)
        context = samples_inputs[:, 0, :]
        context = context.unflatten("E", [("N", 1), ("E", context.size("E"))])

        # get target in index 1 of second dimension, where the output's shape = (B, E)
        ## target = samples_inputs[:, 1, :].unsqueeze(1)
        target = samples_inputs[:, 1, :]
        target = target.unflatten("E", [("N", 1), ("E", context.size("E"))])

        # calculate the similarity of context and target
        outputs = self.similarity(context, target)
        outputs.names = ("B", "O")
        
        return outputs
