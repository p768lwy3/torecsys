from functools import partial
import torch
import torch.nn as nn
from torecsys.utils.decorator import jit_experimental, no_jit_experimental_by_namedtensor
from typing import Callable

class StarSpaceLayer(nn.Module):
    r"""Layer class of Starspace.
    
    Starpspace by Ledell Wu et al, 2017 is proposed by Facebook in 2017. 

    It was implemented in C++ originally for a general purpose to embed different kinds of 
    relations between different pairs, like (word, tag), (user-group) etc. Starspace is 
    calculated in the following way: 

    #. calculate similarity between context and positive samples or negative samples 
    
    #. calculate margin ranking loss between similarity of positive samples and those of negative 
    samples 
    
    :Reference:

    #. `Ledell Wu et al, 2017 StarSpace: Embed All The Things! <https://arxiv.org/abs/1709.03856>`_.

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
        # Refer to parent class
        super(StarSpaceLayer, self).__init__()
        
        # Bind similarity to similarity
        self.similarity = similarity
    
    def extra_repr(self) -> str:
        """Return information in print-statement of layer.
        
        Returns:
            str: Information of print-statement of layer.
        """
        if (self.similarity, partial):
            similarity_cls = self.similarity.func.__qualname__.split(".")[-1].lower()
        else:
            similarity_cls = self.similarity.__class__.__name__.lower()
        return 'similarity={}'.format(similarity_cls)
    
    def forward(self, samples_inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward calculation of StarSpaceLayer
        
        Args:
            samples_inputs (T), shape = (B, N = 2, E), dtype = torch.float: Embedded features tensors of context and target.
        
        Returns:
            T, shape = (B, E), dtype = torch.float: Output of StarSpaceLayer.
        """
        # Get and reshape feature tensors of context
        # inputs: samples_inputs, shape = (B, N = 2, E)
        # output: context, shape = (B, N = 1, E)
        context = samples_inputs[:, 0, :]
        context = context.unflatten("E", [("N", 1), ("E", context.size("E"))])

        # Get and reshape feature tensors of target
        # inputs: samples_inputs, shape = (B, N = 2, E)
        # output: target, shape = (B, N = 1, E)
        target = samples_inputs[:, 1, :]
        target = target.unflatten("E", [("N", 1), ("E", context.size("E"))])

        # Calculate similarity bewteen context and target
        # inputs: context, shape = (B, N = 1, E)
        # inputs: target, shape = (B, N = 1, E)
        # output: outputs, shape = (B, O = E)
        context = context.rename(None)
        target = target.rename(None)
        outputs = self.similarity(context, target)
        outputs.names = ("B", "O")
        
        return outputs
