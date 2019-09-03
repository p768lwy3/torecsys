import torch
import torch.nn as nn
from typing import Callable


class StarSpaceLayer(nn.Module):
    r"""Layer class of Starspace :title:`Ledell Wu et al, 2017`[1], proposed by Facebook in 2017. It was 
    implemented in C++ originally for a general purpose to embed different kinds of relations between 
    different pairs, like (word, tag), (user-group) etc. Starspace is calculated in the following way: 

    #. calculate similarity between context and positive samples or negative samples 
    
    #. calculate margin ranking loss between similarity of positive samples and those of negative samples 
    
    :Reference:

    #. `Ledell Wu et al, 2017 StarSpace: Embed All The Things! <https://arxiv.org/abs/1709.03856>`

    """
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
    
    def forward(self, samples_inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward calculation of StarSpaceLayer
        
        Args:
            samples_inputs (T), shape = (B, 1 + S, E), dtype = torch.float: Embedded features tensors of context (in [:, 0]) and samples (in [:, 1:]).
        
        Returns:
            T, shape = (B, S), dtype = torch.float: Output of StarSpaceLayer.
        """
        # get context in index 0 of second dimension, where the output's shape = (B, E)
        # hence, .squeeze(1) to change the shape into (B, 1, E)
        context = samples_inputs[:, 0].squeeze(1)

        # get samples after index 1 of second dimension, where the output's shape = (B, N, E)
        samples = samples_inputs[:, 1:]

        # calculate the similarity of context and samples
        outputs = self.similarity(context, samples)
        
        return outputs
