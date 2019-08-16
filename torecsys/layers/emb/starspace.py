import torch
import torch.nn as nn
from typing import Callable


class StarSpaceLayer(nn.Module):
    r"""StarSpace is a general purpose embedding algorithm proposed by Facebook in 2017, and it 
    is implemented in C++ originally. As a general purpose model, StarSpace can embed different 
    kinds of relations between different pairs, like (word, tag), (user-group) etc. StarSpace 
    is calculated in the following way: 
    
    #. calculate similarity between context and positive samples or negative samples 
    
    #. calculate margin ranking loss between similarity of positive samples and those of negative samples 
    
    :Reference:

    #. `Ledell Wu et al, 2017 StarSpace: Embed All The Things! <https://arxiv.org/abs/1709.03856>`

    """
    def __init__(self, 
                 similarity: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        r"""initialize a StarSpace Layer of similartiy calculation
        
        Args:
            similarity (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): function to calculate similarity between two vectors, e.g. F.cosine_similarity.
        """
        super(StarSpaceLayer, self).__init__()
        self.similarity = similarity
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""feed forward of StatSpace Embedding
        
        Args:
            inputs (torch.Tensor), shape = (batch size, 1 + (1 or number of samples), embedding size), dtype = torch.float: stacked tensors of context (in [:, 0]) and samples (in [1:, 0])
        
        Returns:
            torch.Tensor, shape = (batch size, number of samples), dtype = torch.float: similarity between context and samples
        """
        # get context in index 0 of second dimension, where the output's shape = (batch size, embedding size)
        # hence, .squeeze(1) to change the shape into (batch size, 1, embedding size)
        context = inputs[:, 0].squeeze(1)

        # get samples after index 1 of second dimension, where the output's shape = (batch size, num samples, embedding size)
        samples = inputs[:, 1:]

        # calculate the similarity of context and samples
        outputs = self.similarity(context, samples)
        
        return outputs
