from . import _EmbModel
from torecsys.layers import StarSpaceLayer
from torecsys.functional import inner_product_similarity
from torecsys.utils.decorator import jit_experimental
from functools import partial
import torch
from typing import Callable, Tuple


class StarSpaceModel(_EmbModel):
    r"""StatSpaceModel"""
    def __init__(self, 
                 similarity: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = partial(inner_product_similarity, dim=2)):
        r"""[summary]
        
        Args:
            similarity (Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optional): function to calculate the similarity between two vectors. Defaults to partial(inner_product_similarity, dim=2).
        """
        super(StarSpaceModel, self).__init__()
        self.starspace = StarSpaceLayer(similarity)

    def forward(self, 
                context_inputs: torch.Tensor, 
                positive_inputs: torch.Tensor, 
                negative_inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""feed-forward calculation of similarity

        Notations:
            B: batch size
            E: embedding size
            nneg: number of negative samples
        
        Args:
            context_inputs (torch.Tensor), shape = (B, 1, E), dtype = torch.float: embedding vectors of context items
            positive_inputs (torch.Tensor), shape = (B, 1, E), dtype = torch.float: embedding vectors of positive items
            negative_inputs (torch.Tensor), shape = (B, nneg, E), dtype = torch.float: embedding vectors of negative items
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor], shape = (B, 1 or nneg): similarity scores between context and positive or negative samples
        """
        # calculate similarity between context inputs and positive samples
        possim = self.starspace(context_inputs, positive_inputs)

        # calculate similarity between context inputs and negative samples
        negsim = self.starspace(context_inputs, negative_inputs)
        
        return possim, negsim