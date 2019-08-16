from . import _EmbModule
from ..layers import StarSpaceLayer
from torecsys.models.functional import inner_product_similarity
from torecsys.utils.logging.decorator import jit_experimental
from functools import partial
import torch
from typing import Callable, Dict, Tuple


class StarSpaceModule(_EmbModule):
    r""""""
    def __init__(self, 
                 similarity: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]: partial(inner_product_similarity, dim=2)):
        r"""[summary]
        
        Args:
            similarity (Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optional): function to calculate the similarity between two vectors. Defaults to partial(inner_product_similarity, dim=2).
        """
        super(StarSpaceModule, self).__init__()
        self.starspace = StarSpaceLayer(similarity)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""feed-forward calculation of similarity

        Notations:
            B: batch size
            E: embedding size
            nneg: number of negative samples
        
        Args:
            inputs (Dict[str, torch.Tensor]): dictionary of inputs tensor
        
        Key-Values:
            context, shape = (B, 1, E), dtype = torch.float: embedding vectors of context items
            positive, shape = (B, 1, E), dtype = torch.float: embedding vectors of positive items
            negative, shape = (B, nneg, E), dtype = torch.float: embedding vectors of negative items
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor], shape = (B, 1 or nneg): similarity scores between context and positive or negative samples
        """
        # calculate similarity between context inputs and positive samples
        possim = self.starspace(inputs["context"], inputs["positive"])

        # calculate similarity between context inputs and negative samples
        negsim = self.starspace(inputs["context"], inputs["negative"])
        
        return possim, negsim