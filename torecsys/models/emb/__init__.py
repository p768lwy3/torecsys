"""
torecsys.models.ctr is a sub model of the implementation of the whole models of Embedding model
"""
from abc import ABC, abstractmethod

import torch

from torecsys.models import BaseModel


class EmbBaseModel(BaseModel, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def predict(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplemented


from torecsys.models.emb.matrix_factorization import MatrixFactorizationModel
from torecsys.models.emb.starspace import StarSpaceModel
