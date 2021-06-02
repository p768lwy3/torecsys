from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn


class BaseMiner(nn.Module, ABC):
    def __init__(self, *args: List[Any], **kwargs: Dict[str, Any]):
        super().__init__()

    @abstractmethod
    def forward(self, anchor: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplemented


__all__ = [
    'BaseMiner',
    'UniformBatchMiner'
]

from torecsys.miners.uniform_batch_miner import UniformBatchMiner
