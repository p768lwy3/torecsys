from typing import Tuple

import torch

from torecsys.miners import BaseMiner


class UniformBatchMiner(BaseMiner):
    def __init__(self, sample_size: int):
        super().__init__()

        self.sample_size = sample_size

    def forward(self, anchor: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # randomly generated index of target tensor, shape = (N Neg, )
        batch_size = target.size(0)
        rand_idx = torch.randint(0, batch_size, (self.sample_size * batch_size,))

        # select from the target tensor with the random index, shape = (N Neg, 1, ...,)
        neg_samples = target[rand_idx].unsqueeze(1)

        # reshape the target tensor and negative tensor to (B, 1, ...) and (B, N Neg, ...) respectively
        pos_samples = target.unsqueeze(1)
        # neg_samples = torch.cat(torch.chunk(neg_samples, self.sample_size, dim=0), dim=1)

        # check if the shape is not correct then raise error
        # assert anchor.size(0) == pos_samples.size(0)
        # assert anchor.size(0) == neg_samples.size(0)
        anchor = anchor.unsqueeze(1)
        repeated = torch.repeat_interleave(anchor, self.sample_size, dim=0)

        pos = torch.cat([anchor, pos_samples], dim=1)
        neg = torch.cat([repeated, neg_samples], dim=1)

        return pos, neg
