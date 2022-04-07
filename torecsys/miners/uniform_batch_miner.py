from typing import Dict, Tuple

import torch

from torecsys.miners import BaseMiner


class UniformBatchMiner(BaseMiner):
    """

    """
    def __init__(self, sample_size: int):
        super().__init__()

        self.sample_size = sample_size

    def forward(self, anchor: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]) \
            -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        # randomly generated index of target tensor, shape = (N Neg, )
        any_key = list(target.keys())[0]
        any_value = target[any_key]
        batch_size = any_value.size(0)
        rand_idx = torch.randint(0, batch_size, (self.sample_size * batch_size,))

        # select from the target tensor with the random index, shape = (N Neg, 1, ...,)
        neg_samples = {k: v[rand_idx].unsqueeze(1) for k, v in target.items()}

        # reshape the target tensor and negative tensor to (B, 1, ...) and (B, N Neg, ...) respectively
        pos_samples = {k: v.unsqueeze(1) for k, v in target.items()}
        # neg_samples = torch.cat(torch.chunk(neg_samples, self.sample_size, dim=0), dim=1)

        # check if the shape is not correct then raise error
        # assert anchor.size(0) == pos_samples.size(0)
        # assert anchor.size(0) == neg_samples.size(0)
        pos = {}
        neg = {}
        for k, v in anchor.items():
            pos_v = v.unsqueeze(1)
            neg_v = torch.repeat_interleave(pos_v, self.sample_size, dim=0)

            pos[k] = torch.cat([pos_v, pos_samples[k]], dim=1)
            neg[k] = torch.cat([neg_v, neg_samples[k]], dim=1)

        return pos, neg
