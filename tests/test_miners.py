import unittest

import torch

from torecsys.miners import *


class UniformBatchMinerTestCase(unittest.TestCase):
    def test_forward(self):
        miner = UniformBatchMiner(sample_size=10)

        anchor = torch.Tensor([1, 2, 3, 4, 5]).unsqueeze(1)
        target = torch.Tensor([5, 4, 3, 2, 1]).unsqueeze(1)

        p, n = miner(anchor, target)
        self.assertNotEqual(p, None)
        self.assertNotEqual(n, None)
        print(f'pos: {p.size()}; neg: {n.size()}')
        print(f'pos: \n{p}\nneg: \n{n}\n')


if __name__ == '__main__':
    unittest.main()
