import unittest

import torch

from torecsys.miners import *


class UniformBatchMinerTestCase(unittest.TestCase):
    def test_forward(self):
        miner = UniformBatchMiner(sample_size=10)

        anchor = {
            'feat_inp': torch.Tensor([1, 2, 3, 4, 5]).unsqueeze(1),
            'emb_inp': torch.Tensor([6, 7, 8, 9, 10]).unsqueeze(1)
        }
        target = {
            'feat_inp': torch.Tensor([5, 4, 3, 2, 1]).unsqueeze(1),
            'emb_inp': torch.Tensor([10, 9, 8, 7, 6]).unsqueeze(1)
        }

        p, n = miner(anchor, target)
        self.assertNotEqual(p['feat_inp'], None)
        self.assertNotEqual(n['feat_inp'], None)
        print(f'pos: {p["feat_inp"].size()}; neg: {n["feat_inp"].size()}')
        print(f'pos: \n{p}\nneg: \n{n}\n')


if __name__ == '__main__':
    unittest.main()
