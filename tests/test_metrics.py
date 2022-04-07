import random
import unittest

import torch

from torecsys.metrics import *


class NoveltyTestCase(unittest.TestCase):
    def test_base_metrics(self):
        occurrence = {
            i: random.randint(1, 1000) for i in range(100)
        }
        k = 5
        num_users = 10
        metric = Novelty(occurrence=occurrence, k=k, num_users=num_users)

        n_batches = 10
        for i in range(n_batches):
            pred = torch.randint(0, 100, size=(num_users, k))
            n = metric(pred)
            print(f'Novelty on batch {i} : {n}')

        n = metric.compute()
        print(f'Novelty on all data : {n}')
        self.assertNotEqual(n, None)


if __name__ == '__main__':
    unittest.main()
