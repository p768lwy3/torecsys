import unittest

import torch
from parameterized import parameterized

from torecsys.losses import *

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class AdaptiveHingeLossTestCase(unittest.TestCase):
    @parameterized.expand([
        (4, 32,),
        (16, 16,),
        (32, 4,),
    ])
    def test_forward(self, batch_size: int, num_neg: int):
        criterion = AdaptiveHingeLoss()
        criterion = criterion.to(device)

        pos_out = torch.rand(batch_size, 1)
        neg_out = torch.rand(batch_size, num_neg)
        mask = torch.randint(0, 1, (batch_size,))
        mask = mask == 1
        loss = criterion(pos_out, neg_out, mask)

        self.assertEqual(loss.size(), torch.Size([]))
        print(f'Loss Size: {loss.size()}; Loss: {loss.item()}')


class BayesianPersonalizedRankingLossTestCase(unittest.TestCase):
    @parameterized.expand([
        (4, 32,),
        (16, 16,),
        (32, 4,),
    ])
    def test_forward(self, batch_size: int, num_neg: int):
        criterion = BayesianPersonalizedRankingLoss(reduction='sum')
        criterion = criterion.to(device)

        pos_out = torch.rand(batch_size, 1)
        neg_out = torch.rand(batch_size, num_neg)
        mask = torch.randint(0, 1, (batch_size,))
        mask = mask == 1
        loss = criterion(pos_out, neg_out, mask)

        self.assertEqual(loss.size(), torch.Size([]))
        print(f'Loss Size: {loss.size()}; Loss: {loss.item()}')


class HingeLossTestCase(unittest.TestCase):
    @parameterized.expand([
        (4, 32,),
        (16, 16,),
        (32, 4,),
    ])
    def test_forward(self, batch_size: int, num_neg: int):
        criterion = HingeLoss()
        criterion = criterion.to(device)

        pos_out = torch.rand(batch_size, 1)
        neg_out = torch.rand(batch_size, num_neg)
        mask = torch.randint(0, 1, (batch_size,))
        mask = mask == 1
        loss = criterion(pos_out, neg_out, mask)

        self.assertEqual(loss.size(), torch.Size([]))
        print(f'Loss Size: {loss.size()}; Loss: {loss.item()}')


class ListnetLossTestCase(unittest.TestCase):
    @parameterized.expand([
        (4, 32,),
        (16, 16,),
        (32, 4,),
    ])
    def test_forward(self, batch_size: int, length: int):
        criterion = ListnetLoss()
        criterion = criterion.to(device)

        y_hat = torch.rand(batch_size, length)
        y_true = torch.rand(batch_size, length)
        mask = torch.randint(0, 1, (batch_size,))
        mask = mask == 1
        loss = criterion(y_hat, y_true, mask)

        self.assertEqual(loss.size(), torch.Size([]))
        print(f'Loss Size: {loss.size()}; Loss: {loss.item()}')


class PointwiseLogisticLossTestCase(unittest.TestCase):
    @parameterized.expand([
        (4, 32,),
        (16, 16,),
        (32, 4,),
    ])
    def test_forward(self, batch_size: int, num_neg: int):
        criterion = PointwiseLogisticLoss()
        criterion = criterion.to(device)

        pos_out = torch.rand(batch_size, 1)
        neg_out = torch.rand(batch_size, num_neg)
        mask = torch.randint(0, 1, (batch_size,))
        mask = mask == 1
        loss = criterion(pos_out, neg_out, mask)

        self.assertEqual(loss.size(), torch.Size([]))
        print(f'Loss Size: {loss.size()}; Loss: {loss.item()}')


class SkipGramLossTestCase(unittest.TestCase):
    @parameterized.expand([
        (4, 32, 32,),
        (16, 64, 16,),
        (32, 128, 4,),
    ])
    def test_forward(self, batch_size: int, embed_size: int, num_neg: int):
        criterion = SkipGramLoss()
        criterion = criterion.to(device)

        content_inp = torch.rand(batch_size, 1, embed_size)
        pos_inp = torch.rand(batch_size, 1, embed_size)
        neg_inp = torch.rand(batch_size, num_neg, embed_size)
        loss = criterion(content_inp, pos_inp, neg_inp)

        self.assertEqual(loss.size(), torch.Size([]))
        print(f'Loss Size: {loss.size()}; Loss: {loss.item()}')


class TripletLossTestCase(unittest.TestCase):
    @parameterized.expand([
        (4, 32, 32,),
        (16, 64, 16,),
        (32, 128, 4,),
    ])
    def test_forward(self, batch_size: int, embed_size: int, num_neg: int):
        criterion = TripletLoss(margin=1.0, reduction='sum')
        criterion = criterion.to(device)

        pos_out = torch.rand(batch_size, 1)
        neg_out = torch.rand(batch_size, num_neg)
        mask = torch.randint(0, 1, (batch_size,))
        mask = mask == 1
        loss = criterion(pos_out, neg_out, mask)

        self.assertEqual(loss.size(), torch.Size([]))
        print(f'Loss Size: {loss.size()}; Loss: {loss.item()}')


if __name__ == '__main__':
    unittest.main()
