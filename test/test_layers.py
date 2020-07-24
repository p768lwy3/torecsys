import unittest
from math import comb

from parameterized import parameterized

import torecsys.layers as layers


class TestLayersMethods(unittest.TestCase):
    @parameterized([
        (4, 2, 8),
        (16, 4, 16),
        (32, 4, 8)
    ])
    def test_attention_factorization_machine(self, batch_size, num_fields, embed_size):
        # initialize layer
        layer = layers.AttentionalFactorizationMachineLayer()

        # testing forward
        emb_inputs = torch.rand(batch_size, num_fields, embed_size)
        outputs, attn_scores = layer.forward(emb_inputs)

        self.assertEqual(outputs.size(), (batch_size, embed_size))
        self.assertEqual(attn_scores.size(), (batch_size, comb(num_fields, 2), 1))

    def test_something(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
