import unittest
from math import comb

import torch
import torch.nn as nn
from parameterized import parameterized
from torch.nn import functional
from torchinfo import summary

from torecsys.layers import *

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class AttentionalFactorizationMachineLayerTestCase(unittest.TestCase):
    @parameterized.expand([
        (4, 2, 8, 1,),
        (16, 4, 16, 2,),
        (32, 4, 8, 4,),
    ])
    def test_forward(self, batch_size: int, num_fields: int, embed_size: int, attn_size: int):
        # Initialize layer
        layer = AttentionalFactorizationMachineLayer(
            embed_size=embed_size,
            num_fields=num_fields,
            attn_size=attn_size,
            dropout_p=0.1
        )
        layer = layer.to(device)
        print(f'Input Size: {layer.inputs_size};\nOutput Size: {layer.outputs_size}')

        # Generate inputs for the layer
        inp = torch.rand(batch_size, num_fields, embed_size)

        # Forward
        outputs, attn_scores = layer.forward(inp)
        self.assertEqual(outputs.size(), (batch_size, embed_size))
        self.assertEqual(attn_scores.size(), (batch_size, comb(num_fields, 2), 1))

        print(f'Output Size: {outputs.size()}, Output Dimensions: {outputs.names}')
        print(f'Attn Scores Size: {attn_scores.size()}, Attn Scores Dimensions: {attn_scores.names}')


class BiasEncodingLayerTestCase(unittest.TestCase):
    @parameterized.expand([
        (4, 32, 32, 256,),
        (16, 16, 32, 64,),
        (32, 8, 32, 128,),
    ])
    def test_forward(self, batch_size: int, max_length: int, embed_size: int, max_num_session: int):
        layer = BiasEncodingLayer(
            embed_size=embed_size,
            max_num_session=max_num_session,
            max_length=max_length
        )
        layer = layer.to(device)
        print(f'Input Size: {layer.inputs_size};\nOutput Size: {layer.outputs_size}')

        # Generate inputs for the layer
        inp_session_embed = torch.rand(batch_size, max_length, embed_size)
        inp_session_embed.names = ('B', 'L', 'E',)
        inp_position = torch.randint(0, max_length, (batch_size,), device=device, dtype=torch.long)
        inp_position.names = ('B',)

        # Forward
        outputs = layer.forward((inp_session_embed, inp_position))
        self.assertEqual(outputs.size(), (batch_size, max_length, embed_size))
        print(f'Output Size: {outputs.size()}, Output Dimensions: {outputs.names}')


class BilinearNetworkLayerTestCase(unittest.TestCase):
    @parameterized.expand([
        (8, 4, 128, 4),
        (16, 6, 64, 3),
        (32, 12, 8, 16)
    ])
    def test_forward(self, batch_size: int, num_fields: int, embed_size: int, num_layers: int):
        layer = BilinearNetworkLayer(
            inputs_size=embed_size,
            num_layers=num_layers
        )
        layer = layer.to(device)
        print(f'Input Size: {layer.inputs_size};\nOutput Size: {layer.outputs_size}')

        # Generate inputs for the layer
        inp = torch.rand(batch_size, num_fields, embed_size)
        inp.names = ('B', 'N', 'E',)
        inp_size = inp.size()

        summary(layer, input_size=[inp_size], device=device, dtypes=[torch.float])

        # Forward
        outputs = layer.forward(inp)
        self.assertEqual(outputs.size(), (batch_size, num_fields, embed_size))
        print(f'Output Size: {outputs.size()}, Output Dimensions: {outputs.names}')


class BilinearInteractionLayerTestCase(unittest.TestCase):
    @parameterized.expand([
        (8, 4, 128),
        (16, 6, 64),
        (32, 12, 8)
    ])
    def test_forward(self, batch_size: int, num_fields: int, embed_size: int):
        layer = BilinearInteractionLayer(
            embed_size=embed_size,
            num_fields=num_fields,
            bilinear_type='all',
            bias=True
        )
        layer = layer.to(device)
        print(f'Input Size: {layer.inputs_size};\nOutput Size: {layer.outputs_size}')

        # Generate inputs for the layer
        inp = torch.rand(batch_size, num_fields, embed_size)
        inp.names = ('B', 'N', 'E',)
        inp_size = inp.size()

        summary(layer, input_size=[inp_size], device=device, dtypes=[torch.float])

        # Forward
        outputs = layer.forward(inp)
        self.assertEqual(outputs.size(), (batch_size, comb(num_fields, 2), embed_size))
        print(f'Output Size: {outputs.size()}, Output Dimensions: {outputs.names}')


class ComposeExcitationNetworkLayerTestCase(unittest.TestCase):
    @parameterized.expand([
        (8, 4, 128),
        (16, 6, 64),
        (32, 12, 8)
    ])
    def test_forward(self, batch_size: int, num_fields: int, embed_size: int):
        layer = ComposeExcitationNetworkLayer(
            num_fields=num_fields,
            reduction=1,
            activation=nn.ReLU6()
        )
        layer = layer.to(device)
        print(f'Input Size: {layer.inputs_size};\nOutput Size: {layer.outputs_size}')

        # Generate inputs for the layer
        field_aware_embed_inp = torch.rand(batch_size, num_fields ** 2, embed_size)
        field_aware_embed_inp.names = ('B', 'N', 'E',)
        inp_size = field_aware_embed_inp.size()

        summary(layer, input_size=[inp_size], device=device, dtypes=[torch.float])

        # Forward
        outputs = layer.forward(field_aware_embed_inp)
        self.assertEqual(outputs.size(), (batch_size, num_fields ** 2, embed_size))
        print(f'Output Size: {outputs.size()}, Output Dimensions: {outputs.names}')


class CompressInteractionNetworkLayerTestCase(unittest.TestCase):
    @parameterized.expand([
        (8, 4, 128, 4),
        (16, 6, 64, 8),
        (32, 12, 8, 16)
    ])
    def test_forward(self, batch_size: int, num_fields: int, embed_size: int, output_size: int):
        layer = CompressInteractionNetworkLayer(
            embed_size=embed_size,
            num_fields=num_fields,
            output_size=output_size,
            layer_sizes=[32, 64, 32],
            is_direct=False,
            use_bias=True,
            use_batchnorm=True,
            activation=nn.ReLU6()
        )
        layer = layer.to(device)
        print(f'Input Size: {layer.inputs_size};\nOutput Size: {layer.outputs_size}')

        # Generate inputs for the layer
        inp = torch.rand(batch_size, num_fields, embed_size)
        inp.names = ('B', 'N', 'E',)
        inp_size = inp.size()

        summary(layer, input_size=[inp_size], device=device, dtypes=[torch.float])

        # Forward
        outputs = layer.forward(inp)
        self.assertEqual(outputs.size(), (batch_size, output_size))
        print(f'Output Size: {outputs.size()}, Output Dimensions: {outputs.names}')


class CrossNetworkLayerTestCase(unittest.TestCase):
    @parameterized.expand([
        (8, 4, 128),
        (16, 6, 64),
        (32, 12, 8)
    ])
    def test_forward(self, batch_size: int, num_fields: int, embed_size: int):
        layer = CrossNetworkLayer(
            inputs_size=embed_size,
            num_layers=4
        )
        layer = layer.to(device)
        print(f'Input Size: {layer.inputs_size};\nOutput Size: {layer.outputs_size}')

        # Generate inputs for the layer
        inp = torch.rand(batch_size, num_fields, embed_size)
        inp.names = ('B', 'N', 'E',)
        inp_size = inp.size()

        summary(layer, input_size=[inp_size], device=device, dtypes=[torch.float])

        # Forward
        outputs = layer.forward(inp)
        self.assertEqual(outputs.size(), (batch_size, num_fields, embed_size))
        print(f'Output Size: {outputs.size()}, Output Dimensions: {outputs.names}')


class DynamicRoutingLayerTestCase(unittest.TestCase):
    @parameterized.expand([
        (8, 4, 128, 4),
        (16, 6, 64, 8),
        (32, 12, 8, 16)
    ])
    def test_forward(self, batch_size: int, num_fields: int, embed_size: int, output_size: int):
        layer = DynamicRoutingLayer(
            embed_size=embed_size,
            routed_size=output_size,
            max_num_caps=16,
            num_iter=2
        )
        layer = layer.to(device)
        print(f'Input Size: {layer.inputs_size};\nOutput Size: {layer.outputs_size}')

        # Generate inputs for the layer
        inp = torch.rand(batch_size, num_fields, embed_size)
        inp.names = ('B', 'N', 'E',)
        inp_size = inp.size()

        summary(layer, input_size=[inp_size], device=device, dtypes=[torch.float])

        # Forward
        outputs = layer.forward(inp)
        self.assertEqual(outputs.size(), (batch_size, layer.num_caps, output_size))
        print(f'Output Size: {outputs.size()}, Output Dimensions: {outputs.names}')


class FactorizationMachineLayerTestCase(unittest.TestCase):
    @parameterized.expand([
        (8, 4, 128),
        (16, 6, 64),
        (32, 12, 8)
    ])
    def test_forward(self, batch_size: int, num_fields: int, embed_size: int):
        layer = FactorizationMachineLayer(dropout_p=0.9)
        layer = layer.to(device)
        print(f'Input Size: {layer.inputs_size};\nOutput Size: {layer.outputs_size}')

        # Generate inputs for the layer
        inp = torch.rand(batch_size, num_fields, embed_size)
        inp.names = ('B', 'N', 'E',)
        inp_size = inp.size()

        summary(layer, input_size=[inp_size], device=device, dtypes=[torch.float])

        # Forward
        outputs = layer.forward(inp)
        self.assertEqual(outputs.size(), (batch_size, embed_size))
        print(f'Output Size: {outputs.size()}, Output Dimensions: {outputs.names}')


class FieldAwareFactorizationMachineLayerTestCase(unittest.TestCase):
    @parameterized.expand([
        (8, 4, 128),
        (16, 6, 64),
        (32, 12, 8)
    ])
    def test_forward(self, batch_size: int, num_fields: int, embed_size: int):
        layer = FieldAwareFactorizationMachineLayer(
            num_fields=num_fields,
            dropout_p=0.9
        )
        layer = layer.to(device)
        print(f'Input Size: {layer.inputs_size};\nOutput Size: {layer.outputs_size}')

        # Generate inputs for the layer
        inp = torch.rand(batch_size, num_fields ** 2, embed_size)
        inp.names = ('B', 'N', 'E',)
        inp_size = inp.size()

        summary(layer, input_size=[inp_size], device=device, dtypes=[torch.float])

        # Forward
        outputs = layer.forward(inp)
        self.assertEqual(outputs.size(), (batch_size, comb(num_fields, 2), embed_size))
        print(f'Output Size: {outputs.size()}, Output Dimensions: {outputs.names}')


class InnerProductNetworkLayerTestCase(unittest.TestCase):
    @parameterized.expand([
        (8, 4, 128),
        (16, 6, 64),
        (32, 12, 8)
    ])
    def test_forward(self, batch_size: int, num_fields: int, embed_size: int):
        layer = InnerProductNetworkLayer(num_fields=num_fields)
        layer = layer.to(device)
        print(f'Input Size: {layer.inputs_size};\nOutput Size: {layer.outputs_size}')

        # Generate inputs for the layer
        inp = torch.rand(batch_size, num_fields, embed_size)
        inp.names = ('B', 'N', 'E',)
        inp_size = inp.size()

        summary(layer, input_size=[inp_size], device=device, dtypes=[torch.float])

        # Forward
        outputs = layer.forward(inp)
        self.assertEqual(outputs.size(), (batch_size, comb(num_fields, 2)))
        print(f'Output Size: {outputs.size()}, Output Dimensions: {outputs.names}')


class MixtureOfExpertsLayerTestCase(unittest.TestCase):
    @parameterized.expand([
        (8, 4, 128),
        (16, 6, 64),
        (32, 12, 8)
    ])
    def test_forward(self, batch_size: int, num_fields: int, embed_size: int):
        num_experts = 4
        expert_output_size = 16
        layer = MixtureOfExpertsLayer(
            inputs_size=embed_size * num_fields,
            output_size=expert_output_size * num_experts,
            num_experts=num_experts,
            expert_func=MultilayerPerceptionLayer,
            expert_inputs_size=embed_size * num_fields,
            expert_output_size=expert_output_size,
            expert_layer_sizes=[128, 64, 64]
        )
        layer = layer.to(device)
        print(f'Input Size: {layer.inputs_size};\nOutput Size: {layer.outputs_size}')

        # Generate inputs for the layer
        inp = torch.rand(batch_size, num_fields, embed_size)
        inp.names = ('B', 'N', 'E',)
        inp_size = inp.size()

        summary(layer, input_size=[inp_size], device=device, dtypes=[torch.float])

        # Forward
        outputs = layer.forward(inp)
        self.assertEqual(outputs.size(), (batch_size, 1, num_experts * expert_output_size))
        print(f'Output Size: {outputs.size()}, Output Dimensions: {outputs.names}')


class MultilayerPerceptionLayerTestCase(unittest.TestCase):
    @parameterized.expand([
        (8, 4, 128),
        (16, 6, 64),
        (32, 12, 8)
    ])
    def test_forward(self, batch_size: int, num_fields: int, embed_size: int):
        output_size = 16
        layer_sizes = [32, 32, 16]
        dropout_p = [0.9, 0.9, 0.9]
        layer = MultilayerPerceptionLayer(
            inputs_size=embed_size,
            output_size=output_size,
            layer_sizes=layer_sizes,
            dropout_p=dropout_p
        )
        layer = layer.to(device)
        print(f'Input Size: {layer.inputs_size};\nOutput Size: {layer.outputs_size}')

        # Generate inputs for the layer
        inp = torch.rand(batch_size, num_fields, embed_size)
        inp.names = ('B', 'N', 'E',)
        inp_size = inp.size()

        summary(layer, input_size=[inp_size], device=device, dtypes=[torch.float])

        # Forward
        outputs = layer.forward(inp)
        self.assertEqual(outputs.size(), (batch_size, num_fields, output_size))
        print(f'Output Size: {outputs.size()}, Output Dimensions: {outputs.names}')


class OuterProductNetworkLayerTestCase(unittest.TestCase):
    @parameterized.expand([
        (8, 4, 128),
        (16, 6, 64),
        (32, 12, 8)
    ])
    def test_forward(self, batch_size: int, num_fields: int, embed_size: int):
        layer = OuterProductNetworkLayer(
            embed_size=embed_size,
            num_fields=num_fields,
            kernel_type='mat'
        )
        layer = layer.to(device)
        print(f'Input Size: {layer.inputs_size};\nOutput Size: {layer.outputs_size}')

        # Generate inputs for the layer
        inp = torch.rand(batch_size, num_fields, embed_size)
        inp.names = ('B', 'N', 'E',)
        inp_size = inp.size()

        summary(layer, input_size=[inp_size], device=device, dtypes=[torch.float])

        # Forward
        outputs = layer.forward(inp)
        self.assertEqual(outputs.size(), (batch_size, comb(num_fields, 2)))
        print(f'Output Size: {outputs.size()}, Output Dimensions: {outputs.names}')


class PositionEmbeddingLayerTestCase(unittest.TestCase):
    @parameterized.expand([
        (8, 4, 128),
        (16, 6, 64),
        (32, 12, 8)
    ])
    def test_forward(self, batch_size: int, seq_length: int, embed_size: int):
        layer = PositionEmbeddingLayer(max_num_position=seq_length)
        layer = layer.to(device)
        print(f'Input Size: {layer.inputs_size};\nOutput Size: {layer.outputs_size}')

        # Generate inputs for the layer
        inp = torch.rand(batch_size, seq_length, embed_size)
        inp.names = ('B', 'L', 'E',)
        inp_size = inp.size()

        summary(layer, input_size=[inp_size], device=device, dtypes=[torch.float])

        # Forward
        outputs = layer.forward(inp)
        self.assertEqual(outputs.size(), (batch_size, seq_length, embed_size))
        print(f'Output Size: {outputs.size()}, Output Dimensions: {outputs.names}')


class PositionBiasAwareLearningFrameworkLayerTestCase(unittest.TestCase):
    @parameterized.expand([
        (8, 4, 128),
        (16, 6, 64),
        (32, 12, 8)
    ])
    def test_forward(self, batch_size: int, seq_length: int, embed_size: int):
        layer = PositionBiasAwareLearningFrameworkLayer(
            input_size=embed_size,
            max_num_position=seq_length
        )
        layer = layer.to(device)
        print(f'Input Size: {layer.inputs_size};\nOutput Size: {layer.outputs_size}')

        # Generate inputs for the layer
        inp_0 = torch.rand(batch_size, embed_size)
        inp_0.names = ('B', 'E',)
        # inp_0_size = inp_0.size()

        inp_1 = torch.randint(0, seq_length, (batch_size,), device=device, dtype=torch.long)
        inp_1.names = ('B',)
        # inp_1_size = inp_1.size()

        # summary(layer, input_size=[inp_0_size, inp_1_size], device=device, dtypes=[torch.float, torch.long])

        # Forward
        outputs = layer.forward((inp_0, inp_1,))
        self.assertEqual(outputs.size(), (batch_size, embed_size))
        print(f'Output Size: {outputs.size()}, Output Dimensions: {outputs.names}')


class WideLayerTestCase(unittest.TestCase):
    @parameterized.expand([
        (8, 4, 128),
        (16, 6, 64),
        (32, 12, 8)
    ])
    def test_forward(self, batch_size: int, num_fields: int, embed_size: int):
        output_size = 16
        dropout_p = 0.9
        layer = WideLayer(
            inputs_size=embed_size,
            output_size=output_size,
            dropout_p=dropout_p
        )
        layer = layer.to(device)
        print(f'Input Size: {layer.inputs_size};\nOutput Size: {layer.outputs_size}')

        # Generate inputs for the layer
        inp = torch.rand(batch_size, num_fields, embed_size)
        inp.names = ('B', 'N', 'E',)
        inp_size = inp.size()

        summary(layer, input_size=[inp_size], device=device, dtypes=[torch.float])

        # Forward
        outputs = layer.forward(inp)
        self.assertEqual(outputs.size(), (batch_size, num_fields, output_size))
        print(f'Output Size: {outputs.size()}, Output Dimensions: {outputs.names}')


class GeneralizedMatrixFactorizationLayerTestCase(unittest.TestCase):
    @parameterized.expand([
        (8, 128),
        (16, 64),
        (32, 8)
    ])
    def test_forward(self, batch_size: int, embed_size: int):
        layer = GeneralizedMatrixFactorizationLayer()
        layer = layer.to(device)
        print(f'Input Size: {layer.inputs_size};\nOutput Size: {layer.outputs_size}')

        # Generate inputs for the layer
        inp = torch.rand(batch_size, 2, embed_size)
        inp.names = ('B', 'N', 'E',)
        inp_size = inp.size()

        summary(layer, input_size=[inp_size], device=device, dtypes=[torch.float])

        # Forward
        outputs = layer.forward(inp)
        self.assertEqual(outputs.size(), (batch_size, 1))
        print(f'Output Size: {outputs.size()}, Output Dimensions: {outputs.names}')


class StarSpaceLayerTestCase(unittest.TestCase):
    @parameterized.expand([
        (8, 128),
        (16, 64),
        (32, 8)
    ])
    def test_forward(self, batch_size: int, embed_size: int):
        layer = StarSpaceLayer(
            similarity=functional.cosine_similarity
        )
        layer = layer.to(device)
        print(f'Input Size: {layer.inputs_size};\nOutput Size: {layer.outputs_size}')

        # Generate inputs for the layer
        inp = torch.rand(batch_size, 2, embed_size)
        inp.names = ('B', 'N', 'E',)
        inp_size = inp.size()

        summary(layer, input_size=[inp_size], device=device, dtypes=[torch.float])

        # Forward
        outputs = layer.forward(inp)
        self.assertEqual(outputs.size(), (batch_size, embed_size))
        print(f'Output Size: {outputs.size()}, Output Dimensions: {outputs.names}')


if __name__ == '__main__':
    unittest.main()
