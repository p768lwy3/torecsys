import unittest
from functools import partial

import torch
import torch.nn as nn
from parameterized import parameterized
from torchinfo import summary

from torecsys.miners import UniformBatchMiner
from torecsys.models import *
from torecsys.utils.operations import inner_product_similarity

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class AttentionalFactorizationMachineModelTestCase(unittest.TestCase):
    @parameterized.expand([
        (8, 4, 128),
        (16, 6, 64),
        (32, 12, 8)
    ])
    def test_forward(self, batch_size: int, num_fields: int, embed_size: int):
        model = AttentionalFactorizationMachineModel(
            embed_size=embed_size,
            num_fields=num_fields,
            attn_size=16,
            use_bias=True,
            dropout_p=0.9
        )
        model = model.to(device)

        # Generate inputs for the layer
        feat_inp = torch.rand(batch_size, num_fields, 1)
        feat_inp.names = ('B', 'N', 'E',)
        feat_inp_size = feat_inp.size()

        emb_inp = torch.rand(batch_size, num_fields, embed_size)
        emb_inp.names = ('B', 'N', 'E',)
        emb_inp_size = emb_inp.size()

        summary(model, input_size=[feat_inp_size, emb_inp_size], device=device, dtypes=[torch.float, torch.float])

        # Forward
        outputs = model.forward(feat_inp, emb_inp)
        self.assertEqual(outputs.size(), (batch_size, 1))
        print(f'Output Size: {outputs.size()}, Output Dimensions: {outputs.names}')


class DeepAndCrossNetworkModelTestCase(unittest.TestCase):
    @parameterized.expand([
        (8, 4, 128),
        (16, 6, 64),
        (32, 12, 8)
    ])
    def test_forward(self, batch_size: int, num_fields: int, embed_size: int):
        output_size = 1
        model = DeepAndCrossNetworkModel(
            inputs_size=embed_size,
            num_fields=num_fields,
            deep_output_size=4,
            deep_layer_sizes=[32, 16, 8],
            cross_num_layers=4,
            output_size=output_size,
            deep_dropout_p=[0.9, 0.9, 0.9],
            deep_activation=nn.ReLU6()
        )
        model = model.to(device)

        # Generate inputs for the layer
        emb_inp = torch.rand(batch_size, num_fields, embed_size)
        emb_inp.names = ('B', 'N', 'E',)
        emb_inp_size = emb_inp.size()

        summary(model, input_size=[emb_inp_size], device=device, dtypes=[torch.float])

        # Forward
        outputs = model.forward(emb_inp)
        self.assertEqual(outputs.size(), (batch_size, output_size))
        print(f'Output Size: {outputs.size()}, Output Dimensions: {outputs.names}')


class DeepFieldAwareFactorizationMachineModelTestCase(unittest.TestCase):
    @parameterized.expand([
        (8, 4, 128),
        (16, 6, 64),
        (32, 12, 8)
    ])
    def test_forward(self, batch_size: int, num_fields: int, embed_size: int):
        model = DeepFieldAwareFactorizationMachineModel(
            embed_size=embed_size,
            num_fields=num_fields,
            deep_output_size=4,
            deep_layer_sizes=[32, 16, 8],
            ffm_dropout_p=0.9,
            deep_dropout_p=[0.9, 0.9, 0.9],
            deep_activation=nn.ReLU6()
        )
        model = model.to(device)

        # Generate inputs for the layer
        field_emb_inp = torch.rand(batch_size, num_fields ** 2, embed_size)
        field_emb_inp.names = ('B', 'N', 'E',)
        field_emb_inp_size = field_emb_inp.size()

        summary(model, input_size=[field_emb_inp_size], device=device, dtypes=[torch.float])

        # Forward
        outputs = model.forward(field_emb_inp)
        self.assertEqual(outputs.size(), (batch_size, 1))
        print(f'Output Size: {outputs.size()}, Output Dimensions: {outputs.names}')


class DeepFactorizationMachineModelTestCase(unittest.TestCase):
    @parameterized.expand([
        (8, 4, 128),
        (16, 6, 64),
        (32, 12, 8)
    ])
    def test_forward(self, batch_size: int, num_fields: int, embed_size: int):
        model = DeepFactorizationMachineModel(
            embed_size=embed_size,
            num_fields=num_fields,
            deep_layer_sizes=[16, 16, 16],
            fm_dropout_p=0.9,
            deep_dropout_p=[0.9, 0.9, 0.9],
            deep_activation=nn.ReLU()
        )
        model = model.to(device)

        # Generate inputs for the layer
        feat_inp = torch.rand(batch_size, num_fields, 1)
        feat_inp.names = ('B', 'N', 'E',)
        feat_inp_size = feat_inp.size()

        emb_inp = torch.rand(batch_size, num_fields, embed_size)
        emb_inp.names = ('B', 'N', 'E',)
        emb_inp_size = emb_inp.size()

        summary(model, input_size=[feat_inp_size, emb_inp_size], device=device, dtypes=[torch.float, torch.float])

        # Forward
        outputs = model.forward(feat_inp, emb_inp)
        self.assertEqual(outputs.size(), (batch_size, 1))
        print(f'Output Size: {outputs.size()}, Output Dimensions: {outputs.names}')


class DeepMatchingCorrelationPredictionModelTestCase(unittest.TestCase):
    @parameterized.expand([
        (8, 4, 16, 128),
        (16, 6, 32, 64),
        (32, 12, 64, 8)
    ])
    def test_forward(self, batch_size: int, user_num_fields: int, item_num_fields: int, embed_size: int):
        model = DeepMatchingCorrelationPredictionModel(
            embed_size=embed_size,
            user_num_fields=user_num_fields,
            item_num_fields=item_num_fields,
            corr_output_size=8,
            match_output_size=8,
            corr_layer_sizes=[16, 16, 16],
            match_layer_sizes=[16, 16, 16],
            pred_layer_sizes=[16, 16, 16],
            corr_dropout_p=[0.9, 0.9, 0.9],
            match_dropout_p=[0.9, 0.9, 0.9],
            pred_dropout_p=[0.9, 0.9, 0.9],
            corr_activation=nn.ReLU(),
            match_activation=nn.ReLU(),
            pred_activation=nn.ReLU()
        )
        model = model.to(device)

        # Generate inputs for the layer
        user_emb_inp = torch.rand(batch_size, user_num_fields, embed_size)
        user_emb_inp.names = ('B', 'N', 'E',)
        user_emb_inp_size = user_emb_inp.size()

        content_emb_inp = torch.rand(batch_size, item_num_fields, embed_size)
        content_emb_inp.names = ('B', 'N', 'E',)
        content_emb_inp_size = content_emb_inp.size()

        pos_emb_inp = torch.rand(batch_size, item_num_fields, embed_size)
        pos_emb_inp.names = ('B', 'N', 'E',)
        pos_emb_inp_size = pos_emb_inp.size()

        # num_neg_samples = 16
        neg_emb_inp = torch.rand(batch_size, item_num_fields, embed_size)
        neg_emb_inp.names = ('B', 'N', 'E',)
        neg_emb_inp_size = neg_emb_inp.size()

        summary(model, input_size=[user_emb_inp_size, content_emb_inp_size, pos_emb_inp_size, neg_emb_inp_size],
                device=device, dtypes=[torch.float, torch.float, torch.float, torch.float])

        # Forward
        y_pred, y_match, y_corr_pos, y_corr_neg = model.forward(user_emb_inp, content_emb_inp, pos_emb_inp, neg_emb_inp)
        self.assertEqual(y_pred.size(), (batch_size, 1))
        self.assertEqual(y_match.size(), (batch_size, 1))
        self.assertEqual(y_corr_pos.size(), (batch_size, 1))
        self.assertEqual(y_corr_neg.size(), (batch_size, 1))
        print(f'y_pred Size: {y_pred.size()},\n'
              f'y_match Size: {y_match.size()},\n'
              f'y_corr_pos Size: {y_corr_pos.size()},\n'
              f'y_corr_neg Size: {y_corr_neg.size()},\n')


class DeepMixtureOfExpertsModelTestCase(unittest.TestCase):
    @parameterized.expand([
        (8, 4, 128),
        (16, 6, 64),
        (32, 12, 8)
    ])
    def test_forward(self, batch_size: int, num_fields: int, embed_size: int):
        model = DeepMixtureOfExpertsModel(
            embed_size=embed_size,
            num_fields=num_fields,
            num_experts=4,
            moe_layer_sizes=[16, 16, 16],
            deep_layer_sizes=[16, 16, 16],
            deep_dropout_p=[0.9, 0.9, 0.9],
            deep_activation=nn.ReLU()
        )
        model = model.to(device)

        # Generate inputs for the layer
        emb_inp = torch.rand(batch_size, num_fields, embed_size)
        emb_inp.names = ('B', 'N', 'E',)
        emb_inp_size = emb_inp.size()

        summary(model, input_size=[emb_inp_size], device=device, dtypes=[torch.float])

        # Forward
        outputs = model.forward(emb_inp)
        self.assertEqual(outputs.size(), (batch_size, 1))
        print(f'Output Size: {outputs.size()}, Output Dimensions: {outputs.names}')


class ElaboratedEntireSpaceSupervisedMultiTaskModelTestCase(unittest.TestCase):
    @parameterized.expand([
        (8, 4, 128),
        (16, 6, 64),
        (32, 12, 8)
    ])
    def test_forward(self, batch_size: int, num_fields: int, embed_size: int):
        model = ElaboratedEntireSpaceSupervisedMultiTaskModel(
            num_fields=num_fields,
            layer_sizes=[16, 16, 16],
            dropout_p=[0.9, 0.9, 0.9],
            activation=nn.ReLU()
        )
        model = model.to(device)

        # Generate inputs for the layer
        emb_inp = torch.rand(batch_size, num_fields, embed_size)
        emb_inp.names = ('B', 'N', 'E',)
        emb_inp_size = emb_inp.size()

        summary(model, input_size=[emb_inp_size], device=device, dtypes=[torch.float])

        # Forward
        prob_impress_to_click, prob_impress_to_d_action, prob_impress_to_buy = model.forward(emb_inp)
        self.assertEqual(prob_impress_to_click.size(), (batch_size, 1))
        self.assertEqual(prob_impress_to_d_action.size(), (batch_size, 1))
        self.assertEqual(prob_impress_to_buy.size(), (batch_size, 1))
        print(f'Prob impress to click Size: {prob_impress_to_click.size()},\n'
              f'Prob impress to d action Size: {prob_impress_to_d_action.size()},\n'
              f'Prob impress to buy Size: {prob_impress_to_buy.size()},\n')


class EntireSpaceMultiTaskModelTestCase(unittest.TestCase):
    @parameterized.expand([
        (8, 4, 128),
        (16, 6, 64),
        (32, 12, 8)
    ])
    def test_forward(self, batch_size: int, num_fields: int, embed_size: int):
        model = EntireSpaceMultiTaskModel(
            num_fields=num_fields,
            layer_sizes=[16, 16, 16],
            dropout_p=[0.9, 0.9, 0.9],
            activation=nn.ReLU()
        )
        model = model.to(device)

        # Generate inputs for the layer
        emb_inp = torch.rand(batch_size, num_fields, embed_size)
        emb_inp.names = ('B', 'N', 'E',)
        emb_inp_size = emb_inp.size()

        summary(model, input_size=[emb_inp_size], device=device, dtypes=[torch.float])

        # Forward
        pcvr, pctr = model.forward(emb_inp)
        self.assertEqual(pcvr.size(), (batch_size, 1))
        self.assertEqual(pctr.size(), (batch_size, 1))
        print(f'pcvr Size: {pcvr.size()}\n'
              f'pctr Size: {pctr.names}')


class FactorizationMachineModelTestCase(unittest.TestCase):
    @parameterized.expand([
        (8, 4, 128),
        (16, 6, 64),
        (32, 12, 8)
    ])
    def test_forward(self, batch_size: int, num_fields: int, embed_size: int):
        model = FactorizationMachineModel(dropout_p=0.9)
        model = model.to(device)

        # Generate inputs for the layer
        feat_inp = torch.rand(batch_size, num_fields, 1)
        feat_inp.names = ('B', 'N', 'E',)
        feat_inp_size = feat_inp.size()

        emb_inp = torch.rand(batch_size, num_fields, embed_size)
        emb_inp.names = ('B', 'N', 'E',)
        emb_inp_size = emb_inp.size()

        summary(model, input_size=[feat_inp_size, emb_inp_size], device=device, dtypes=[torch.float, torch.float])

        # Forward
        outputs = model.forward(feat_inp, emb_inp)
        self.assertEqual(outputs.size(), (batch_size, 1))
        print(f'Output Size: {outputs.size()}, Output Dimensions: {outputs.names}')


class FactorizationMachineSupportedNeuralNetworkModelTestCase(unittest.TestCase):
    @parameterized.expand([
        (8, 4, 128),
        (16, 6, 64),
        (32, 12, 8)
    ])
    def test_forward(self, batch_size: int, num_fields: int, embed_size: int):
        output_size = 4
        model = FactorizationMachineSupportedNeuralNetworkModel(
            embed_size=embed_size,
            num_fields=num_fields,
            deep_output_size=output_size,
            deep_layer_sizes=[32, 32, 16],
            fm_dropout_p=0.9,
            deep_dropout_p=[0.9, 0.9, 0.9],
            deep_activation=nn.ReLU()
        )
        model = model.to(device)

        # Generate inputs for the layer
        feat_inp = torch.rand(batch_size, num_fields, 1)
        feat_inp.names = ('B', 'N', 'E',)
        feat_inp_size = feat_inp.size()

        emb_inp = torch.rand(batch_size, num_fields, embed_size)
        emb_inp.names = ('B', 'N', 'E',)
        emb_inp_size = emb_inp.size()

        summary(model, input_size=[feat_inp_size, emb_inp_size], device=device, dtypes=[torch.float, torch.float])

        # Forward
        outputs = model.forward(feat_inp, emb_inp)
        self.assertEqual(outputs.size(), (batch_size, output_size))
        print(f'Output Size: {outputs.size()}, Output Dimensions: {outputs.names}')


class FieldAttentiveDeepFieldAwareFactorizationMachineModelTestCase(unittest.TestCase):
    @parameterized.expand([
        (8, 4, 128),
        (16, 6, 64),
        (32, 12, 8)
    ])
    def test_forward(self, batch_size: int, num_fields: int, embed_size: int):
        output_size = 16
        model = FieldAttentiveDeepFieldAwareFactorizationMachineModel(
            embed_size=embed_size,
            num_fields=num_fields,
            deep_output_size=output_size,
            deep_layer_sizes=[32, 32, 16],
            reduction=2,
            ffm_dropout_p=0.9,
            deep_dropout_p=[0.9, 0.9, 0.9],
            deep_activation=nn.ReLU()
        )
        model = model.to(device)

        # Generate inputs for the layer
        field_emb_inp = torch.rand(batch_size, num_fields ** 2, embed_size)
        field_emb_inp.names = ('B', 'N', 'E',)
        field_emb_inp_size = field_emb_inp.size()

        summary(model, input_size=[field_emb_inp_size], device=device, dtypes=[torch.float])

        # Forward
        outputs = model.forward(field_emb_inp)
        self.assertEqual(outputs.size(), (batch_size, output_size))
        print(f'Output Size: {outputs.size()}, Output Dimensions: {outputs.names}')


class FieldAwareFactorizationMachineModelTestCase(unittest.TestCase):
    @parameterized.expand([
        (8, 4, 128),
        (16, 6, 64),
        (32, 12, 8)
    ])
    def test_forward(self, batch_size: int, num_fields: int, embed_size: int):
        model = FieldAwareFactorizationMachineModel(
            num_fields=num_fields,
            dropout_p=0.9
        )
        model = model.to(device)

        # Generate inputs for the layer
        feat_inp = torch.rand(batch_size, num_fields, 1)
        feat_inp.names = ('B', 'N', 'E',)
        feat_inp_size = feat_inp.size()

        field_emb_inp = torch.rand(batch_size, num_fields ** 2, embed_size)
        field_emb_inp.names = ('B', 'N', 'E',)
        field_emb_inp_size = field_emb_inp.size()

        summary(model, input_size=[feat_inp_size, field_emb_inp_size], device=device, dtypes=[torch.float, torch.float])

        # Forward
        outputs = model.forward(feat_inp, field_emb_inp)
        self.assertEqual(outputs.size(), (batch_size, 1))
        print(f'Output Size: {outputs.size()}, Output Dimensions: {outputs.names}')


class FeatureImportanceAndBilinearFeatureInteractionNetworkTestCase(unittest.TestCase):
    @parameterized.expand([
        (8, 4, 128),
        (16, 6, 64),
        (32, 12, 8)
    ])
    def test_forward(self, batch_size: int, num_fields: int, embed_size: int):
        output_size = 1
        model = FeatureImportanceAndBilinearFeatureInteractionNetwork(
            embed_size=embed_size,
            num_fields=num_fields,
            senet_reduction=4,
            deep_output_size=output_size,
            deep_layer_sizes=[16, 16, 16],
            bilinear_type='all',
            bilinear_bias=True,
            deep_dropout_p=[0.9, 0.9, 0.9],
            deep_activation=nn.ReLU6()
        )
        model = model.to(device)

        # Generate inputs for the layer
        emb_inp = torch.rand(batch_size, num_fields, embed_size)
        emb_inp.names = ('B', 'N', 'E',)
        emb_inp_size = emb_inp.size()

        summary(model, input_size=[emb_inp_size], device=device, dtypes=[torch.float])

        # Forward
        outputs = model.forward(emb_inp)
        self.assertEqual(outputs.size(), (batch_size, output_size))
        print(f'Output Size: {outputs.size()}, Output Dimensions: {outputs.names}')


class LearningToRankWrapperTestCase(unittest.TestCase):
    @parameterized.expand([
        (8, 128),
        (16, 64),
        (32, 8)
    ])
    def test_forward(self, batch_size: int, embed_size: int):
        sample_size = 10
        miner = UniformBatchMiner(sample_size=sample_size)

        model = MatrixFactorizationModel()
        wrapped = LearningToRankWrapper(model=model)
        wrapped = wrapped.to(device)

        # Generate inputs for the layer
        emb_inp = torch.rand(batch_size, 2, embed_size)

        # mine negative samples with miner
        # outputs: p, shape = (B, N, E)
        # outputs: n, shape = (B * N Neg, N, E)
        p, n = miner(emb_inp[:, 0], emb_inp[:, 1])
        p.names = ('B', 'N', 'E',)
        n.names = ('B', 'N', 'E',)
        # p_size = p.size()
        # n_size = n.size()

        # summary(wrapped, input_size=[p_size, n_size], device=device, dtypes=[torch.float, torch.float])

        # Forward
        outputs = wrapped.forward(pos_inputs={'emb_inputs': p}, neg_inputs={'emb_inputs': n})
        self.assertEqual(outputs['pos_outputs'].size(), (batch_size, 1))
        self.assertEqual(outputs['neg_outputs'].size(), (batch_size * sample_size, 1))
        print(f'Pos Output Size: {outputs["pos_outputs"].size()},\n'
              f'Pos Output Dimensions: {outputs["pos_outputs"].names},\n'
              f'Neg Output Size: {outputs["neg_outputs"].size()},\n'
              f'NegOutput Dimensions: {outputs["neg_outputs"].names}')


class LogisticRegressionModelTestCase(unittest.TestCase):
    @parameterized.expand([
        (8, 4, 128),
        (16, 6, 64),
        (32, 12, 8)
    ])
    def test_forward(self, batch_size: int, num_fields: int, embed_size: int):
        output_size = 1
        model = LogisticRegressionModel(
            inputs_size=num_fields * embed_size,
            output_size=output_size
        )
        model = model.to(device)

        # Generate inputs for the layer
        emb_inp = torch.rand(batch_size, num_fields, embed_size)
        emb_inp.names = ('B', 'N', 'E',)
        emb_inp_size = emb_inp.size()

        summary(model, input_size=[emb_inp_size], device=device, dtypes=[torch.float])

        # Forward
        outputs = model.forward(emb_inp)
        self.assertEqual(outputs.size(), (batch_size, output_size))
        print(f'Output Size: {outputs.size()}, Output Dimensions: {outputs.names}')


class MatrixFactorizationModelTestCase(unittest.TestCase):
    @parameterized.expand([
        (8, 128),
        (16, 64),
        (32, 8)
    ])
    def test_forward(self, batch_size: int, embed_size: int):
        model = MatrixFactorizationModel()
        model = model.to(device)

        # Generate inputs for the layer
        emb_inp = torch.rand(batch_size, 2, embed_size)
        emb_inp.names = ('B', 'N', 'E',)
        emb_inp_size = emb_inp.size()

        summary(model, input_size=[emb_inp_size], device=device, dtypes=[torch.float])

        # Forward
        outputs = model.forward(emb_inp)
        self.assertEqual(outputs.size(), (batch_size, 1))
        print(f'Output Size: {outputs.size()}, Output Dimensions: {outputs.names}')


class MultiGateMixtureOfExpertsModelTestCase(unittest.TestCase):
    @parameterized.expand([
        (8, 4, 128),
        (16, 6, 64),
        (32, 12, 8)
    ])
    def test_forward(self, batch_size: int, num_fields: int, embed_size: int):
        model = MultiGateMixtureOfExpertsModel(
            embed_size=embed_size,
            num_fields=num_fields,
            num_tasks=4,
            num_experts=4,
            expert_output_size=1,
            expert_layer_sizes=[16, 16, 16],
            deep_layer_sizes=[16, 16, 16],
            expert_dropout_p=[0.9, 0.9, 0.9],
            deep_dropout_p=[0.9, 0.9, 0.9],
            expert_activation=nn.ReLU6(),
            deep_activation=nn.ReLU6()
        )
        model = model.to(device)

        # Generate inputs for the layer
        emb_inp = torch.rand(batch_size, num_fields, embed_size)
        emb_inp.names = ('B', 'N', 'E',)
        emb_inp_size = emb_inp.size()

        summary(model, input_size=[emb_inp_size], device=device, dtypes=[torch.float])

        # Forward
        outputs = model.forward(emb_inp)
        self.assertEqual(outputs.size(), (batch_size, 1))
        print(f'Output Size: {outputs.size()}, Output Dimensions: {outputs.names}')


class NeuralCollaborativeFilteringModelTestCase(unittest.TestCase):
    @parameterized.expand([
        (8, 128),
        (16, 64),
        (32, 8)
    ])
    def test_forward(self, batch_size: int, embed_size: int):
        model = NeuralCollaborativeFilteringModel(
            embed_size=embed_size,
            deep_output_size=8,
            deep_layer_sizes=[16, 16, 16],
            deep_dropout_p=[0.9, 0.9, 0.9],
            deep_activation=nn.ReLU6()
        )
        model = model.to(device)

        # Generate inputs for the layer
        emb_inp = torch.rand(batch_size, 2, embed_size)
        emb_inp.names = ('B', 'N', 'E',)
        emb_inp_size = emb_inp.size()

        summary(model, input_size=[emb_inp_size], device=device, dtypes=[torch.float])

        # Forward
        outputs = model.forward(emb_inp)
        self.assertEqual(outputs.size(), (batch_size, 1))
        print(f'Output Size: {outputs.size()}, Output Dimensions: {outputs.names}')


class NeuralFactorizationMachineModelTestCase(unittest.TestCase):
    @parameterized.expand([
        (8, 4, 128),
        (16, 6, 64),
        (32, 12, 8)
    ])
    def test_forward(self, batch_size: int, num_fields: int, embed_size: int):
        model = NeuralFactorizationMachineModel(
            embed_size=embed_size,
            deep_layer_sizes=[16, 16, 16],
            use_bias=True,
            fm_dropout_p=0.9,
            deep_dropout_p=[0.9, 0.9, 0.9],
            deep_activation=nn.ReLU()
        )
        model = model.to(device)

        # Generate inputs for the layer
        feat_inp = torch.rand(batch_size, num_fields, 1)
        feat_inp.names = ('B', 'N', 'E',)
        feat_inp_size = feat_inp.size()

        emb_inp = torch.rand(batch_size, num_fields, embed_size)
        emb_inp.names = ('B', 'N', 'E',)
        emb_inp_size = emb_inp.size()

        summary(model, input_size=[feat_inp_size, emb_inp_size], device=device, dtypes=[torch.float, torch.float])

        # Forward
        outputs = model.forward(feat_inp, emb_inp)
        self.assertEqual(outputs.size(), (batch_size, 1))
        print(f'Output Size: {outputs.size()}, Output Dimensions: {outputs.names}')


class PersonalizedReRankingModelTestCase(unittest.TestCase):
    @parameterized.expand([
        (8, 4, 128),
        (16, 6, 64),
        (32, 12, 8)
    ])
    def test_forward(self, batch_size: int, length: int, embed_size: int):
        model = PersonalizedReRankingModel(
            embed_size=embed_size,
            max_num_position=length,
            encoding_size=16,
            num_heads=4,
            num_layers=2,
            use_bias=True,
            dropout=0.9,
            fnn_dropout_p=0.9,
            fnn_activation=nn.ReLU()
        )
        model = model.to(device)

        # Generate inputs for the layer
        feat_inp = torch.randint(0, length, (batch_size, length, embed_size))
        feat_inp.names = ('B', 'L', 'E',)
        feat_inp_size = feat_inp.size()

        summary(model, input_size=[feat_inp_size], device=device, dtypes=[torch.int])

        # Forward
        outputs = model.forward(feat_inp)
        self.assertEqual(outputs.size(), (batch_size, length))
        print(f'Output Size: {outputs.size()}, Output Dimensions: {outputs.names}')


class PositionBiasAwareLearningFrameworkModelTestCase(unittest.TestCase):
    @parameterized.expand([
        (8, 4, 128),
        (16, 6, 64),
        (32, 12, 8)
    ])
    def test_forward(self, batch_size: int, num_fields: int, embed_size: int):
        max_num_position = 32
        pctr_model = NeuralFactorizationMachineModel(
            embed_size=embed_size,
            deep_layer_sizes=[16, 16, 16],
            use_bias=True,
            fm_dropout_p=0.9,
            deep_dropout_p=[0.9, 0.9, 0.9],
            deep_activation=nn.ReLU()
        )
        model = PositionBiasAwareLearningFrameworkModel(
            pctr_model=pctr_model,
            output_size=1,
            max_num_position=max_num_position,
            layer_sizes=[16, 16, 16],
            dropout_p=[0.9, 0.9, 0.9],
            activation=nn.ReLU()
        )
        model = model.to(device)

        # Generate inputs for the layer
        feat_inp = torch.rand(batch_size, num_fields, 1)
        feat_inp.names = ('B', 'N', 'E',)
        # feat_inp_size = feat_inp.size()

        emb_inp = torch.rand(batch_size, num_fields, embed_size)
        emb_inp.names = ('B', 'N', 'E',)
        # emb_inp_size = emb_inp.size()

        pos_inp = torch.randint(0, max_num_position, (batch_size,))
        pos_inp.names = ('B',)

        # summary(model, input_size=[feat_inp_size, emb_inp_size], device=device, dtypes=[torch.float, torch.float])

        # Forward
        outputs = model.forward({'feat_inputs': feat_inp, 'emb_inputs': emb_inp}, pos_inp)
        self.assertEqual(outputs.size(), (batch_size, 1))
        print(f'Output Size: {outputs.size()}, Output Dimensions: {outputs.names}')


class ProductNeuralNetworkModelTestCase(unittest.TestCase):
    @parameterized.expand([
        (8, 4, 128),
        (16, 6, 64),
        (32, 12, 8)
    ])
    def test_forward(self, batch_size: int, num_fields: int, embed_size: int):
        output_size = 1
        model = ProductNeuralNetworkModel(
            embed_size=embed_size,
            num_fields=num_fields,
            deep_layer_sizes=[64, 32, 16],
            output_size=output_size,
            prod_method='outer',
            use_bias=True,
            deep_dropout_p=[0.9, 0.9, 0.9],
            deep_activation=nn.ReLU6(),
            kernel_type='mat'
        )
        model = model.to(device)

        # Generate inputs for the layer
        feat_inp = torch.rand(batch_size, num_fields, 1)
        feat_inp.names = ('B', 'N', 'E',)
        feat_inp_size = feat_inp.size()

        emb_inp = torch.rand(batch_size, num_fields, embed_size)
        emb_inp.names = ('B', 'N', 'E',)
        emb_inp_size = emb_inp.size()

        summary(model, input_size=[feat_inp_size, emb_inp_size], device=device, dtypes=[torch.float, torch.float])

        # Forward
        outputs = model.forward(feat_inp, emb_inp)
        self.assertEqual(outputs.size(), (batch_size, output_size))
        print(f'Output Size: {outputs.size()}, Output Dimensions: {outputs.names}')


class StarSpaceModelTestCase(unittest.TestCase):
    @parameterized.expand([
        (8, 128),
        (16, 64),
        (32, 8)
    ])
    def test_forward(self, batch_size: int, embed_size: int):
        num_neg = 16
        model = StarSpaceModel(
            embed_size=embed_size,
            num_neg=num_neg,
            similarity=partial(inner_product_similarity, dim=2)
        )
        model = model.to(device)

        # Generate inputs for the layer
        samples_size = batch_size * (1 + num_neg)

        context_inp = torch.rand(samples_size, 1, embed_size)
        context_inp.names = ('B', 'N', 'E',)
        context_inp_size = context_inp.size()

        target_inp = torch.rand(samples_size, 1, embed_size)
        target_inp.names = ('B', 'N', 'E',)
        target_inp_size = target_inp.size()

        summary(model, input_size=[context_inp_size, target_inp_size], device=device, dtypes=[torch.float, torch.float])

        # Forward
        outputs = model.forward(context_inp, target_inp)
        self.assertEqual(outputs.size(), (samples_size, 1))
        print(f'Output Size: {outputs.size()}, Output Dimensions: {outputs.names}')


class WideAndDeepModelTestCase(unittest.TestCase):
    @parameterized.expand([
        (8, 4, 128),
        (16, 6, 64),
        (32, 12, 8)
    ])
    def test_forward(self, batch_size: int, num_fields: int, embed_size: int):
        output_size = 1
        model = WideAndDeepModel(
            embed_size=embed_size,
            num_fields=num_fields,
            deep_layer_sizes=[64, 32, 16],
            out_dropout_p=None,
            wide_dropout_p=None,
            deep_dropout_p=[0.9, 0.9, 0.9],
            deep_activation=nn.ReLU()
        )
        model = model.to(device)

        # Generate inputs for the layer
        feat_inp = torch.rand(batch_size, num_fields, 1)
        feat_inp.names = ('B', 'N', 'E',)
        feat_inp_size = feat_inp.size()

        emb_inp = torch.rand(batch_size, num_fields, embed_size)
        emb_inp.names = ('B', 'N', 'E',)
        emb_inp_size = emb_inp.size()

        summary(model, input_size=[feat_inp_size, emb_inp_size], device=device, dtypes=[torch.float, torch.float])

        # Forward
        outputs = model.forward(feat_inp, emb_inp)
        self.assertEqual(outputs.size(), (batch_size, output_size))
        print(f'Output Size: {outputs.size()}, Output Dimensions: {outputs.names}')


class XDeepFactorizationMachineModelTestCase(unittest.TestCase):
    @parameterized.expand([
        (8, 4, 128),
        (16, 6, 64),
        (32, 12, 8)
    ])
    def test_forward(self, batch_size: int, num_fields: int, embed_size: int):
        output_size = 1
        model = XDeepFactorizationMachineModel(
            embed_size=embed_size,
            num_fields=num_fields,
            cin_layer_sizes=[64, 32, 16],
            deep_layer_sizes=[64, 32, 16],
            cin_is_direct=False,
            cin_use_bias=True,
            cin_use_batchnorm=True,
            cin_activation=nn.ReLU6(),
            deep_dropout_p=[0.9, 0.9, 0.9],
            deep_activation=nn.ReLU()
        )
        model = model.to(device)

        # Generate inputs for the layer
        feat_inp = torch.rand(batch_size, num_fields, 1)
        feat_inp.names = ('B', 'N', 'E',)
        feat_inp_size = feat_inp.size()

        emb_inp = torch.rand(batch_size, num_fields, embed_size)
        emb_inp.names = ('B', 'N', 'E',)
        emb_inp_size = emb_inp.size()

        summary(model, input_size=[feat_inp_size, emb_inp_size], device=device, dtypes=[torch.float, torch.float])

        # Forward
        outputs = model.forward(feat_inp, emb_inp)
        self.assertEqual(outputs.size(), (batch_size, output_size))
        print(f'Output Size: {outputs.size()}, Output Dimensions: {outputs.names}')


if __name__ == '__main__':
    unittest.main()
