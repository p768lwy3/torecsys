import unittest

import torch
import torch.nn as nn
import torchinfo

from torecsys.inputs import *


class InputsTestCase(unittest.TestCase):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    def test_inputs(self):
        # initialize embedding layers used in Input
        emb_0_field_size = 128
        emb_0 = SingleIndexEmbedding(
            embed_size=32,
            field_size=emb_0_field_size,
            padding_idx=0
        )
        emb_0.set_schema(["userId"])

        emb_1_field_sizes = [64, 8]
        emb_1 = MultiIndicesEmbedding(
            embed_size=32,
            field_sizes=emb_1_field_sizes,
            flatten=True
        )
        emb_1.set_schema(['movieId', 'moveTypeId'])

        embedder = Inputs(schema={
            'user': emb_0,
            'movie': emb_1,
        })
        embedder = embedder.to(self.device)

        batch_size = 8
        inp_user_id = torch.randint(0, emb_0_field_size, (batch_size,), device=self.device, dtype=torch.long)
        inp_movie_id = torch.randint(0, emb_1_field_sizes[0], (batch_size, 1,), device=self.device, dtype=torch.long)
        inp_movie_type_id = torch.randint(0, emb_1_field_sizes[1], (batch_size, 1,), device=self.device,
                                          dtype=torch.long)
        inp = {
            'userId': inp_user_id,
            'movieId': inp_movie_id,
            'moveTypeId': inp_movie_type_id
        }
        for k, v in inp.items():
            print(f'Input {k} Size: {v.size()}')

        out = embedder(inp)
        for k, v in out.items():
            print(f'Output {k} Size: {v.size()}')

    def test_concat_inputs(self):
        # initialize embedding layers used in ConcatInput
        emb_0_field_size = 128
        emb_0 = SingleIndexEmbedding(
            embed_size=32,
            field_size=emb_0_field_size,
            padding_idx=0
        )
        emb_0.set_schema(['userId'])

        emb_1_field_sizes = [64, 8]
        emb_1 = MultiIndicesEmbedding(
            embed_size=32,
            field_sizes=emb_1_field_sizes,
            flatten=True
        )
        emb_1.set_schema(['movieId', 'moveTypeId'])

        embedder = ConcatInput(
            inputs=[
                emb_0,
                emb_1,
            ]
        )
        embedder = embedder.to(self.device)

        batch_size = 8
        inp_user_id = torch.randint(0, emb_0_field_size, (batch_size,), device=self.device, dtype=torch.long)
        inp_movie_id = torch.randint(0, emb_1_field_sizes[0], (batch_size, 1,), device=self.device, dtype=torch.long)
        inp_movie_type_id = torch.randint(0, emb_1_field_sizes[1], (batch_size, 1,), device=self.device,
                                          dtype=torch.long)
        inp = {
            'userId': inp_user_id,
            'movieId': inp_movie_id,
            'moveTypeId': inp_movie_type_id
        }
        for k, v in inp.items():
            print(f'Input {k} Size: {v.size()}')

        out = embedder(inp)
        print(f'Output Size: {out.size()}')

    def test_image_inputs(self):
        embedder = ImageInput(
            embed_size=128,
            in_channels=3,
            layers_size=[128, 128, 128],
            kernels_size=[3, 3, 3],
            strides=[1, 1, 1],
            paddings=[1, 1, 1],
            pooling='max_pooling',
            use_batchnorm=True,
            dropout_p=0.2,
            activation=nn.ReLU()
        )
        embedder = embedder.to(self.device)

        batch_size = 8
        num_channels = 3
        height = 256
        weight = 256
        inp = torch.rand(batch_size, num_channels, height, weight)
        inp = inp.to(self.device)
        inp_size = inp.size()
        print(f'Input Size: {inp_size}')

        torchinfo.summary(embedder, input_size=list(inp_size))

        out = embedder(inp)
        print(f'Output Size: {out.size()}')

    def test_list_indices_embedding(self):
        field_size = 256
        batch_size = 16
        length = 32

        embedder = ListIndicesEmbedding(
            embed_size=128,
            field_size=field_size,
            padding_idx=0,
            use_attn=True,
            output_method='max_pooling'
        )
        embedder = embedder.to(self.device)

        inp = torch.randint(0, field_size, size=(batch_size, length,), dtype=torch.long, device=self.device)
        inp_size = inp.size()
        print(f'Input Size: {inp_size}')

        torchinfo.summary(embedder, input_size=list(inp_size), device=self.device, dtypes=[torch.long])

        out = embedder(inp)
        print(f'Output Size: {out.size()}')

    def test_multi_indices_embedding(self):
        field_sizes = [256, 128, 64]
        embedder = MultiIndicesEmbedding(
            embed_size=128,
            field_sizes=field_sizes,
            device=self.device,
            flatten=True
        )

        batch_size = 16
        inp_1 = torch.randint(0, field_sizes[0], size=(batch_size, 1,), dtype=torch.long, device=self.device)
        inp_2 = torch.randint(0, field_sizes[1], size=(batch_size, 1,), dtype=torch.long, device=self.device)
        inp_3 = torch.randint(0, field_sizes[2], size=(batch_size, 1,), dtype=torch.long, device=self.device)
        inp = torch.cat((inp_1, inp_2, inp_3,), dim=1)
        inp_size = inp.size()
        print(f'Input Size: {inp_size}')

        torchinfo.summary(embedder, input_size=list(inp_size), device=self.device, dtypes=[torch.long])

        out = embedder(inp)
        print(f'Output Size: {out.size()}')

    def test_multi_indices_field_aware_embedding(self):
        field_sizes = [256, 128, 64]
        embedder = MultiIndicesFieldAwareEmbedding(
            embed_size=128,
            field_sizes=field_sizes,
            device=self.device,
            flatten=False
        )

        batch_size = 16
        inp_1 = torch.randint(0, field_sizes[0], size=(batch_size, 1,), dtype=torch.long, device=self.device)
        inp_2 = torch.randint(0, field_sizes[1], size=(batch_size, 1,), dtype=torch.long, device=self.device)
        inp_3 = torch.randint(0, field_sizes[2], size=(batch_size, 1,), dtype=torch.long, device=self.device)
        inp = torch.cat((inp_1, inp_2, inp_3,), dim=1)
        inp_size = inp.size()
        print(f'Input Size: {inp_size}')

        torchinfo.summary(embedder, input_size=list(inp_size), device=self.device, dtypes=[torch.long])

        out = embedder(inp)
        print(f'Output Size: {out.size()}')

    def test_pretrained_image_inputs(self):
        embedder = PretrainedImageInput(
            embed_size=128,
            model_name='vgg16',
            pretrained=True,
            no_grad=False,
            verbose=False,
            device=self.device
        )

        batch_size = 16
        image_size = [256, 256]
        channel_size = 3
        inp = torch.rand(batch_size, channel_size, image_size[0], image_size[1])
        inp_size = inp.size()
        print(f'Input Size: {inp_size}')

        torchinfo.summary(embedder, input_size=list(inp_size), device=self.device)

        out = embedder(inp)
        print(f'Output Size: {out.size()}')

    def test_sequence_indices_embedding(self):
        field_size = 256
        embedder = SequenceIndicesEmbedding(
            embed_size=128,
            field_size=field_size,
            padding_idx=0,
            rnn_method='gru',
            output_method='avg_pooling'
        )
        embedder = embedder.to(self.device)

        batch_size = 16
        length = 32
        inp = torch.randint(0, field_size, (batch_size, length,), dtype=torch.long, device=self.device)
        lengths = torch.randint(1, length, (batch_size,), dtype=torch.long, device=self.device)
        inp_size = inp.size()
        # lengths_size = lengths.size()
        print(f'Input Size: {inp_size}')

        # torchinfo.summary(embedder, input_size=[inp_size, lengths_size], device=self.device,
        #                   dtypes=[torch.long, torch.long])

        out = embedder(inp, lengths)
        print(f'Output Size: {out.size()}')

    def test_single_index_embedding(self):
        field_size = 256
        embedder = SingleIndexEmbedding(
            embed_size=128,
            field_size=field_size,
            padding_idx=0
        )
        embedder = embedder.to(self.device)

        batch_size = 16
        inp = torch.randint(0, field_size, (batch_size, 1,), dtype=torch.long, device=self.device)
        inp_size = inp.size()
        print(f'Input Size: {inp_size}')

        torchinfo.summary(embedder, input_size=list(inp_size), device=self.device, dtypes=[torch.long])

        out = embedder(inp)
        print(f'Output Size: {out.size()}')

    def test_stacked_inputs(self):
        # initialize embedding layers used in StackedInput
        emb_0_field_size = 128
        emb_0 = SingleIndexEmbedding(
            embed_size=32,
            field_size=emb_0_field_size,
            padding_idx=0
        )
        emb_0.set_schema(['userId'])

        emb_1_field_sizes = [64, 8]
        emb_1 = MultiIndicesEmbedding(
            embed_size=16,
            field_sizes=emb_1_field_sizes,
            flatten=True
        )
        emb_1.set_schema(['movieId', 'moveTypeId'])

        embedder = StackedInput(
            inputs=[
                emb_0,
                emb_1,
            ]
        )
        embedder = embedder.to(self.device)

        batch_size = 8
        inp_user_id = torch.randint(0, emb_0_field_size, (batch_size,), device=self.device, dtype=torch.long)
        inp_movie_id = torch.randint(0, emb_1_field_sizes[0], (batch_size, 1,), device=self.device, dtype=torch.long)
        inp_movie_type_id = torch.randint(0, emb_1_field_sizes[1], (batch_size, 1,), device=self.device,
                                          dtype=torch.long)
        inp = {
            'userId': inp_user_id,
            'movieId': inp_movie_id,
            'moveTypeId': inp_movie_type_id
        }
        for k, v in inp.items():
            print(f'Input {k} Size: {v.size()}')

        out = embedder(inp)
        print(f'Output Size: {out.size()}')

    def test_value_inputs(self):
        batch_size = 8
        num_fields = 4
        inputs_fn = ValueInput(
            num_fields=num_fields,
            transforms=None
        )
        inputs_fn = inputs_fn.to(self.device)

        inp = torch.rand(batch_size, num_fields)
        inp_size = inp.size()
        print(f'Input Size: {inp_size}')

        torchinfo.summary(inputs_fn, input_size=list(inp_size))

        out = inputs_fn(inp)
        print(f'Output Size: {out.size()}')


if __name__ == '__main__':
    unittest.main()
