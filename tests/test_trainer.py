import unittest

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.utils.data

from torecsys.data.dataloader.collate_fn import CollateFunction
from torecsys.data.dataset import DataFrameToDataset
from torecsys.inputs import MultiIndicesEmbedding
from torecsys.trainer.torecsys_pipeline import TorecsysPipeline as Module
from torecsys.trainer.torecsys_trainer import TorecsysTrainer as Trainer


class TrainerTestCase(unittest.TestCase):
    def test_train(self):
        user_field_size = 64
        movie_field_size = 512

        embed_size = 32

        total_train_samples = 16
        total_val_samples = 4

        columns = ['user_id', 'movie_id', 'labels']
        train_user_id = np.random.randint(0, user_field_size, size=(total_train_samples, 1,), dtype=np.int64)
        train_movie_id = np.random.randint(0, movie_field_size, size=(total_train_samples, 1,), dtype=np.int64)
        train_score = np.random.rand(total_train_samples, 1)
        train_df = pd.DataFrame(np.hstack((train_user_id, train_movie_id, train_score,)), columns=columns)

        val_user_id = np.random.randint(0, user_field_size, size=(total_val_samples, 1,), dtype=np.int64)
        val_movie_id = np.random.randint(0, movie_field_size, size=(total_val_samples, 1,), dtype=np.int64)
        val_score = np.random.rand(total_val_samples, 1)
        val_df = pd.DataFrame(np.hstack((val_user_id, val_movie_id, val_score,)), columns=columns)

        train_dataset = DataFrameToDataset(train_df, columns=columns)
        val_dataset = DataFrameToDataset(val_df, columns=columns)

        user_id = 'userId'
        movie_id = 'movieId'
        target_field_name = 'target'

        schema = {
            user_id: [columns[0], 'indices'],
            movie_id: [columns[1], 'indices'],
            target_field_name: [columns[2], 'values']
        }
        collate_function = CollateFunction(schema=schema)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=0,
                                                       collate_fn=collate_function.to_tensor)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=0,
                                                     collate_fn=collate_function.to_tensor)

        feat_inp = MultiIndicesEmbedding(embed_size=1, field_sizes=[user_field_size, movie_field_size])
        emb_inp = MultiIndicesEmbedding(embed_size=embed_size, field_sizes=[user_field_size, movie_field_size])

        feat_inp.set_schema([user_id, movie_id])
        emb_inp.set_schema([user_id, movie_id])

        module = Module.build(
            objective=Module.MODULE_TYPE_CTR,
            inputs_config={
                'feat_inputs': feat_inp,
                'emb_inputs': emb_inp,
            },
            model_config={
                'method': 'WideAndDeepModel',
                'embed_size': embed_size,
                'num_fields': 2,
                'deep_layer_sizes': [16, 16, 8],
                'out_dropout_p': 0.9,
                'wide_dropout_p': 0.9,
                'deep_dropout_p': [0.9, 0.9, 0.9],
                'deep_activation': nn.ReLU()
            },
            regularizer_config={
                'weight_decay': 0.01,
                'norm': 2
            },
            criterion_config={
                'method': 'MSELoss',
                'size_average': True,
                'reduce': True,
                'reduction': 'mean'
            },
            optimizer_config={
                'method': 'AdamW',
                'lr': 0.001,
                'betas': (0.9, 0.999),
                'eps': 1e-08,
                'weight_decay': 0.01,
                'amsgrad': True
            },
            targets_name=target_field_name
        )

        trainer = Trainer(auto_select_gpus=True, max_epochs=1)
        trainer.fit(module, train_dataloader, val_dataloader)

        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
