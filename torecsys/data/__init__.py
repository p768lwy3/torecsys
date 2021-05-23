"""
torecsys.data is a model to handling data preprocessing before embedding layers, including:

#. download samples data

#. load downloaded data

#. convert data from source to torch.utils.data.Dataset

#. create DataLoader to batch Dataset for embedder and embeddings

#. create sampler to clip data before training

"""

__all__ = [
    'dataloader',
    'dataset',
    'sampledata',
    'sub_sampling'
]

import torecsys.data.dataloader
import torecsys.data.dataset
import torecsys.data.sample_data
import torecsys.data.sub_sampling
