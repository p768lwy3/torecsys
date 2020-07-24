r"""torecsys.data is a module to handling data preprocessing before embedding layers, including:

#. download samples data

#. load downloaded data

#. convert data from source to torch.utils.data.Dataset

#. create DataLoader to batch Dataset for inputs and embeddings

#. create sampler to clip data before training

"""

from .dataloader import *
from .dataset import *
from .sampledata import *
from .subsampling import *
