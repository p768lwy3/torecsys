# from . import _EmbEstimator
from functools import partial
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torecsys.data.negsampling import _NegativeSampler
from torecsys.functional.regularization import Regularizer
from torecsys.inputs import InputsWrapper
from torecsys.inputs.base import SingleIndexEmbedding
from torecsys.losses.ltr.pairwise_ranking_loss import HingeLoss
from torecsys.losses.ltr.pairwise_ranking_loss import TripletLoss
from torecsys.models.emb.starspace import StarSpaceModel
from torecsys.utils.training.ranking_trainer import RankingTrainer
from typing import Callable

class StarSpaceEstimator(RankingTrainer):
    def __init__(self, 
                 content_label  : str,
                 target_label   : str,
                 content_size   : int,
                 target_size    : int,
                 embedding_size : int,
                 neg_sampler    : _NegativeSampler,
                 neg_number     : int,
                 regularizer    : float = 0.1,
                 loss           : str = "hinge",
                 margin         : float = 0.5,
                 optimizer      : type = None,
                 learning_rate  : float = 0.01,
                 device         : str = "cpu",
                 epochs         : int = 5,
                 **kwargs):
        """Initialize Estimator `Starspace`.
        """
        # Initialize InputsWrapper for embeddings
        embedding_schema = {
            "context_inputs": (SingleIndexEmbedding(embedding_size, content_size), [content_label]),
            "positive_inputs": (SingleIndexEmbedding(embedding_size, target_size), [target_label]),
            "negative_inputs": (SingleIndexEmbedding(embedding_size, target_size), [target_label])
        }
        inputs_wrapper = InputsWrapper(embedding_schema)

        # Initialize StarSpaceModel. TO BE UPDATED: similarity.
        model = StarSpaceModel()

        # Initialize Regularizer.
        regularizer = Regularizer(regularizer, 2)

        # Initialize Loss.
        if loss == "hinge":
            loss_fn = HingeLoss(margin=margin)
        else:
            raise NotImplementedError("Not yet implement other functions.")

        # Initialize optimizer
        if optimizer is not None:
            optimizer = optimizer
        else:
            optimizer = partial(optim.SGD, lr=learning_rate)
        
        super(StarSpaceEstimator, self).__init__(
            inputs_wrapper = inputs_wrapper,
            model          = model, 
            neg_sampler    = neg_sampler,
            neg_number     = neg_number,
            regularizer    = regularizer,
            loss           = loss_fn,
            optimizer      = optimizer,
            epochs         = epochs,
            verboses       = kwargs.get("verboses", 2),
            log_step       = kwargs.get("log_step", 100),
            log_dir        = kwargs.get("log_dir", "logdir"),
            use_jit        = kwargs.get("use_jit", False)
        )

    def get_params(self) -> torch.Tensor:
        """Get embedding vector of the input index.
        
        Returns:
            torch.Tensor: [description]
        """
        pass

    def nearest_neighbour(self) -> torch.Tensor:
        pass

    def save_params(self):
        pass
