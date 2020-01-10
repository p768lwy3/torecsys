from .trainer import Trainer
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from tqdm.autonotebook import tqdm

class ExperimentalTrainer(Trainer):

    def __init__(self, hyper_params: List[dict]):
        self.hyper_params = hyper_params
        self.results = {}
        raise NotImplementedError("not implemented.")
    
    def gridsearch_fit(self):
        # initialize results container and get hyperparameters of gridsearchs' step
        self.gridsearch_step()
        self.summmary()
    
    def gridsearch_step(self, search_id: int):

        # Get hyperparameters of gridsearchs' step
        hyper_params = hyper_params[search_id]
        
        # Initialize results containers
        self.results[search_id] = dict()

        # build model and fit
        model = self.model_build(**hyper_params)
        results = model.fit()

        return results
    
    def model_build(self):
        return

    def fit(self):
        return
    
    def summary(self):
        return
