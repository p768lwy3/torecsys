from typing import List, Optional

import torch
import torch.nn as nn

from torecsys.layers import DNNLayer, MOELayer
from torecsys.models.ctr import CtrBaseModel


class DeepMixtureOfExpertsModel(CtrBaseModel):
    """
    Model class of Deep Mixture-of-Experts (MoE) model.

    Deep Mixture-of-Experts is purposed by David Eigen et at at 2013, which is to combine outputs of several `expert`
    models, each of which specializes in a different part of input space. To combine them, a gate, which is a stack of
    linear and softmax, will be trained for weighing outputs of expert before return.

    :Reference:

    #. `David Eigen et al, 2013. Learning Factored Representations in a Deep Mixture of Experts
    <https://arxiv.org/abs/1312.4314>`_.

    """

    def __init__(self,
                 embed_size: int,
                 num_fields: int,
                 num_experts: int,
                 moe_layer_sizes: List[int],
                 deep_layer_sizes: List[int],
                 deep_dropout_p: Optional[List[float]] = None,
                 deep_activation: Optional[nn.Module] = nn.ReLU()):
        """
        Initialize DeepMixtureOfExpertsModel
        
        Args:
            embed_size (int): size of embedding tensor
            num_fields (int): number of inputs' fields
            num_experts (int): number of experts' model
            moe_layer_sizes (List[int]): size of mixture-of-experts models' outputs
            deep_layer_sizes (List[int]): layer sizes of dense network in expert's model
            deep_dropout_p (List[float], optional): probability of Dropout in dense network in expert's model.
                Defaults to None
            deep_activation (torch.nn.Module, optional): activation function of dense network in expert's model.
                Defaults to nn.ReLU()
        """
        super().__init__()

        inputs_size = embed_size * num_fields
        layer_sizes = [inputs_size] + moe_layer_sizes

        self.module = nn.ModuleList()
        for i, (inp, out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            inp = num_experts * inp if i != 0 else inp
            moe = MOELayer(
                inputs_size=inp,
                output_size=num_experts * out,
                num_experts=num_experts,
                expert_func=DNNLayer,
                expert_inputs_size=inp,
                expert_output_size=out,
                expert_layer_sizes=deep_layer_sizes,
                expert_dropout_p=deep_dropout_p,
                expert_activation=deep_activation
            )
            self.module.append(moe)

    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward calculation of DeepMixtureOfExpertsModel
        
        Args:
            emb_inputs (T), shape = (B, N, E), data_type = torch.float: embedded features tensors
        
        Returns:
            T, shape = (B, O), data_type = torch.float: output of DeepMixtureOfExpertsModel
        """
        for moe in self.module:
            # Calculate with mixture-of-experts layer forwardly
            # inputs: emb_inputs, shape = (B, N, E)
            # output: emb_inputs, shape = (B, N = 1, E = O)
            emb_inputs = moe(emb_inputs).rename(O='E')

        # Aggregate emb_inputs on dimension E = O
        # inputs: emb_inputs, shape = (B, N, E)
        # output: output, shape = (B, O = 1)
        output = emb_inputs.sum(dim='E').rename(N='O')

        # Drop names of outputs, since auto grad doesn't support NamedTensor yet.
        output = output.rename(None)

        return output
