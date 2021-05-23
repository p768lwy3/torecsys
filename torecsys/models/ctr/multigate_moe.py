from typing import List, Optional

import torch
import torch.nn as nn

from torecsys.layers import DNNLayer, MOELayer
from torecsys.models.ctr import CtrBaseModel


class MultiGateMixtureOfExpertsModel(CtrBaseModel):
    """
    Model class of Multi-gate Mixture-of-Experts (MMoE) model.

    Multi-gate Mixture-of-Experts is a variant of Deep MoE for multi-task learning to learn 
    relationships between tasks explicitly by multiple gate functions instead of one gate.
    
    :Reference:

    #. `Jiaqi Ma et al, 2018. Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts
        <https://www.kdd.org/kdd2018/accepted-papers/view/
        modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture->_.

    """

    def __init__(self,
                 embed_size: int,
                 num_fields: int,
                 num_tasks: int,
                 num_experts: int,
                 expert_output_size: int,
                 expert_layer_sizes: List[int],
                 deep_layer_sizes: List[int],
                 expert_dropout_p: Optional[List[float]] = None,
                 deep_dropout_p: Optional[List[float]] = None,
                 expert_activation: Optional[nn.Module] = nn.ReLU(),
                 deep_activation: Optional[nn.Module] = nn.ReLU()):
        """
        Initialize MultiGateMixtureOfExpertsModel

        Args:
            embed_size (int): size of embedding tensor
            num_fields (int): number of inputs' fields
            num_tasks (int): number of tasks
            num_experts (int): number of experts
            expert_output_size (int): output size of expert layer
            expert_layer_sizes (List[int]): layer sizes of expert layer
            deep_layer_sizes: layer sizes of dense network
            expert_dropout_p: probability of Dropout in expert layer. Defaults to None
            deep_dropout_p: probability of Dropout in dense network. Defaults to None
            expert_activation: activation function of expert layer. Defaults to nn.ReLU()
            deep_activation: activation function of dense network. Defaults to nn.ReLU()
        """
        super().__init__()

        self.num_tasks = num_tasks
        self.moe_layer = MOELayer(
            inputs_size=embed_size * num_fields,
            output_size=num_experts * expert_output_size,
            num_gates=num_tasks,
            num_experts=num_experts,
            expert_func=DNNLayer,
            expert_inputs_size=embed_size * num_fields,
            expert_output_size=expert_output_size,
            expert_layer_sizes=expert_layer_sizes,
            expert_dropout_p=expert_dropout_p,
            expert_activation=expert_activation
        )
        self.towers = nn.ModuleDict()
        for i in range(num_tasks):
            tower = DNNLayer(
                inputs_size=expert_output_size * num_experts,
                output_size=1,
                layer_sizes=deep_layer_sizes,
                dropout_p=deep_dropout_p,
                activation=deep_activation
            )
            self.towers[f'Tower_{i}'] = tower

    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward calculation of MultiGateMixtureOfExpertsModel

        Args:
            emb_inputs (T), shape = (B, N, E), data_type = torch.float: embedded features tensors

        Returns:
            T, shape = (B, O), data_type = torch.float: output of MultiGateMixtureOfExpertsModel
        """
        # Calculate moe layer
        # inputs: emb_inputs, shape = (B, N, E)
        # output: outputs, shape = (B, N = num_tasks, E = O)
        outputs = self.moe_layer(emb_inputs)

        # Chunk outputs into num_tasks parts by dimension 1
        # inputs: outputs, shape = (B, N = num_tasks, E = O)
        # output: outputs, shape = (B, 1, E), length = num_tasks
        outputs = torch.chunk(outputs.rename(None), self.num_tasks, dim=1)

        # Initialize a list to store output of each tower
        towers_out = []

        # Calculate output of each task per tower forwardly
        for i, (tower_name, tower_module) in enumerate(self.towers.items()):
            tower_out = tower_module(outputs[i])
            tower_out.names = ('B', 'N', 'O',)
            towers_out.append(tower_out)

        # Concatenate the output of towers by dimension N
        # inputs: towers_out, shape = (B, 1, E)
        # output: towers_out, shape = (B, E)
        towers_out = torch.cat(towers_out, dim='N').sum('N')

        # since autograd does not support Named Tensor at this stage, drop the name of output tensor.
        towers_out.names = None

        return towers_out
