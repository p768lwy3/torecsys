from . import _CtrModel
import torch
import torch.nn as nn
from torecsys.layers import DNNLayer, MOELayer
from torecsys.utils.decorator import jit_experimental, no_jit_experimental_by_namedtensor
from typing import Callable, List

class MultigateMixtureOfExpertsModel(_CtrModel):
    """Model class of Multi-gate Mixture-of-Experts (MMoE) model.

    Multi-gate Mixture-of-Experts is a variant of Deep MoE for multi-task learning to learn 
    relationships between tasks explicitly by multiple gate functions instead of one gate.
    
    :Reference:

    #. `Jiaqi Ma et al, 2018. Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts <https://www.kdd.org/kdd2018/accepted-papers/view/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture->_.

    """
    @no_jit_experimental_by_namedtensor
    def __init__(self,
                 embed_size         : int,
                 num_fields         : int,
                 num_tasks          : int,
                 num_experts        : int,
                 expert_output_size : int,
                 expert_layer_sizes : List[int],
                 deep_layer_sizes   : List[int],
                 expert_dropout_p   : List[float],
                 deep_dropout_p     : List[float],
                 expert_activation  : Callable[[torch.Tensor], torch.Tensor] = nn.ReLU(),
                 deep_activation    : Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()):
        # refer to parent class
        super(MultigateMixtureOfExpertsModel, self).__init__()

        # Bind num_tasks to num_tasks
        self.num_tasks = num_tasks

        # Initialize a multi-gate mixture of experts layer
        self.mmoe = MOELayer(
            inputs_size = embed_size * num_fields,
            output_size = num_experts * expert_output_size,
            num_gates   = num_tasks,
            num_experts = num_experts,
            expert_func = DNNLayer,
            expert_inputs_size = embed_size * num_fields,
            expert_output_size = expert_output_size,
            expert_layer_sizes = expert_layer_sizes,
            expert_dropout_p   = expert_dropout_p,
            expert_activation  = expert_activation
        )

        # Initialize a dictionary of tasks' models
        self.towers = nn.ModuleDict()
        for i in range(num_tasks):
            tower = DNNLayer(
                inputs_size = expert_output_size * num_experts,
                output_size = 1,
                layer_sizes = deep_layer_sizes,
                dropout_p   = deep_dropout_p,
                activation  = deep_activation
            )
            self.towers[("Tower_%d" % i)] = tower
    
    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:

        # Calculate with multi-gate mixture-of-experts forwardly
        # inputs: emb_inputs, shape = (B, N, E)
        # output: outputs, shape = (B, N = num_tasks, E = O)
        outputs = self.mmoe(emb_inputs)
        print(outputs.size())

        # Chunk outputs into num_tasks parts by dimension 1
        # inputs: outputs, shape = (B, N = num_tasks, E = O)
        # output: outputs, shape = (B, 1, E), length = num_tasks
        outputs = torch.chunk(outputs.rename(None), self.num_tasks, dim=1)

        # Initialize a list to store output of each tower
        towers_out = []

        # Calculate output of each task per tower forwardly
        for i, (tower_name, tower_module) in enumerate(self.towers.items()):
            tower_out = tower_module(outputs[i])
            tower_out.names = ("B", "N", "O")
            towers_out.append(tower_out)
        
        towers_out = torch.cat(towers_out, dim="N")
        towers_out.names = None

        return towers_out