from typing import Dict, Tuple

import torch
import torch.nn as nn

from torecsys.layers import BaseLayer


class MixtureOfExpertsLayer(BaseLayer):
    """
    Layer class of Mixture-of-Experts (MoE), which is to combine outputs of several models, each of which called
    `expert` and specialized in a different part of input space. To combine them, a gate, which is a stack of linear
    and softmax, will be trained to weight experts' outputs before return.

    :Reference:

    #. `Robert A. Jacobs et al, 1991. Adaptive Mixtures of Local Experts
    <https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf>_.

    #. `David Eigen et al, 2013. Learning Factored Representations in a Deep Mixture of Experts
    <https://arxiv.org/abs/1312.4314>`_.

    #. `Jiaqi Ma et al, 2018. Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts
    <https://www.kdd.org/kdd2018/accepted-papers/view/modeling-task-relationships-in-multi-task-learning-with-multi
    -gate-mixture->_.

    """

    @property
    def inputs_size(self) -> Dict[str, Tuple[str, ...]]:
        return {
            'inputs': ('B', 'N', 'E',)
        }

    @property
    def outputs_size(self) -> Dict[str, Tuple[str, ...]]:
        return {
            'outputs': ('B', '1', 'Number of Experts * Expert Output Size',)
        }

    def __init__(self,
                 inputs_size: int,
                 output_size: int,
                 num_experts: int,
                 expert_func: type,
                 num_gates: int = 1,
                 **kwargs):
        """
        Initialize MixtureOfExpertsLayer
        
        Args:
            inputs_size (int): input size of MOE, i.e. number of fields * size of embedding tensor
            output_size (int): output size of MOE, i.e. number of experts * output size of expert
            num_experts (int): number of expert
            expert_func (type): module of expert, e.g. trs.layers.DNNLayer
            num_gates (int): number of gates. Defaults to 1
        
        Arguments:
            expert_*: Arguments of expert model, e.g. expert_inputs_size
        
        Example:
            to initialize a mixture-of-experts layer, you can follow the below example:

            .. code-block:: python

                import torecsys as trs

                embed_size = 128
                num_fields = 4
                num_experts = 4
                expert_output_size = 16

                layer = trs.layers.MOELayer(
                    inputs_size = embed_size * num_fields,
                    outputs_size = expert_output_size * num_experts,
                    num_experts = num_experts,
                    expert_func = trs.layers.DNNLayer,
                    expert_inputs_size = embed_size * num_fields,
                    expert_output_size = expert_output_size,
                    expert_layer_sizes = [128, 64, 64]
                )
        """
        super().__init__()

        expert_kwargs = {}
        for k, v in kwargs.items():
            if k.startswith('expert_'):
                expert_kwargs[k[7:]] = v

        self.experts = nn.ModuleDict()
        for i in range(num_experts):
            self.experts[f'Expert_{i}'] = expert_func(**expert_kwargs)

        self.gates = nn.ModuleDict()
        for i in range(num_gates):
            gate = nn.Sequential()
            gate.add_module('Linear', nn.Linear(inputs_size, output_size))
            gate.add_module('Softmax', nn.Softmax())
            self.gates[f'Gate_{i}'] = gate

    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward calculation of MixtureOfExpertsLayer
        
        Args:
            emb_inputs (T), shape = (B, N, E), data_type = torch.float: embedded features tensors
        
        Returns:
            T, shape = (B, O), data_type = torch.float: output of Mixture-of-Experts.
        """
        # Name the inputs tensor for alignment
        emb_inputs.names = ('B', 'N', 'E',)

        # Flatten embed_inputs
        # inputs: embed_inputs, shape = (B, N, E)
        # output: flatten_inputs, shape = (B, N * E)
        emb_inputs = emb_inputs.flatten(('N', 'E',), 'E')
        emb_inputs.names = None

        # Initialize a list of experts' output
        experts_output = []

        for expert_name, expert_module in self.experts.items():
            # forward calculate on expert i
            # inputs: flatten_inputs, shape = (B, N * E)
            # output: outputs, shape = (B, O)
            expert_output = expert_module(emb_inputs)
            expert_output.names = ('B', 'O',)
            experts_output.append(expert_output)

        # concatenate outputs on dimension O
        # inputs: outputs, shape = (B, Oi), for i = 1, ..., n
        # output: outputs, shape = (B, num_experts * Oi = [O1, ..., On])
        experts_output = torch.cat(experts_output, dim='O')
        experts_output.names = None

        # initialize a list of gates' output
        gated_weights = []

        # calculate gated weights
        for gate_name, gate_module in self.gates.items():
            # inputs: emb_inputs, shape = (B, N, E)
            # output: gated_weights, shape = (B, num_experts * Oi)
            gated_weight = gate_module(emb_inputs)
            gated_weight.names = ('B', 'O',)
            gated_weight = gated_weight.unflatten('O', (('N', 1,), ('O', gated_weight.size('O'),),))
            gated_weights.append(gated_weight)

        # Concatenate outputs on dimension O
        # inputs: gated_weights, shape = (B, N = 1, num_experts * Oi) for i in 1:num_gates
        # output: gated_weights, shape = (B, num_gates, num_experts * Oi)
        gated_weights = torch.cat(gated_weights, dim='N')
        gated_weights = gated_weights.rename(None)

        # Apply gate weights on outputs
        # inputs: experts_output, shape = (B, num_experts * Oi)
        # inputs: gated_weights, shape = (B, num_gates, num_experts * Oi)
        # output: outputs, shape = (B, num_gates, num_experts * Oi)
        gated_experts = torch.einsum('ik,ijk->ijk', experts_output, gated_weights)
        gated_experts.names = ('B', 'N', 'O',)

        return gated_experts
