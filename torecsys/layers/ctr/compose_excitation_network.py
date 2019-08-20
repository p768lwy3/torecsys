from torecsys.utils.decorator import jit_experimental
import torch
import torch.nn as nn


class ComposeExcitationNetworkLayer(nn.Module):
    r"""ComposeExcitationNetwork is a layer used in FAT-DeepFM, which is to compose the field aware embedding matrix feature-wisely 
    with Convalution 1D layer with :math:`1 * 1` kernel from a :math:`k * n` matrix of field i into :math:`k * 1` vector, and 
    concatenate all the vectors and pass to the fully-connect feed foward layers to calculate weights of attention. 
    Finally, apply the attentional weights on the inputs tensors.

    :Reference:

    #. `Junlin Zhang et al, 2019. FAT-DeepFFM: Field Attentive Deep Field-aware Factorization Machine <https://arxiv.org/abs/1905.06336>`_.

    """
    @jit_experimental
    def __init__(self, 
                 num_fields : int,
                 reduction  : int = 16):
        r"""initialize compose excitation network layer
        
        Args:
            num_fields (int): [description]
            reduction (int, optional): [description]. Defaults to 16.
        """
        super(ComposeExcitationNetworkLayer, self).__init__()

        # initialize 1d pooling layer to compose the embedding vectors
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        # initialize fully-connect layers to calculate attention
        self.fc = nn.Sequential(
            nn.Linear(num_fields ** 2, num_fields ** 2 // reduction),
            nn.ReLU(),
            nn.Linear(num_fields ** 2 // reduction, num_fields ** 2),
            nn.ReLU()
        )


    def forward(self, field_emb_inputs: torch.Tensor) -> torch.Tensor:
        r"""feed forward calculation of compose excitation network layer
        
        Args:
            field_emb_inputs (T), shape = (B, N * N, E), dtype = torch.long: field-aware embedding matrices
        
        Returns:
            T, shape = (B, N * N, E), dtype = torch.long: output of compose excitation network
        """
        # inputs' shape = (B, N * N, E) and output's shape = (B, N * N, 1)
        pooled_inputs = self.pooling(field_emb_inputs)

        # squeeze pooled_inputs into shape = (B, N * N)
        pooled_inputs = pooled_inputs.squeeze()

        # output's shape of attn_weights = (B, N * N)
        attn_weights = self.fc(pooled_inputs)

        # unsqueeze to (B, N * N, 1) and expand as x's shape = (B, N * N, E)
        attn_weights = attn_weights.unsqueeze(-1)
        outputs = field_emb_inputs * attn_weights.expand_as(field_emb_inputs)

        return outputs
