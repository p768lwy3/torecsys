from . import _RerankingModel
import torch
import torch.nn as nn
from torecsys.layers import PositionEmbeddingLayer

class PersonalizedRerankingModel(_RerankingModel):
    r"""Model class of Personalized Re-ranking Model (PRM).

    Personalized Re-ranking model is proposed by Alibaba Group in 2019, which is to re-ranking 
    results from a ranked model with extra information, including user preference from another 
    pre-trained model and features from crossing items.

    The main mechanism of PRM are: 

    #. to concatenate user-features before encoding layers (i.e. transformer part).

    #. to use Transformer to capture the features without considering sequence order.

    Finally, sort items list by scores calculated by softmax layer at the end.

    :Reference:

    #. `Changhua Pei et al, 2019. Personalized Re-ranking for Recommendation <https://arxiv.org/abs/1904.06813>`_.

    """
    def __init__(self,
                 embed_size       : int,
                 max_num_position : int,
                 encoding_size    : int,
                 num_heads        : int,
                 num_layers       : int,
                 dropout          : float = 0,
                 use_bias         : bool = True,
                 **kwargs):
        r"""Initialize PersonalizedRerankingModel
        
        Args:
            embed_size (int): Size of embedding tensor
            max_num_position (int): Maximum length of list, i.e. Maximum number of postion
            encoding_size (int): Size of input of encoding layer
            num_heads (int): Number of heads in Multihead Attention
            num_layers (int): Number of layers of Transformer
            dropout (float, optional): Probability of Dropout in Multihead Attention. 
                Defaults to 0.
            use_bias (bool, optional): Boolean flag to use bias.
                Default to True.
        
        Arguments:
            fnn_activation (Callable[[T], T]): Activation function of feed-foward of Transformer.
            fnn_dropout_p (float): Probability of Dropout in feed-foward of Transformer.
        
        Attributes:
            layers (nn.ModuleDict): Module dictionary of Personalized Reranking Model.
        """
        # refer to parent class
        super(PersonalizedRerankingModel, self).__init__()

        # Initialize module list to store layers in model
        self.layers = nn.ModuleDict()

        # Initialize partial of input layer, i.e. position embedding (pe)
        self.layers["InputLayer"] = nn.ModuleDict()
        if use_bias:
            self.layers["InputLayer"]["PositionEmbedding"] = PositionEmbeddingLayer(
                max_num_position = max_num_position
            )
        else:
            self.layers["InputLayer"]["PositionEmbedding"] = None
        
        self.layers["InputLayer"]["FeedForward"] = nn.Linear(embed_size, encoding_size)

        # Initialize encoding layer, i.e. a stack of transformer
        self.layers["EncodingLayer"] = nn.ModuleDict()
        for i in range(num_layers):
            layer = nn.ModuleDict()

            # Initialize multihead part in attention of transformer block
            multihead = nn.MultiheadAttention(encoding_size, num_heads, dropout)
            layer["MultiheadAttention"] = multihead

            # Initialize batchnorm part in attention of transformer block
            batchnorm = nn.BatchNorm1d(max_num_position)
            layer["AttentionBatchNorm"] = batchnorm

            # Initialize feed forward part in FNN of transformer block
            feedforward = nn.Sequential()
            feedforward.add_module("FeedForward", nn.Linear(encoding_size, encoding_size))

            if kwargs.get("fnn_activation"):
                feedforward.add_module("Activation", kwargs.get("fnn_activation"))
            else:
                feedforward.add_module("Activation", nn.ReLU())
            
            if kwargs.get("fnn_dropout_p"):
                feedforward.add_module("Dropout", nn.Dropout(kwargs.get("fnn_dropout_p")))
            
            layer["Feedforward"] = feedforward
            
            # Initialize batchnorm part in FFN of transformer block
            batchnorm = nn.BatchNorm1d(max_num_position)
            layer["FNNBatchNorm"] = batchnorm

            self.layers["EncodingLayer"]["Transformer_%d" % i] = layer
        
        # Initialize output layer, i.e. linear layer + softmax layer
        self.layers["OutputLayer"] = nn.ModuleDict()
        self.layers["OutputLayer"]["FeedForward"] = nn.Linear(encoding_size, 1)
        self.layers["OutputLayer"]["Softmax"] = nn.Softmax(dim=1)

    def forward(self, feat_inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward calculation of PersonalizedRerankingModel
        
        Args:
            feat_inputs (T), shape = (B, L, E), dtype = torch.float: Features tensors.
        
        Returns:
            T, shape = (B, O), dtype = torch.float: Output of PersonalizedRerankingModel.
        """
        # 1) Input Layer Part
        # Add up feat_inputs and posiotnal embedding
        # inputs: feat_inputs, shape = (B, L, E)
        # inputs: layers["InputLayer"]["PositionEmbedding"], shape = (B, L, E)
        # output: output, shape = (B, L, E)
        if self.layers["InputLayer"]["PositionEmbedding"] is not None:
            output = self.layers["InputLayer"]["PositionEmbedding"](feat_inputs)
        else:
            output = feat_inputs
        
        # Calculate forwardly with feed-forward layer of input layer
        # inputs: output, shape = (B, L, E)
        # output: output, shape = (B, L, E')
        output = self.layers["InputLayer"]["FeedForward"](output)

        # 2) Encoding Layer Part
        for i in range(len(self.layers["EncodingLayer"])):
            # Get layer from model
            layer = self.layers["EncodingLayer"]["Transformer_%d" % i]

            # Copy output as residual
            residual_a = output

            # Compute multihead attention
            # inputs: output, shape = (B, L, E)
            # output: output, shape = (B, L, E)
            output.names = ("B", "L", "E")
            output = output.align_to("L", "B", "E")
            output.names = None
            
            output, _ = layer["MultiheadAttention"](output, output, output)
            
            output.names = ("L", "B", "E")
            output = output.align_to("B", "L", "E")
            output.names = None

            # Add residual and apply batchnorm
            # inputs: output, shape = (B, L, E)
            # inputs: input_i, shape = (B, L, E)
            # output: output, shape = (B, L, E)
            output = output + residual_a
            output = layer["AttentionBatchNorm"](output.rename(None))
            
            # Copy output as residual
            residual_f = output
            
            # Calculate forwardly with feed-forward
            # inputs: output, shape = (B, L, E)
            # output: output, shape = (B, L, E)
            output = layer["Feedforward"](output)
            
            # Add residual and apply batchnorm
            # inputs: output, shape = (B, L, E)
            # inputs: input_i, shape = (B, L, E)
            # output: output, shape = (B, L, E)
            output = output + residual_f
            output = layer["FNNBatchNorm"](output.rename(None))
        
        # 3) Output Layer Part
        # Calculate forwardly with feed-forward
        # inputs: output, shape = (B, L, E)
        # output: output, shape = (B, L, O = 1)
        output = self.layers["OutputLayer"]["FeedForward"](output)
        output.names = ("B", "L", "O")
        
        # Flatten output
        # inputs: output, shape = (B, L, O)
        # output: output, shape = (B, O = L)
        output = output.flatten(["L", "O"], "O")
        
        # Apply softmax
        # output: output, shape = (B, O)
        # output: output, shape = (B, O)
        output = self.layers["OutputLayer"]["Softmax"](output)
            
        return output
