from torecsys.utils.decorator import jit_experimental
import torch
import torch.nn as nn
from typing import Callable, List


class CompressInteractionNetworkLayer(nn.Module):
    r"""Layer class of Compress Interation Network used in xDeepFM :title:`Jianxun Lian et al, 2018`[1], 
    which is to compress cross-features tensors calculated by element-wise cross features interactions 
    with outer product by Convalution with a :math:`1 * 1` kernel.

    :Reference:

    #. `Jianxun Lian et al, 2018. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems <https://arxiv.org/abs/1803.05170.pdf>`_.

    """
    @jit_experimental
    def __init__(self, 
                 embed_size    : int,
                 num_fields    : int,
                 output_size   : int,
                 layer_sizes   : List[int],
                 is_direct     : bool = False,
                 use_bias      : bool = True,
                 use_batchnorm : bool = True,
                 activation    : Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()):
        r"""Initialize CompressInteractionNetworkLayer
        
        Args:
            embed_size (int): Size of embedding tensor
            num_fields (int): Number of inputs' fields
            output_size (int): Output size of compress interaction network
            layer_sizes (List[int]): Layer sizes of compress interaction network
            is_direct (bool, optional): Whether outputs is passed to next step directly or not.
                Defaults to False.
            use_bias (bool, optional): Whether bias added to Conv1d or not. 
                Defaults to True.
            use_batchnorm (bool, optional): Whether batch normalization is applied or not after Conv1d. 
                Defaults to True.
            activation (Callable[[T], T], optional): Activation function of Conv1d. 
                Allow: [None, Callable[[T], T]]. 
                Defaults to nn.ReLU().
        
        Attributes:
            embed_size (int): Size of embedding tensor.
            layers_sizes (List[int]): List of integer concatenated by num_fields and layer_sizes.
            is_direct (bool): Flag to show outputs is passed to next step directly or not.
            model (torch.nn.ModuleList): Module List of compress interaction network.
            fc (torch.nn.Module): Fully-connect layer (i.e. Linear layer) of outputs.
        """
        # refer to parent class
        super(CompressInteractionNetworkLayer, self).__init__()

        # bind embed_size, is_direct to embed_size, is_direct
        self.embed_size = embed_size
        self.is_direct = is_direct
        
        # set layer_sizes to list concatenated by num_fields and layer_sizes
        self.layer_sizes = [num_fields] + layer_sizes

        # initialize module list of model
        self.model = nn.ModuleList()

        # add modules including conv1d, batchnorm and activation to model
        for i, (s_i, s_j) in enumerate(zip(self.layer_sizes[:-1], self.layer_sizes[1:])):
            # create a nn.Sequential to store a layer of cin
            cin = nn.Sequential()

            # calculate in channel and out channel of the 1D convalutional in cin
            in_c = self.layer_sizes[0] * s_i
            if is_direct or i == (len(self.layer_sizes) - 1):
                out_c = s_j
            else:
                out_c = s_j * 2
            
            # add 1D convalutional to the sequential module
            cin.add_module("conv1d", nn.Conv1d(in_c, out_c, kernel_size=1, bias=use_bias))

            # add batch norm to the sequential module
            if use_batchnorm:
                cin.add_module("batchnorm", nn.BatchNorm1d(out_c))
            
            # add activation to the sequential module
            if activation is not None:
                cin.add_module("activation", activation)
            
            # add cin to the module list
            self.model.append(cin)
        
        # calculate output size of model for the argument of fc
        model_output_size = int(sum(layer_sizes))

        # initialize linear layer of fully-connect outputs
        self.fc = nn.Linear(model_output_size, output_size)

    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward calculation of CompressInteractionNetworkLayer
        
        Args:
            emb_inputs (T), shape = (B, N, E), dtype = torch.float: Embedded features tensors.
        
        Returns:
            T, shape = (B, 1, O), dtype = torch.float: Output of CompressInteractionNetworkLayer.
        """
        # initialize lists to store tensors temporarily for outputs and next steps
        direct_list = list()
        hidden_list = list()

        # transpose x0 from (B, N, E) to (B, E, N)
        ## x0 = emb_inputs.transpose(1, 2)
        x0 = emb_inputs.align_to("B", "E", "N")
        hidden_list.append(x0)
        
        # expand x0 to (B, E, N_x, H = 1)
        ## x0 = x0.unsqueeze(-1)
        x0 = x0.unflatten("N", [("Nx", x0.size("N")), ("H", 1)])

        # loop through 1d conv layers list
        for i, layer_size in enumerate(self.layer_sizes[:-1]):
            # get (i - 1) th hidden layer with shape = (B, E, N),
            # then expand the shape to (B, E, H = 1, N_y)
            ## xi = hidden_list[-1].unsqueeze(2)
            xi = hidden_list[-1]
            xi = xi.unflatten("N", [("H", 1), ("Ny", xi.size("N"))])

            # calculate outer product,  where the shape = (B, E, N_x, N_y), by torch.matmul
            out_prod = torch.matmul(x0, xi)
            
            # then reshape the outer product to (B, N_x * N_y, E)
            ## out_prod = out_prod.view(-1, self.embed_size, layer_size * self.layer_sizes[0])
            ## out_prod = out_prod.transpose(1, 2)
            out_prod = out_prod.flatten(["Nx", "Ny"], "N")
            out_prod = out_prod.align_to("B", "N", "E")

            # apply convalution, batchnorm and activation, 
            # and the shape will be (B, H_i * 2 or H_i, N)
            outputs = self.model[i](out_prod.rename(None))
            outputs.names = ("B", "N", "E")
            
            if self.is_direct:
                # pass the whole tensors as hidden value to next step 
                # and also keep the whole tensors for outputs with shape = (B, N, H_i)
                direct = outputs
                ## hidden = outputs.transpose(1, 2)
                hidden = outputs.align_to("B", "E", "N")
            else:
                if i != (len(self.layer_sizes) - 1):
                    # pass half tensors to next step and keep half tensors
                    # for outputs with shape = (B, N, H_i) if not in last step
                    direct, hidden = torch.chunk(outputs, 2, dim=1)
                    ## hidden = hidden.transpose(1, 2)
                    hidden = hidden.align_to("B", "E", "N")
                else:
                    # keep the whole tensors for the outputs
                    direct = outputs
                    hidden = 0

            # store tensors to direct and hidden
            direct_list.append(direct)
            hidden_list.append(hidden)
        
        # concatenate direct_list into a tensor for output with shape = (B, E_1 + ... + E_k, N)
        ## outputs = torch.cat(direct_list, dim=1)
        outputs = torch.cat(direct_list, dim="N")

        # aggregate outputs into (B, E_1 + ... + E_k) and pass into output fc layer
        ## outputs = self.fc(outputs.sum(dim=-1))
        outputs = self.fc(outputs.sum("E"))
        outputs.names = ("B", "O")

        # unsqueeze outputs to (B, 1, O)
        ## outputs = outputs.unsqueeze(1)

        return outputs
