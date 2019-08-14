import torch
import torch.nn as nn

class OuterProductNetworkLayer(nn.Module):
    r"""OuterProductNetworkLayer is a layer used in Product-based Neural Network to calculate 
    element-wise cross-feature interactions by outer-product of matrix multiplication.
    
    :Reference:

    #. `Yanru Qu et at, 2016. Product-based Neural Networks for User Response Prediction <https://arxiv.org/abs/1611.00144>`
    
    """
    def __init__(self, 
                 embed_size  : int,
                 num_fields  : int,
                 kernel_type : str = "mat"):
        r"""initialize outer product network layer module
        
        Args:
            embed_size (int): embedding size
            num_fields (int): number of fields in inputs
            kernel_type (str, optional): kernel to compress outer-product. Defaults to "mat".
        
        Raises:
            ValueError: when kernel_size is not in ["mat", "num", "vec"]
        """
        super(OuterProductNetworkLayer, self).__init__()

        # indices for outer product
        self.rowidx = list()
        self.colidx = list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                self.rowidx.append(i)
                self.colidx.append(j)
        
        # kernel for compressing outer product output
        __kernel_size__ = ["mat", "num", "vec"]
        if kernel_type == "mat":
            kernel_size = (embed_size, num_fields * (num_fields - 1) // 2, embed_size)
        elif kernel_type == "vec":
            kernel_size = (num_fields * (num_fields - 1) // 2, embed_size)
        elif kernel_type == "num":
            kernel_size = (num_fields * (num_fields - 1) // 2, 1)
        else:
            raise ValueError("kernel_type only allows: [%s]." % (", ".join(__kernel_size__)))
        self.param = nn.Parameter(torch.zeros(kernel_size))
        nn.init.xavier_normal_(self.param.data)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""feed-forward calculation of outer product network
        
        Args:
            inputs (torch.Tensor), shape = (B, N, E), dtype = torch.float: features vectors of inputs
        
        Returns:
            torch.Tensor, shape = (B, 1, NC2), dtype = torch.float: output of outer product network
        """
        # indexing data for outer product
        p = inputs[:, self.rowidx] # shape = (B, NC2, E)
        q = inputs[:, self.colidx] # shape = (B, NC2, E)

        # apply kernel on outer product
        if self.kernel_type == "mat":
            # unsqueeze p to (B, 1, NC2, E), 
            # then multiply kernel and return shape = (B, E, NC2, E)
            kp = p.unsqueeze(1) * self.param
            
            # aggregate last dimension of kp and return shape = (B, E, NC2)
            # then tranpose to shape = (B, NC2, E)
            kp = kp.sum(dim=-1).transpose(1, 2)

            # multiply q to kp and return shape = (B, NC2, E)
            # then aggregate outputs with last dimension to shape (B, NC2)
            outputs = (kp * q).sum(dim=-1)
        else:
            # multiply q and param to p and return shape = (B, NC2, E)
            # then aggregate outputs with last dimension to shape (B, NC2)
            outputs = (p * q * self.param.unsqueeze(0)).sum(dim=-1)
        
        # reshape outputs to (B, 1, NC2)
        return outputs.unsqueeze(1)
