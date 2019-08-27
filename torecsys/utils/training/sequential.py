import torch
import torch.nn as nn


class Sequential(nn.Module):
    def __init__(self, embeddings, model):

        super(Sequential, self).__init__()

        self.embeddings = embeddings
        self.model = model

        for name, param in self.embeddings.named_parameters():
            self.add_module()
    
    def forward(self, inputs):

        emb_inputs = self.embeddings(inputs)
        outputs = self.model(**emb_inputs)
        return outputs
        