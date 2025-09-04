import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP:

    def __init__(self, in_features, out_features, hidden_dims):
        super().__init__()
        dim_seq= [in_features, *hidden_dims, out_features]
        self.layers = nn.ModuleList([
            nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip(dim_seq, dim_seq[1:])
        ])

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.relu(x)
        x = self.layers[-1](x)
        return x
