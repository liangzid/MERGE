




import torch.nn as nn
import torch

class softmax_2QUAD(nn.Module):
    def __init__(self, norm, dim):
        super().__init__()
        self.dim = dim
        self.norm = norm
    
    def forward(self, x):
        a, b, c, d = x.size()
        #quad = x#self.norm(x)
        quad = (x+5) * (x+5)
        return quad / quad.sum(dim=self.dim, keepdims=True)

