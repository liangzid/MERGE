




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

class activation_quad(cnn.Module):
    def __init__(self):
        super().__init__()
        self.first_coef = torch.tensor([0.125]).item()
        self.second_coef = torch.tensor([0.5]).item()
        self.third_coef = torch.tensor([0.25]).item()
        self.pow = torch.tensor([2]).item()
     
    def forward(self, x):
        return self.first_coef*x*x + self.second_coef*x + self.third_coef
        #return x*x
