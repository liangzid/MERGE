import math
import torch

import crypten
import crypten.communicator as comm
import crypten.nn as cnn

def encrypt_tensor(input,config):
    """input: the plaintext input tensor"""
    """Encrypt data tensor for multi-party setting"""
    device=config.device
    # get rank of current process
    rank = comm.get().get_rank()
    # get world size
    world_size = comm.get().get_world_size()
    assert world_size  == 2
    
    # assumes party 1 is the actual data provider
    src_id = 1

    if rank == src_id:
        input_upd = input.to(device)
    else:
        input_upd = torch.empty(input.size()).to(device)
    private_input = crypten.cryptensor(input_upd, src=src_id)
#    print(private_input)
    return private_input

def encrypt_model(model, modelFunc, config, dummy_input):
    """
    model: model for encryption
    modelFunc: model initilization function
    config: model configuration information
    dummy_input: unused
    """
    rank = comm.get().get_rank()
    device=config[0].device
    
    # assumes party 0 is the actual model provider
    if rank == 0:
        model_upd = model.to(device)
    else:
        if isinstance(config, tuple):
            model_upd = modelFunc(config[0], config[1]).to(device)
        else:
            model_upd = modelFunc(config).to(device)

    private_model = model_upd.encrypt(src=0)
    return private_model


class softmax_2RELU(cnn.Module):
    def __init__(self, dim):
        super().__init__()
        self.func = cnn.ReLU()
        self.dim = dim

    def forward(self, x):
        func_x = self.func(x)
        return func_x / func_x.sum(keepdim=True, dim=self.dim)

class softmax2RELU_2(cnn.Module):
    def __init__(self, dim):
        super().__init__()
        self.func = cnn.ReLU()
        self.layer1=cnn.Linear(1,16)
        self.layer2=cnn.Linear(16,16)
        self.layer3=cnn.Linear(16,1)
        self.dim = dim

    def forward(self, x):
        func_x = self.func((x/2+1).pow(3)).sum(keepdim=True,dim=self.dim)
        # print(func_x.shape)
        func_x=self.layer3(self.layer2(self.layer1(func_x)))
        return x*func_x

class softmax_2QUAD(cnn.Module):
    def __init__(self, norm, dim):
        super().__init__()
        self.dim = dim
        self.norm = norm
    
    def forward(self, x):
        a, b, c, d = x.size()
        #quad = x#self.norm(x)
        quad = (x+5) * (x+5)
        return quad / quad.sum(dim=self.dim, keepdims=True)

class activation_newGeLU(cnn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super().__init__()
        self.half = torch.tensor([0.5]).item()
        self.one = torch.tensor([1.0]).item()
        self.three = torch.tensor([3.0]).item()
        self.constant = torch.tensor([0.044715]).item()
        self.pi_const = torch.tensor([math.sqrt(2/math.pi)]).item()
        self.pow = cnn.Pow()
        self.tanh = cnn.Hardtanh()

    def forward(self, x):
        return self.half * x * (self.one + self.tanh(self.pi_const * (x + self.constant * self.pow((x, self.three)))))


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

