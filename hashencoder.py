import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

#L = 16      #Number of levels
#T = 2**14   #hash table size
#F = 2       #Number of feature dimensions per entry
#N_min = 16  #Coarsest resolution
#N_max = 512 #Finest resolution

def norm_coord(width, height):
    x, y = torch.meshgrid(torch.linspace(0,1,steps = width),torch.linspace(0,1,steps = height),indexing = 'xy')
    return torch.stack([x, y])

def HashEncoder(nn.Module):
    def __init__(self, L=16, T=2**14, F=2, N_min=16, N_max=512, interpolation='bilinear'):
        super().__init__()
        self.L = L
        self.T = T
        self.F = F
        self.interpolation = interpolation
        b = math.exp((math.log(N_max/N_min))/(L-1))
        self.N_l =  [floor(N_min*(b**l)) for l in range(L)]
        self.Pi('primes',torch.tensor([1, 2654435761]))
        self.hash_table = nn.Parameter(torch.rand([1, T, F], requires_grid=True)*2e-3-1e-3)

    @property
    def enc_size(self):
        return self.L * self.F
    
    def forward(self, x):
        