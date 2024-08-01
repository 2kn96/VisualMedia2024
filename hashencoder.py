import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as Fnc
from torchvision.io import read_image, ImageReadMode
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#D = 2       #Dimension of input
#L = 16      #Number of levels
#T = 2**14   #hash table size
#F = 2       #Number of feature dimensions per entry
#N_min = 16  #Coarsest resolution
#N_max = 512 #Finest resolution
init_range = 1e-4

def norm_coord(width, height, depth):
    x, y, z = torch.meshgrid(torch.linspace(0,1,steps = width),torch.linspace(0,1,steps = height),torch.linspace(0,1,steps = depth),indexing = 'xy')
    return torch.stack([x, y, z])

class HashEncoder(nn.Module):
    def __init__(self, L=16, T=2**8, F=3, N_min=4, N_max=64):
        super().__init__()
        self.L = L
        self.T = T
        self.F = F
        b = math.exp((math.log(N_max/N_min))/(L-1))
        self.N_l = []    #Each numbers of divisions 
        for l in range(L):
            self.N_l.append(math.floor(N_min * (b**l)))
        self.register_buffer('pi_i',torch.tensor([1, 2654435761, 805459861]))  #table of prime
        self.hash_table = nn.Parameter(torch.rand([L, T, F], requires_grad=True)*2*init_range-init_range)

    @property
    def enc_size(self):
        return self.L * self.F
    
    def forward(self, x):
        b, c, h, w, d= x.size()
        def make_grid(x, n):
            grid = Fnc.max_pool3d(x*n, (h//n, w//n, d//n)).to(dtype=torch.long)
            grid = grid*self.pi_i.view([3, 1, 1, 1])
            grid = (grid[:,0]^grid[:,1]^grid[:,2])%self.T
            return grid

        grids = []
        for n in self.N_l:
            grids.append(make_grid(x, n))
        feat = []
        for i,g in enumerate(grids):
            feat.append(self.hash_table[i, g].permute(0, 4, 1, 3, 2))
        feat_vec = []
        for f in feat:
            feat_vec.append(Fnc.interpolate(f, (h, w, d), mode='trilinear')) #outputs feature vector
        feat_map = torch.hstack(feat_vec)
        return feat_map

class Net(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        input_size = encoder.enc_size
        self.fc1 = nn.Conv3d(input_size, 16, 1)
        self.fc2 = nn.Conv3d(16, 1, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = Fnc.sigmoid(x)
        x = self.fc2(x)
        return x

class MainModel(nn.Module):
    def __init__(self):
        super(MainModel, self).__init__()
        self.net1 = HashEncoder()
        self.net2 = Net(HashEncoder())
    
    def forward(self, x):
        x = self.net1(x)
        x = self.net2(x)
        return x

def ref_func(r):
    r = r.squeeze(0)
    x, y, z = r
    R = 60
    return ((x-60)**2 + (y-60)**2 + (z-60)**2 - R**2).unsqueeze(0)

def plot_surface(coords, f):
    # フィルタリングされた点を3Dプロット
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    print(f)
    for i in range(120):
        for j in range(120):
            for k in range(120):
                if f[0,i,j,k].item()**2<5:
                    ax.plot(i,j,k,c='b', marker='o')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_xlim=([0, 119])
    ax.set_ylim=([0, 119])
    ax.set_zlim=([0, 119])
    plt.show()

main = MainModel()

vec = norm_coord(120, 120, 120)
vec = vec.unsqueeze(0)
y = ref_func(120*vec)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam([{'params':main.net1.parameters()}, {'params': main.net2.parameters()}], lr=0.1, betas=(0.9, 0.99), eps=1e-15)

num_epoch = 100
losses = []
for i in range(num_epoch):
    main.train()
    optimizer.zero_grad()
    y_pred = main(vec).squeeze(0)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    losses.append([loss.item()])

plot_surface(vec, y_pred)
plt.plot(range(num_epoch),losses)
plt.yscale('log')
plt.title('loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()