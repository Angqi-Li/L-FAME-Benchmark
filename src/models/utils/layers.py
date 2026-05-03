import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.module import Module


class GraphConvolution(nn.Module):

    def __init__(self, in_channels, out_channels, bias=False):

        super(GraphConvolution, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        nn.init.xavier_normal_(self.weight)
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels))
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        out = torch.matmul(adj, x)
        out = torch.matmul(out, self.weight)
        if self.bias is not None:
            return out + self.bias
        else:
            return out

class Linear(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        nn.init.xavier_normal_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, inputs):
        return self.linear(inputs)

def normalize_A(A, lmax=2):
    A = F.relu(A)
    N = A.shape[0]
    device = A.device
    A = A * (torch.ones(N, N, device=device) - torch.eye(N, N, device=device))
    A = A + A.T
    d = torch.sum(A, 1)
    d = 1 / torch.sqrt((d + 1e-10))
    D = torch.diag_embed(d)
    L = torch.eye(N, N, device=device) - torch.matmul(torch.matmul(D, A), D)
    Lnorm = (2 * L / lmax) - torch.eye(N, N, device=device)
    return Lnorm


def generate_cheby_adj(L, K):
    support = []
    device = L.device
    for i in range(K):
        if i == 0:
            support.append(torch.eye(L.shape[-1], device=device))
        elif i == 1:
            support.append(L)
        else:
            temp = torch.matmul(2 * L, support[-1]) - support[-2]
            support.append(temp)
    return support