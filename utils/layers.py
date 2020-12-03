import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearSVD(nn.Module):
    """SVD implementation taken from Deep.Bayes Summer school
    """
    def __init__(self, in_features, out_features, bias=True):
        super(LinearSVD, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.Tensor(out_features, in_features))
        self.log_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(1, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.bias.data.zero_()
        self.W.data.normal_(0, 0.02)
        self.log_sigma.data.fill_(-5)        
        
    def forward(self, x):
        self.log_alpha = self.log_sigma * 2.0 - 2.0 * torch.log(1e-16 + torch.abs(self.W))
        self.log_alpha = torch.clamp(self.log_alpha, -10, 10) 
        
        if self.training:
            lrt_mean =  F.linear(x, self.W) + self.bias
            lrt_std = torch.sqrt(F.linear(x * x, torch.exp(self.log_sigma * 2.0)) + 1e-8)
            eps = lrt_std.data.new(lrt_std.size()).normal_()
            return lrt_mean + lrt_std * eps
    
        return F.linear(x, self.W * (self.log_alpha < 3).float()) + self.bias

    def compute_kl(self):
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        kl = k1*torch.sigmoid(k2 + k3*self.log_alpha)-0.5*torch.log1p(torch.exp(-self.log_alpha))
        kl = -torch.sum(kl)
        return kl
