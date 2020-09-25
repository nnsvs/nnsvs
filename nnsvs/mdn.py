# coding: utf-8

import torch
from torch import nn

class MDNLayer(nn.Module):
    """ Mixture Density Network layer

    The input maps to the parameters of a Mixture of Gaussians (MoG) probability 
    distribution, where each Gaussian has out_dim dimensions and diagonal covariance.

    Implementation references:
    1. Mixture Density Networks by Mike Dusenberry https://mikedusenberry.com/mixture-density-networks
    2. PRML book https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/
    3. sagelywizard/pytorch-mdn https://github.com/sagelywizard/pytorch-mdn
    4. sksq96/pytorch-mdn https://github.com/sksq96/pytorch-mdn

    Arguments:
        in_dim (int): the number of dimensions in the input
        out_dim (int): the number of dimensions in the output
        num_gaussians (int): the number of mixture component
    Input:
        minibatch (B, max(T), D_in): B is the batch size and max(T) is the max frame lengths
            in this batch, and D_in is in_dim
    Output:
        pi, sigma, mu (B, max(T), D_out), (B, max(T), G, D_out), (B, max(T), G, D_out): 
            G is num_gaussians and D_out is out_dim.
            pi is a multinomial distribution of the Gaussians. 
            mu and sigma are the mean and the standard deviation of each Gaussian.
    """
    def __init__(self, in_dim, out_dim, num_gaussians=30):
        super(MDNLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_gaussians = num_gaussians

        self.pi = nn.Sequential(nn.Linear(in_dim, num_gaussians),
                                nn.Softmax(dim=1)
        )                                
        self.sigma = nn.Linear(in_dim, out_dim * num_gaussians)
        self.mu = nn.Linear(in_dim, out_dim * num_gaussians)

    def forward(self, minibatch):
        pi = self.pi(minibatch) 
        sigma = torch.exp(self.sigma(minibatch))
        sigma = sigma.view(len(minibatch), -1, self.num_gaussians, self.out_dim)        
        mu = self.mu(minibatch)
        mu = mu.view(len(minibatch), -1, self.num_gaussians, self.out_dim)
        return pi, sigma, mu

def mdn_loss(pi, sigma, mu, target):
    """Calculates the error, given the MoG parameters and the target.
    The loss is the negative log likelihood of the data given the MoG
    parameters.
    Arguments:
        pi (B, max(T), G): The multinomial distribution of the Gaussians. B is the batch size,
            max(T) is the max frame length in this batch, and G is num_gaussians of class MDNLayer.
        sigma (B, max(T) , G ,D_out): The standard deviation of the Gaussians. D_out is out_dim of class 
            MDNLayer.
        mu (B , max(T), G, D_out): The means of the Gaussians. 
        target (B,max(T), D_out): The target variables.
    Returns:
        loss (B, max(T), 1): Negative Log Likelihood of Mixture Density Networks.
    """
    # Expand the dim of target from B,max(T),D_out -> B,max(T),1,D_out -> B,max(T),G,D_out
    target = target.unsqueeze(2).expand_as(sigma)
    # Create gaussians with mean=mu and variance=sigma^2
    g = torch.distributions.Normal(loc=mu, scale=sigma)
    # p(y|x,w) = exp(log p(y|x,w))
    loss = torch.exp(g.log_prob(target))
    # Multiply along the dimension of targets variable to reduce the dim of loss
    # B, max(T), G, D_out -> B, max(T), G 
    loss = torch.prod(loss, dim=3)
    # Sum all Gaussians with weight coefficients pi
    # B, max(T), G -> B, max(T)
    loss = torch.sum(loss * pi, dim=2)
    # Calculate negative log likelihood and average it
    return torch.mean(-torch.log(loss), dim=1)

def mdn_sample_mode(pi, mu):
    """ Returns the mean of the Gaussian component whose weight coefficient is the largest
    as the most probable predictions.

    Arguments:
        pi (B, max(T), G): The multinomial distribution of the Gaussians. B is the batch size,
            G is num_gaussians of class MDNLayer.
        mu (B, max(T), G, D_out): The means of the Gaussians. 
    Returns:
        mode (B, max(T), D_out): The means of the Gaussians whose weight coefficient (pi) is the largest.
    """
    
    batch_size, max_T, _ , out_dim = mu.shape
    # Get the indexes of the largest pi 
    _, max_component = torch.max(pi, dim=2) # shape (B, max(T), 1)
    mode = torch.zeros(batch_size, max_T, out_dim)
    for i in range(batch_size):
        for j in range(max_T):
            for k in range(out_dim):
                mode[i, j, k] = mu[i, j, max_component[i, j], k]
    return mode
