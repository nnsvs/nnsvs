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
        minibatch (BxD): B is the batch size and D is in_dim
    Output:
        (pi, sigma, mu) (BxG, BxGxO, BxGxO): G is num_gaussians and O is out_dim.
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

    def forward(self, x):
        # p(z_k = 1) for all k; num_gaussians mixing components that sum to 1
        pi = self.pi(x) 
        # mixtuer_num * out_dim gaussian variances, which must be >= 0
        sigma = torch.exp(self.sigma(x))
        sigma = sigma.view(-1, self.num_gaussians, self.out_dim)        
        # num_gaussians * out_dim  gaussian means        
        mu = self.mu(x)
        mu = mu.view(-1, self.num_gaussians, self.out_dim)
        return pi, sigma, mu

def mdn_loss(pi, sigma, mu, target):
    """Calculates the error, given the MoG parameters and the target
    The loss is the negative log likelihood of the data given the MoG
    parameters.
    Arguments:
        pi (BxG): The multinomial distribution of the Gaussians. B is the batch size,
            G is num_gaussians of class MDNLayer.
        sigma (BxGxO): The standard deviation of the Gaussians. O is out_dim of class 
            MDNLayer.
        mu (BxGxO): The means of the Gaussians. 
        target (BxO): The target variables.
    Returns:
        loss (scalar): Negative Log Likelihood of Mixture Density Networks.
    """
    # Expand the dim of target from BxO -> Bx1xO -> BxGxO
    target = target.unsqueeze(1).expand_as(sigma)
    # Create gaussian with mean=mu and variance=sigma^2
    g = torch.distributions.Normal(loc=mu, scale=sigma)
    # p(y|x,w) = exp(log p(y|x,w))
    loss = torch.exp(g.log_prob(target))
    # Multiply along the dimension of target variable to reduce the dim of loss
    # to get scalar loss
    # BxGxO -> BxG 
    loss = torch.prod(loss, dim=2)
    # Sum all gaussian components with weight coefficient pi
    loss = torch.sum(loss * pi, dim=1)
    # Calculate negative log likelihood and average it
    return torch.mean(-torch.log(loss))

def mdn_sample_mode(pi, sigma, mu):
    """ Returns the mean of the Gaussian component whose weight coefficient is the largest
    as the most probable predictions.

    Arguments:
        pi (BxG): The multinomial distribution of the Gaussians. B is the batch size,
            G is num_gaussians of class MDNLayer.
        sigma (BxGxO): The standard deviation of the Gaussians. O is out_dim of class 
            MDNLayer.
        mu (BxGxO): The means of the Gaussians. 
    Returns:
        mode (BxO): The mean of the Gaussian component whose weight coefficient is the largest.
    """
    
    batch_size, _ , out_dim = mu.shape
    # Get the indexes of the largest pi 
    max_component = torch.max(pi, dim=1) # shape (Bx1)
    mode = torch.zeros(batch_size, out_dim)
    for i in range(batch_size):
        for j in range(out_dim):
            mode[i, j] = mu[i, max_component[i], j]
    return mode
