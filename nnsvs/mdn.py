import torch
import torch.nn.functional as F
from torch import nn


class MDNLayer(nn.Module):
    """Mixture Density Network layer

    The input maps to the parameters of a Mixture of Gaussians (MoG) probability
    distribution, where each Gaussian has out_dim dimensions and diagonal covariance.
    If dim_wise is True, features for each dimension are modeld by independent 1-D GMMs
    instead of modeling jointly. This would workaround training difficulty
    especially for high dimensional data.

    Implementation references:
    1. Mixture Density Networks by Mike Dusenberry
    https://mikedusenberry.com/mixture-density-networks
    2. PRML book
    https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/
    3. sagelywizard/pytorch-mdn
    https://github.com/sagelywizard/pytorch-mdn
    4. sksq96/pytorch-mdn
    https://github.com/sksq96/pytorch-mdn

    Attributes:
        in_dim (int): the number of dimensions in the input
        out_dim (int): the number of dimensions in the output
        num_gaussians (int): the number of mixture component
        dim_wise (bool): whether to model data for each dimension separately
    """

    def __init__(self, in_dim, out_dim, num_gaussians=30, dim_wise=False):
        super(MDNLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_gaussians = num_gaussians
        self.dim_wise = dim_wise

        odim_log_pi = out_dim * num_gaussians if dim_wise else num_gaussians
        self.log_pi = nn.Linear(in_dim, odim_log_pi)

        self.log_sigma = nn.Linear(in_dim, out_dim * num_gaussians)
        self.mu = nn.Linear(in_dim, out_dim * num_gaussians)

    def forward(self, minibatch):
        """Forward for MDN

        Args:
            minibatch (torch.Tensor): tensor of shape (B, T, D_in)
                B is the batch size and T is data lengths of this batch,
                and D_in is in_dim.

        Returns:
            torch.Tensor: Tensor of shape (B, T, G) or (B, T, G, D_out)
                Log of mixture weights. G is num_gaussians and D_out is out_dim.
            torch.Tensor: Tensor of shape (B, T, G, D_out)
                the log of standard deviation of each Gaussians.
            torch.Tensor: Tensor of shape (B, T, G, D_out)
                mean of each Gaussians
        """
        B = len(minibatch)
        if self.dim_wise:
            # (B, T, G, D_out)
            log_pi = self.log_pi(minibatch).view(
                B, -1, self.num_gaussians, self.out_dim
            )
            log_pi = F.log_softmax(log_pi, dim=2)
        else:
            # (B, T, G)
            log_pi = F.log_softmax(self.log_pi(minibatch), dim=2)
        log_sigma = self.log_sigma(minibatch)
        log_sigma = log_sigma.view(B, -1, self.num_gaussians, self.out_dim)
        mu = self.mu(minibatch)
        mu = mu.view(B, -1, self.num_gaussians, self.out_dim)
        return log_pi, log_sigma, mu


def mdn_loss(
    log_pi, log_sigma, mu, target, log_pi_min=-7.0, log_sigma_min=-7.0, reduce=True
):
    """Calculates the error, given the MoG parameters and the target.
    The loss is the negative log likelihood of the data given the MoG
    parameters.

    Args:
        log_pi (torch.Tensor): Tensor of shape (B, T, G) or (B, T, G, D_out)
            The log of multinomial distribution of the Gaussians. B is the batch size,
            T is data length of this batch, and G is num_gaussians of class MDNLayer.
        log_sigma (torch.Tensor): Tensor of shape (B, T, G ,D_out)
            The log standard deviation of the Gaussians. D_out is out_dim of class
            MDNLayer.
        mu (torch.Tensor): Tensor of shape (B, T, G, D_out)
            The means of the Gaussians.
        target (torch.Tensor): Tensor of shape (B, T, D_out)
            The target variables.
        log_pi_min (float): Minimum value of log_pi (for numerical stability)
        log_sigma_min (float): Minimum value of log_sigma (for numerical stability)
        reduce: If True, the losses are averaged for each batch.

    Returns:
        loss (B) or (B, T): Negative Log Likelihood of Mixture Density Networks.
    """
    dim_wise = len(log_pi.shape) == 4

    # Clip log_sigma and log_pi with log_clamp_min for numerical stability
    log_sigma = torch.clamp(log_sigma, min=log_sigma_min)
    log_pi = torch.clamp(log_pi, min=log_pi_min)

    # Expand the dim of target as (B, T, D_out) -> (B, T, 1, D_out) -> (B, T,G, D_out)
    target = target.unsqueeze(2).expand_as(log_sigma)

    # Center target variables and clamp them within +/- 5SD for numerical stability.
    centered_target = target - mu
    scale = torch.exp(log_sigma)
    edge = 5 * scale
    centered_target = torch.where(centered_target > edge, edge, centered_target)
    centered_target = torch.where(centered_target < -edge, -edge, centered_target)

    # Create gaussians with mean=0 and variance=torch.exp(log_sigma)^2
    dist = torch.distributions.Normal(loc=0, scale=scale)

    log_prob = dist.log_prob(centered_target)

    if dim_wise:
        # (B, T, D_out. D_out)
        loss = log_prob + log_pi
    else:
        # Here we assume that the covariance matrix of multivariate Gaussian
        # distribution is diagonal to handle the mean and the variance in each
        # dimension separately.
        # Reference:
        # https://markusthill.github.io/gaussian-distribution-with-a-diagonal-covariance-matrix/
        # log pi(x)N(y|mu(x),sigma(x)) = log pi(x) + log N(y|mu(x),sigma(x))
        # log N(y_1,y_2,...,y_{D_out}|mu(x),sigma(x))
        #  = log N(y_1|mu(x),sigma(x))...N(y_{D_out}|mu(x),sigma(x))
        #  = \sum_{i=1}^{D_out} log N(y_i|mu(x),sigma(x))
        # (B, T, G, D_out) -> (B, T, G)
        loss = torch.sum(log_prob, dim=3) + log_pi

    # Calculate negative log likelihood.
    # Use torch.log_sum_exp instead of the combination of torch.sum and torch.log
    # (Reference: https://github.com/r9y9/nnsvs/pull/20#discussion_r495514563)
    # if dim_wise is True: (B, T, G, D_out) -> (B, T, D_out)
    # else (B, T, G) -> (B, T)
    loss = -torch.logsumexp(loss, dim=2)

    if reduce:
        # (B, T) -> (B)
        return torch.mean(loss, dim=1)
    else:
        # not averaged (for applying mask later)
        # (B, T)
        return loss
    return


# from r9y9/wavenet_vocoder/wavenet_vocoder/mixture.py
def to_one_hot(tensor, n, fill_with=1.0):
    # we perform one hot encore with respect to the last axis
    one_hot = torch.FloatTensor(tensor.size() + (n,)).zero_()
    if tensor.is_cuda:
        one_hot = one_hot.cuda()
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return one_hot


def mdn_get_most_probable_sigma_and_mu(log_pi, log_sigma, mu):
    """Return the mean and standard deviation of the Gaussian component
    whose weight coefficient is the largest as the most probable predictions.

    Args:
        log_pi (torch.Tensor): Tensor of shape (B, T, G) or (B, T, G, D_out)
            The log of multinomial distribution of the Gaussians.
            B is the batch size, T is data length of this batch,
            G is num_gaussians of class MDNLayer.
        log_sigma (torch.Tensor): Tensor of shape (B, T, G, D_out)
            The standard deviation of the Gaussians. D_out is out_dim of class
            MDNLayer.
        mu (torch.Tensor): Tensor of shape (B, T, G, D_out)
            The means of the Gaussians. D_out is out_dim of class MDNLayer.

    Returns:
        tuple: tuple of torch.Tensor
            torch.Tensor of shape (B, T, D_out). The standardd deviations
            of the most probable Gaussian component.
            torch.Tensor of shape (B, T, D_out). Means of the Gaussians.
    """
    dim_wise = len(log_pi.shape) == 4
    _, _, num_gaussians, _ = mu.shape
    # Get the indexes of the largest log_pi
    _, max_component = torch.max(log_pi, dim=2)  # (B, T) or (B, T, C_out)

    # Convert max_component to one_hot manner
    # if dim_wise: (B, T, D_out) -> (B, T, D_out, G)
    # else: (B, T) -> (B, T, G)
    one_hot = to_one_hot(max_component, num_gaussians)

    if dim_wise:
        # (B, T, G, D_out)
        one_hot = one_hot.transpose(2, 3)
        assert one_hot.shape == mu.shape
    else:
        # Expand the dim of one_hot as  (B, T, G) -> (B, T, G, d_out)
        one_hot = one_hot.unsqueeze(3).expand_as(mu)

    # Multiply one_hot and sum to get mean(mu) and standard deviation(sigma)
    # of the Gaussians whose weight coefficient(log_pi) is the largest.
    #  (B, T, G, d_out) -> (B, T, d_out)
    max_mu = torch.sum(mu * one_hot, dim=2)
    max_sigma = torch.exp(torch.sum(log_sigma * one_hot, dim=2))

    return max_sigma, max_mu


def mdn_get_sample(log_pi, log_sigma, mu):
    """Sample from mixture of the Gaussian component whose weight coefficient is
    the largest as the most probable predictions.

    Args:
        log_pi (torch.Tensor): Tensor of shape (B, T, G) or (B, T, G, D_out)
            The log of multinomial distribution of the Gaussians.
            B is the batch size, T is data length of this batch,
            G is num_gaussians of class MDNLayer.
        log_sigma (torch.Tensor): Tensor of shape (B, T, G, D_out)
            The log of standard deviation of the Gaussians.
            D_out is out_dim of class MDNLayer.
        mu (torch.Tensor): Tensor of shape (B, T, G, D_out)
            The means of the Gaussians. D_out is out_dim of class MDNLayer.

    Returns:
        torch.Tensor: Tensor of shape (B, T, D_out)
            Sample from the mixture of the Gaussian component.
    """
    max_sigma, max_mu = mdn_get_most_probable_sigma_and_mu(log_pi, log_sigma, mu)

    # Create gaussians with mean=max_mu and variance=max_log_sigma^2
    dist = torch.distributions.Normal(loc=max_mu, scale=max_sigma)

    # Sample from normal distribution
    sample = dist.sample()

    return sample
