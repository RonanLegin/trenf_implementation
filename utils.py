import torch
import math

def gaussian_likelihood(mean, logs, z):
    """
    Compute log p(z | mean, logs) = sum over all dims of
      -0.5 log(2*pi) - logs - 0.5 * ((z - mean) * exp(-logs))^2

    mean, logs, z shapes: (B, C, H, W)
    Returns a tensor of shape (B,) with the sum over all channels/pixels.
    """
    # Because logs = log(sigma), exp(-logs) = 1 / sigma.
    # We'll do elementwise below.
    log2pi = math.log(2 * math.pi)
    # log prob per element
    logp_per_elem = -0.5 * (log2pi + 2.0 * logs) - 0.5 * ((z - mean) * torch.exp(-logs))**2
    # sum over C,H,W
    logp = logp_per_elem.flatten(start_dim=1).sum(dim=1) 
    return logp