import torch
import torch.nn as nn
from layers import RadialFourierConv, PixelwiseNonlinearity, Conv2dZeros, ActNorm2d,CubicSplineRadialFourierConv
from utils import gaussian_likelihood



class InvariantFlowModel(nn.Module):
    """
    A simple normalizing flow that stacks radial Fourier conv layers
    and pixelwise nonlinearities to achieve translation+rotation invariance,
    with an optional learnable top prior.
    """
    def __init__(self,
                 image_shape=(1,64,64),
                 n_layers=4,
                 n_kernel_knots=8,
                 n_nonlinearity_knots=8,
                 learn_top=False):
        """
        image_shape: (C, H, W)
        n_layers: number of (FourierConv + Nonlinearity) pairs
        n_kernel_knots: number of knots in the FourierConv radial basis
        n_nonlinearity_knots: number of knots in the PixelwiseNonlinearity
        learn_top: if True, we learn a convolution that predicts mean, log_scale
                   of the prior in the latent space.
        """
        super().__init__()
        
        self.image_shape = image_shape
        self.n_layers = n_layers
        self.learn_top = learn_top
        self.n_kernel_knots = n_kernel_knots
        self.n_nonlinearity_knots = n_nonlinearity_knots

        # -- Build the flow layers --
        layers = []
        for _ in range(n_layers):
            layers.append(CubicSplineRadialFourierConv(image_shape=image_shape, num_knots=n_kernel_knots))
            layers.append(PixelwiseNonlinearity(ndim=image_shape[0], nknot=n_nonlinearity_knots))
        self.layers = nn.ModuleList(layers)

        # -- Register a buffer for the 'prior_h' (like Glow's top-h).
        # We'll store [1, 2*C, H, W] so that the first C is mean, second C is logs
        C, H, W = image_shape
        self.register_buffer(
            "prior_h",
            torch.zeros(1, 2*C, H, W)
        )

        # If learn_top, define a small conv that modifies prior_h => (mean, logs).
        # Otherwise, we just use zeros => (mean=0, logs=0).
        if self.learn_top:
            in_ch = 2*C
            out_ch = 2*C  # we want to produce mean & log_scale for each channel
            self.top_conv = Conv2dZeros(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        else:
            self.top_conv = None


    def prior(self, batch_size):
        """
        Returns (mean, logs) for the top-level prior distribution over z.
        Shape: (B, C, H, W) for each of mean, logs.
        """
        C, H, W = self.image_shape
        # Repeat self.prior_h for the batch dimension:
        # prior_h shape => [B, 2*C, H, W]
        h = self.prior_h.repeat(batch_size, 1, 1, 1)

        if self.learn_top:
            # pass through a zero-initialized conv
            h = self.top_conv(h)

        # Split in half: first C channels = mean, second C channels = logs
        mean, logs = torch.split(h, C, dim=1)  # each => (B, C, H, W)
        return mean, logs


    def forward(self, x=None, z=None, reverse=False, **kwargs):
        """
        Switch between normal_flow (data -> z) or reverse_flow (z -> data).
        x : (B, C, H, W) data in normal direction
        z : (B, C, H, W) latents in reverse direction; if None, sample from prior
        reverse : bool
        temperature : used when sampling from prior (z ~ N(mean, exp(logs)*temperature))
        """
        if reverse:
            return self.reverse_flow(z)
        else:
            return self.normal_flow(x)

    def normal_flow(self, x):
        """
        Forward pass: x -> z, plus the log-likelihood under the prior.
        Returns (z, log_lik), where log_lik is the sum of:
            log p(z) + sum_of_layer_logdets
        """
        b, c, h, w = x.shape

        # 1) Pass x forward through flow => z
        logdet = torch.zeros(b, device=x.device, dtype=x.dtype, requires_grad=True)
        z = x
        for layer in self.layers:
            z, logdet = layer(z, logdet=logdet, reverse=False)

        # 2) Compute log p(z) under prior
        mean, logs = self.prior(b)  # shape => (b, c, h, w) each
        # Gaussian likelihood
        log_pz = gaussian_likelihood(mean, logs, z)

        # total objective:
        #   objective = log p(z) + log|det dz/dx|
        # we store it typically as "log_lik" or "objective"
        objective = log_pz + logdet

        return z, objective

    def reverse_flow(self, z=None):
        """
        Inverse pass: z -> x
        If z is None, we sample from the prior distribution.
        Then pass z backward through flow.
        Returns x.
        """
        bsz = z.shape[0] if z is not None else None

        if z is None:
            # sample z ~ N(mean, sigma*temperature)
            # first get prior means and logs
            # but we need a batch size; let's choose something or require user to pass it
            # if user didn't pass z, we must define a batch size (e.g. 1). Or raise an error.
            raise ValueError("reverse_flow requires either a batch-size or a provided z. "
                             "Please pass z of shape (B, C, H, W).")
        else:
            bsz = z.shape[0]

        # Now invert the flow layers in reverse order
        x = z
        logdet = torch.zeros(bsz, device=z.device, dtype=z.dtype)
        for layer in reversed(self.layers):
            x, logdet = layer(x, logdet=logdet, reverse=True)

        return x, logdet