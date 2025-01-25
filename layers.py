import torch
import torch.nn as nn
import torch.fft as fft

class Conv2dZeros(nn.Module):
    """
    Convolution layer with zero-initialized weights and biases.
    This is often used in normalizing flows (like Glow) to allow
    the network to start close to the identity transform.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True)

        # Zero-init
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)


def catmull_rom_spline_1d(x, knot_x, knot_y):
    """
    Evaluate a 1D Catmull-Rom spline at points `x` in [0,1], given:
      - knot_x: shape [M] (fixed, equally spaced in [0,1])
      - knot_y: shape [M] (trainable values)
    Returns an interpolated value for each x.
    
    This is a simplified example:
      - We clamp x to [0,1].
      - We do not do special end-conditions (natural or clamped). 
      - For each sub-interval, we use 4 control points: p_(i-1), p_i, p_(i+1), p_(i+2).
    """
    x_clamped = torch.clamp(x, 0.0, 1.0)
    # Find which sub-interval each x belongs to
    # Because knot_x is equally spaced, we can do an integer index:
    num_knots = knot_x.shape[0]  # = M
    dx = 1.0 / (num_knots - 1)
    
    # sub-interval index i in [0..M-2]
    i = torch.floor(x_clamped / dx).long()
    i = torch.clamp(i, 0, num_knots - 2)  # ensure in [0, M-2]
    
    # Local parameter t in [0,1] within the sub-interval
    t = (x_clamped - i * dx) / dx  # shape = same as x
    
    # For Catmull-Rom, we need p_{i-1}, p_i, p_{i+1}, p_{i+2}
    # We'll clamp indices for boundary cases
    i0 = torch.clamp(i - 1, 0, num_knots - 1)
    i1 = i
    i2 = i + 1
    i3 = torch.clamp(i + 2, 0, num_knots - 1)
    
    p0 = knot_y[i0]
    p1 = knot_y[i1]
    p2 = knot_y[i2]
    p3 = knot_y[i3]
    
    # Catmull-Rom basis (one common way):
    #  p(t) = 0.5 * [ (2 * p1) 
    #                 + (-p0 + p2) * t
    #                 + (2p0 - 5p1 + 4p2 - p3) * t^2
    #                 + (-p0 + 3p1 - 3p2 + p3) * t^3 ]
    t2 = t * t
    t3 = t2 * t
    
    a = 2 * p1
    b = -p0 + p2
    c = 2*p0 - 5*p1 + 4*p2 - p3
    d = -p0 + 3*p1 - 3*p2 + p3
    
    spline_val = 0.5 * (a + b*t + c*t2 + d*t3)
    return spline_val


class CubicSplineRadialFourierConv(nn.Module):
    """
    Flow layer that applies a radially symmetric kernel in Fourier space:
      X_out(k) = X(k) * spline(|k|), if forward
      X_out(k) = X(k) / spline(|k|), if reverse
    """
    def __init__(self, image_shape, num_knots=16):
        super().__init__()
        
        # Suppose image_shape = (C, H, W)
        _, self.H, self.W = image_shape
        
        # Precompute the magnitude of frequencies for each (kx, ky).
        # We'll define them using fftfreq:
        ky = torch.fft.fftfreq(self.H)  # shape [H]
        kx = torch.fft.fftfreq(self.W)  # shape [W]
        KX, KY = torch.meshgrid(kx, ky, indexing="xy")  # shape [W, H]
        
        # radial magnitude r = sqrt(kx^2 + ky^2)
        # We'll store as a buffer (not a parameter)
        r = torch.sqrt(KX**2 + KY**2)  # shape [W,H]
        self.register_buffer("r", r)
        
        # For the spline, we want to evaluate in [0,1].
        # So we'll also store r_max to normalize r by r_max.
        r_max = r.max()
        self.register_buffer("r_max", r_max)
        
        # Define a fixed set of knot positions in [0,1].
        # (M = num_knots). We typically want them equally spaced.
        self.num_knots = num_knots
        knot_x = torch.linspace(0, 1, steps=num_knots)
        # Usually we don't need them to be trainable, so register as buffer:
        self.register_buffer("knot_x", knot_x)
        
        # The knot values (y) are the actual trainable parameters:
        # Initialize them to 1.0 + small noise, for example.
        self.knot_y = nn.Parameter(
            torch.ones(num_knots, dtype=torch.float32) 
            + 0.1 * torch.randn(num_knots)
        )

    def forward(self, x, logdet=None, reverse=False):
        """
        x: (batch, channels, H, W) real tensor
        logdet: either None or cumulative log determinant so far
        reverse: whether we do forward multiplication or inverse (division)
        """
        device = x.device
        if logdet is None:
            logdet = torch.zeros(x.size(0), device=device)
        
        # 1) FFT
        X = fft.fftn(x, dim=(-2, -1))  # shape (batch, channels, H, W), complex
        
        # 2) Build the radial kernel in freq space via our spline
        #    r_norm in [0,1] = r / r_max
        r_norm = (self.r / (self.r_max + 1e-9)).to(device)
        
        # Evaluate cubic spline at each r_norm
        # We'll do it elementwise. r_norm has shape [W,H].
        # We can flatten, evaluate, reshape.
        r_norm_flat = r_norm.reshape(-1)
        kernel_flat = catmull_rom_spline_1d(r_norm_flat, self.knot_x, self.knot_y)
        kernel_2d = kernel_flat.view_as(r_norm)  # shape [W,H]
        
        # Expand to broadcast over (batch, channels, W, H)
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # (1,1,W,H)
        
        # 3) Multiply or divide in the freq domain
        if not reverse:
            # Forward pass: multiply in freq domain
            X_out = X * kernel_2d
            # logdet = sum log|kernel| over all spatial freq for each sample
            kernel_broad = kernel_2d.expand_as(X_out)
            local_logdet = torch.log(torch.abs(kernel_broad)).sum(dim=list(range(1, X_out.ndim)))  
            logdet = logdet + local_logdet
        else:
            # Inverse pass: X_out = X / kernel
            eps = 1e-9
            X_out = X / (kernel_2d + eps)
            # Now the logdet is negative
            kernel_broad = (kernel_2d + eps).expand_as(X_out)
            local_logdet = -torch.log(torch.abs(kernel_broad)).sum(dim=list(range(1, X_out.ndim)))
            logdet = logdet + local_logdet
        
        # 4) IFFT
        x_out = fft.ifftn(X_out, dim=(-2, -1)).real
        return x_out, logdet

class RadialFourierConv(nn.Module):
    """
    Flow layer that applies a radially symmetric kernel in Fourier space.
    F(x) = FFT(x), then multiply by kernel(|k|), then IFFT.
    If reverse=True, we multiply by 1 / kernel(|k|) instead.
    """
    def __init__(self, image_shape):
        super().__init__()
        
        # image_shape = (batch, channels, height, width) or something similar
        # We'll assume it's something like (C, H, W) for single-batch usage,
        # or just store H, W for this example.
        _, self.H, self.W = image_shape
        
        # Precompute the magnitude of frequencies for each (kx, ky).
        # kx, ky each range from 0..(H-1) or W-1 in freq space after an FFT.
        # We'll define them as 'meshgrid' in the *shifted* sense for clarity.
        ky = torch.fft.fftfreq(self.H)  # shape [H]
        kx = torch.fft.fftfreq(self.W)  # shape [W]
        KX, KY = torch.meshgrid(kx, ky, indexing="xy")
        
        # radial magnitude
        self.register_buffer("r", torch.sqrt(KX**2 + KY**2))  # shape [W, H]
        # Notice the shape may be transposed depending on your indexing.

        # We'll define a trainable radial function.  For simplicity, store as
        # a direct table of shape [num_radial_bins]. For a more flexible approach,
        # you could store a small MLP and pass self.r.view(-1) to it.
        # Here, let's define some binning approach:
        self.num_radial_bins = 16
        # Suppose we store one trainable parameter per radial bin:
        self.radial_params = nn.Parameter(torch.ones(self.num_radial_bins, dtype=torch.float32) + 0.1 * torch.randn(self.num_radial_bins))
        
        # You'd also define a function to map self.r -> some bin index, e.g.:
        #   bin_idx = (r / r.max()) * (self.num_radial_bins-1)
        # and then interpolate or nearest-neighbor to get a kernel amplitude.

    def forward(self, x, logdet=None, reverse=False):
        """
        x: (batch, channels, H, W) real tensor
        logdet: either None or cumulative log determinant so far
        reverse: whether we do forward multiplication or inverse (division)
        """
        if logdet is None:
            logdet = torch.zeros(x.size(0), device=x.device)
        # 1) FFT
        X = fft.fftn(x, dim=(-2, -1))  # shape (batch, channels, H, W), complex
        # 2) Build the kernel in frequency space
        #    We'll do a simple radial bin approach. Real example might do interpolation.
        #    We'll first compute bin indices:
        r_norm = self.r / (self.r.max() + 1e-9) * (self.num_radial_bins - 1)
        r_bin = r_norm.long().clamp(min=0, max=self.num_radial_bins - 1)
        
        # Create kernel by indexing into self.radial_params
        kernel_2d = self.radial_params[r_bin]  # shape [W, H]
        # Expand for channels if needed, or broadcast:
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, W, H)
        
        # 3) Multiply or divide in the freq domain
        if not reverse:
            # Forward pass: multiply in freq domain
            X_out = X * kernel_2d  # broadcasting
            # logdet per batch sample
            # broadcast kernel to shape (B,C,H,W)
            kernel_broad = kernel_2d.expand_as(X_out)
            local_logdet = torch.log(torch.abs(kernel_broad)).sum(dim=list(range(1, X_out.ndim)))  # shape (B,)
            logdet = logdet + local_logdet
        else:
            # Inverse pass: X_out = X / kernel
            eps = 1e-9
            X_out = X / (kernel_2d + eps)
            # Now the logdet is negative
            kernel_broad = (kernel_2d + eps).expand_as(X_out)
            local_logdet = -torch.log(torch.abs(kernel_broad)).sum(dim=list(range(1, X_out.ndim)))
            logdet = logdet + local_logdet
        
        # 4) Inverse FFT
        x_out = fft.ifftn(X_out, dim=(-2, -1)).real  # keep real part if original data is real
        return x_out, logdet


# class PixelwiseNonlinearity(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Instead of a shift, we use a fixed offset of 2.0
#         self.shift = 2.0
#         # Scale parameter that can be learned
#         self.scale = nn.Parameter(torch.ones(1))

#     def forward(self, x, logdet=None, reverse=False):
#         if logdet is None:
#             logdet = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
#         if not reverse:
#             # Forward: out = sigmoid(x + shift) * scale
#             out = torch.sigmoid(x + self.shift) * self.scale
#             # dout/dx = (sigmoid(x + shift) * (1 - sigmoid(x + shift))) * scale
#             doutdx = out * (1 - out / self.scale) * self.scale
#             # log|dout/dx| = log((sigmoid(x + shift) * (1 - sigmoid(x + shift))) * scale)
#             local_logdet = torch.log(torch.abs(doutdx) + 1e-9)
#             local_logdet = local_logdet.sum(dim=list(range(1, out.ndim)))  # shape (B,)
#         else:
#             # Inverse transformation is not straightforward and generally requires numerical approximation
#             # For demonstration, let's assume x is scaled back before applying the logit
#             scaled_x = x / self.scale
#             out = torch.logit(scaled_x) - self.shift
#             print('poop',out)
#             # dout/dx = 1 / (scaled_x * (1 - scaled_x)) / scale
#             doutdx = 1 / (scaled_x * (1 - scaled_x)) / self.scale
#             # log|dout/dx| = -log(scaled_x * (1 - scaled_x)) - log(scale)
#             local_logdet = -torch.log(scaled_x * (1 - scaled_x)) - torch.log(torch.abs(self.scale))
#             local_logdet = local_logdet.sum(dim=list(range(1, x.ndim)))  # shape (B,)
#         logdet = logdet + local_logdet
#         return out, logdet


# class PixelwiseNonlinearity(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Instead of a shift, we use a fixed offset of 2.0
#         self.shift = 0.0
#         # Scale parameter (learnable)
#         self.scale = nn.Parameter(torch.ones(1))

#     def forward(self, x, logdet=None, reverse=False):
#         """
#         If not `reverse`, we apply the forward transform:
#             y = scale * exp(x + shift)
#         If `reverse`, we apply the inverse transform:
#             x = log(y / scale) - shift
#         """
#         if logdet is None:
#             logdet = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)

#         if not reverse:
#             # Forward: out = scale * exp(x + shift)
#             out = torch.clamp(self.scale * torch.exp(x + self.shift), max=100.)
#             # dy/dx = out
#             # log|dy/dx| = log(out)
#             local_logdet = torch.log(out + 1e-9)

#             # Sum logdet over all but batch dimension
#             local_logdet = local_logdet.sum(dim=list(range(1, out.ndim)))

#         else:
#             # Reverse (inverse) transform:
#             # x = log(y / scale) - shift
#             # Here, 'x' in this function’s argument is actually the 'y' from forward pass
#             # So let's rename for clarity:
#             y = x

#             # Safeguard: clamp y to avoid log of zero or negative
#             y = torch.clamp(y, min=1e-9)

#             out = torch.log(y / self.scale) - self.shift

#             # dx/dy = 1 / y
#             # log|dx/dy| = -log(y)
#             local_logdet = -torch.log(y)

#             local_logdet = local_logdet.sum(dim=list(range(1, x.ndim)))

#         # Accumulate into total logdet
#         logdet = logdet + local_logdet
#         return out, logdet

class RQspline(nn.Module):
    '''
    Ratianal quadratic spline.
    See appendix B of https://arxiv.org/pdf/2007.00674.pdf
    The main advantage compared to cubic spline is that the
    inverse is analytical and does not require binary search

    x: (ndim, nknot) 2d array, each row should be monotonic increasing
    y: (ndim, nknot) 2d array, each row should be monotonic increasing
    deriv: (ndim, nknot) 2d array, should be positive
    '''

    def __init__(self, ndim, nknot):

        super().__init__()
        self.ndim = ndim
        self.nknot = nknot

        x0 = torch.rand(ndim, 1)-4.5
        logdx = torch.log(torch.abs(-2*x0 / (nknot-1)))

        #use log as parameters to make sure monotonicity
        self.x0 = nn.Parameter(x0)
        self.y0 = nn.Parameter(x0.clone())
        self.logdx = nn.Parameter(torch.ones(ndim, nknot-1)*logdx)
        self.logdy = nn.Parameter(torch.ones(ndim, nknot-1)*logdx)
        self.logderiv = nn.Parameter(torch.zeros(ndim, nknot))


    def set_param(self, x, y, deriv):

        dx = x[:,1:] - x[:,:-1]
        dy = y[:,1:] - y[:,:-1]
        assert (dx > 0).all()
        assert (dy > 0).all()
        assert (deriv > 0).all()

        self.x0[:] = x[:, 0].view(-1,1)
        self.y0[:] = y[:, 0].view(-1,1)
        self.logdx[:] = torch.log(dx)
        self.logdy[:] = torch.log(dy)
        self.logderiv[:] = torch.log(deriv)


    def _prepare(self):
        #return knot points and derivatives
        xx = torch.cumsum(torch.exp(self.logdx), dim=1)
        xx += self.x0
        xx = torch.cat((self.x0, xx), dim=1)
        yy = torch.cumsum(torch.exp(self.logdy), dim=1)
        yy += self.y0
        yy = torch.cat((self.y0, yy), dim=1)
        delta = torch.exp(self.logderiv)
        return xx, yy, delta

    def forward(self, x):
        # x: (ndata, ndim) 2d array
        xx, yy, delta = self._prepare() #(ndim, nknot)

        index = torch.searchsorted(xx.detach(), x.T.contiguous().detach()).T
        y = torch.zeros_like(x)
        logderiv = torch.zeros_like(x)

        #linear extrapolation
        select0 = index == 0
        dim = torch.repeat_interleave(torch.arange(self.ndim).view(1,-1), len(x), dim=0)[select0]
        y[select0] = yy[dim, 0] + (x[select0]-xx[dim, 0]) * delta[dim, 0]
        logderiv[select0] = self.logderiv[dim, 0]
        selectn = index == self.nknot
        dim = torch.repeat_interleave(torch.arange(self.ndim).view(1,-1), len(x), dim=0)[selectn]
        y[selectn] = yy[dim, -1] + (x[selectn]-xx[dim, -1]) * delta[dim, -1]
        logderiv[selectn] = self.logderiv[dim, -1]

        #rational quadratic spline
        select = ~(select0 | selectn)
        index = index[select]
        dim = torch.repeat_interleave(torch.arange(self.ndim).view(1,-1), len(x), dim=0)[select]
        xi = (x[select] - xx[dim, index-1]) / (xx[dim, index] - xx[dim, index-1])
        s = (yy[dim, index]-yy[dim, index-1]) / (xx[dim, index]-xx[dim, index-1])
        xi1_xi = xi*(1-xi)
        denominator = s + (delta[dim, index]+delta[dim, index-1]-2*s)*xi1_xi
        xi2 = xi**2

        y[select] = yy[dim, index-1] + ((yy[dim, index]-yy[dim, index-1]) * (s*xi2+delta[dim, index-1]*xi1_xi)) / denominator
        logderiv[select] = 2*torch.log(s) + torch.log(delta[dim, index]*xi2 + 2*s*xi1_xi + delta[dim, index-1]*(1-xi)**2) - 2 * torch.log(denominator)

        return y, logderiv

    def inverse(self, y):
        xx, yy, delta = self._prepare()

        index = torch.searchsorted(yy.detach(), y.T.contiguous().detach()).T
        x = torch.zeros_like(y)
        logderiv = torch.zeros_like(y)

        #linear extrapolation
        select0 = index == 0
        dim = torch.repeat_interleave(torch.arange(self.ndim).view(1,-1), len(x), dim=0)[select0]
        x[select0] = xx[dim, 0] + (y[select0]-yy[dim, 0]) / delta[dim, 0]
        logderiv[select0] = self.logderiv[dim, 0]
        selectn = index == self.nknot
        dim = torch.repeat_interleave(torch.arange(self.ndim).view(1,-1), len(x), dim=0)[selectn]
        x[selectn] = xx[dim, -1] + (y[selectn]-yy[dim, -1]) / delta[dim, -1]
        logderiv[selectn] = self.logderiv[dim, -1]

        #rational quadratic spline
        select = ~(select0 | selectn)
        index = index[select]
        dim = torch.repeat_interleave(torch.arange(self.ndim).view(1,-1), len(x), dim=0)[select]
        deltayy = yy[dim, index]-yy[dim, index-1]
        s = deltayy / (xx[dim, index]-xx[dim, index-1])
        delta_2s = delta[dim, index]+delta[dim, index-1]-2*s
        deltay_delta_2s = (y[select]-yy[dim, index-1]) * delta_2s

        a = deltayy * (s-delta[dim, index-1]) + deltay_delta_2s
        b = deltayy * delta[dim, index-1] - deltay_delta_2s
        c = - s * (y[select]-yy[dim, index-1])
        discriminant = b.pow(2) - 4 * a * c
        #discriminant[discriminant<0] = 0 
        assert (discriminant >= 0).all()
        xi = - 2*c / (b + torch.sqrt(discriminant))
        xi1_xi = xi * (1-xi)

        x[select] = xi * (xx[dim, index] - xx[dim, index-1]) + xx[dim, index-1]
        logderiv[select] = 2*torch.log(s) + torch.log(delta[dim, index]*xi**2 + 2*s*xi1_xi + delta[dim, index-1]*(1-xi)**2) - 2 * torch.log(s + delta_2s*xi1_xi)

        return x, logderiv


class PixelwiseNonlinearity(nn.Module):
    """
    A pixelwise transform that applies the same 1D monotonic rational-quadratic spline
    to each pixel value. This is a 'normalizing flow' style transformation:
        - forward(...)  :  x -> y = spline(x)
        - forward(..., reverse=True):  y -> x = spline^{-1}(y)
    
    The RQspline is from your snippet. We'll wrap it so that each pixel is treated as
    dimension=1 in that spline.
    """
    def __init__(self, nknot=8):
        """
        nknot: number of knots in the rational-quadratic spline.
        """
        super().__init__()
        
        self.ndim = 1  
        self.nknot = nknot
        
        # Our RQspline class expects (ndim, nknot). We do (1, nknot).
        self.rqspline = RQspline(ndim=self.ndim, nknot=self.nknot)
        
        # Optionally, you might want to initialize or set param,
        # e.g. self.rqspline.set_param(...) if you have a desired init.

    def forward(self, x, logdet=None, reverse=False):
        """
        x: Tensor of shape (B, C, H, W) or any shape [B, *spatial_dims].
        logdet: Tensor of shape (B,) or None
        reverse: bool indicating forward/inverse pass
        
        Returns:
            out, logdet  (both shaped accordingly)
        """
        # 1) If logdet is None, create a zeros-like placeholder.
        if logdet is None:
            logdet = torch.zeros(
                x.size(0), 
                device=x.device, 
                dtype=x.dtype
            )

        # 2) Flatten x so RQspline sees shape (N, 1),
        #    where N = B*C*H*W if we treat each scalar as its own dimension.
        B = x.size(0)
        x_flat = x.view(B, -1)  # shape (B, C*H*W)
        # Now we have x_flat of shape (B, N_pixels). We want shape (B*N_pixels, 1).
        x_flat = x_flat.transpose(0, 1).contiguous()    # shape (N_pixels, B)
        x_flat = x_flat.view(-1, 1)                     # shape (N_pixels*B, 1)
        
        # 3) Forward or Inverse through RQspline
        if not reverse:
            # forward transform: y, log|dy/dx|
            y_flat, logabsdet_flat = self.rqspline.forward(x_flat)
            # y_flat: (B*N_pixels, 1)
            # logabsdet_flat: (B*N_pixels, 1) or (B*N_pixels,) depending on your RQspline code
        else:
            # inverse transform: x = rqspline^{-1}(y)
            y_flat, logabsdet_flat = self.rqspline.inverse(x_flat)
            # y_flat: same shape as x_flat
            # logabsdet_flat: same shape as well

        # 4) Reshape y_flat back to (B, C, H, W) 
        #    We do the inverse of the flatten steps.
        y_flat = y_flat.view(-1, B)  # shape (N_pixels, B)
        y_flat = y_flat.transpose(0, 1).contiguous()  # shape (B, N_pixels)
        
        # If original x was (B, C, H, W), let's restore that shape:
        # We'll get the "spatial" part from x.shape[1:].
        spatial_shape = x.shape[1:]
        out = y_flat.view(B, *spatial_shape)  # shape (B, C, H, W)

        # 5) Accumulate log|det|. 
        #    The RQspline returns 'logderiv' per sample in shape (B*N_pixels,).
        #    We need to sum over the pixels, but *separately* for each batch item.
        #
        #    So we reshape logabsdet_flat similarly:
        logabsdet_flat = logabsdet_flat.view(-1)  # ensure it's shape (B*N_pixels,)
        
        # We'll chunk it into B groups, each of size (N_pixels).
        N_pixels = logabsdet_flat.numel() // B
        logabsdet_flat = logabsdet_flat.view(B, N_pixels)
        
        # Sum across the pixel dimension. Now shape is (B,).
        local_logdet = logabsdet_flat.sum(dim=1)
        
        # 6) Add to the running logdet
        logdet = logdet + local_logdet
        
        return out, logdet
    
class _ActNorm(nn.Module):
    """
    Activation Normalization
    Initialize the bias and scale with a given minibatch,
    so that the output per-channel have zero mean and unit variance for that.
    After initialization, `bias` and `logs` will be trained as parameters.
    """

    def __init__(self, num_features, scale=1.0):
        super().__init__()
        # register mean and scale
        size = [1, num_features, 1, 1]
        self.bias = nn.Parameter(torch.zeros(*size))
        self.logs = nn.Parameter(torch.zeros(*size))
        self.num_features = num_features
        self.scale = scale
        self.inited = False

    def initialize_parameters(self, input):
        if not self.training:
            raise ValueError("In Eval mode, but ActNorm not inited")

        with torch.no_grad():
            bias = -torch.mean(input.clone(), dim=[0, 2, 3], keepdim=True)
            vars = torch.mean((input.clone() + bias) ** 2, dim=[0, 2, 3], keepdim=True)
            logs = torch.log(self.scale / (torch.sqrt(vars) + 1e-6))

            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)

            self.inited = True

    def _center(self, input, reverse=False):
        if reverse:
            return input - self.bias
        else:
            return input + self.bias
        

    def _scale(self, input, logdet=None, reverse=False):
        if reverse:
            input = input * torch.exp(-self.logs)
        else:
            input = input * torch.exp(self.logs)

        if logdet is not None:
            """
            logs is log_std of `mean of channels`
            so we need to multiply by number of pixels
            """
            b, c, h, w = input.shape

            dlogdet = torch.sum(self.logs) * h * w

            if reverse:
                dlogdet *= -1

            logdet = logdet + dlogdet

        return input, logdet

    def forward(self, input, logdet=None, reverse=False):
        self._check_input_dim(input)
        if not self.inited:
            self.initialize_parameters(input)

        if reverse:
            input, logdet = self._scale(input, logdet, reverse)
            input = self._center(input, reverse)
        else:
            input = self._center(input, reverse)
            input, logdet = self._scale(input, logdet, reverse)

        return input, logdet
    
class ActNorm2d(_ActNorm):
    def __init__(self, num_features, scale=1.0):
        super().__init__(num_features, scale)

    def _check_input_dim(self, input):
        assert len(input.size()) == 4
        assert input.size(1) == self.num_features, ("[ActNorm]: input should be in shape as `BCHW`,"
                                                    " channels should be {} rather than {}".format(self.num_features,
            input.size()))
