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
        #self.knot_y = torch.ones(num_knots, dtype=torch.float32)

        self.num_params = num_knots

    # def set_params(self, params):
    #     """
    #     Replace the trainable knot vector with new_params.
    #     new_params should be a 1D tensor of length self.num_knots.
    #     """
    #     self.knot_y = params.reshape(self.num_knots).detach().clone()

    def get_params(self, params):
        knot_y = params[:self.num_params]
        return knot_y

    def forward(self, x, params, logdet=None, reverse=False):
        """
        x: (batch, channels, H, W) real tensor
        logdet: either None or cumulative log determinant so far
        reverse: whether we do forward multiplication or inverse (division)
        """
        device = x.device
        if logdet is None:
            logdet = torch.zeros(x.size(0), device=device)
        
        knot_y = self.get_params(params)

        # 1) FFT
        X = fft.fftn(x, dim=(-2, -1))  # shape (batch, channels, H, W), complex
        
        # 2) Build the radial kernel in freq space via our spline
        #    r_norm in [0,1] = r / r_max
        r_norm = (self.r / (self.r_max + 1e-9))#.to(device)
        
        # Evaluate cubic spline at each r_norm
        # We'll do it elementwise. r_norm has shape [W,H].
        # We can flatten, evaluate, reshape.
        r_norm_flat = r_norm.reshape(-1)
        kernel_flat = catmull_rom_spline_1d(r_norm_flat, self.knot_x, knot_y)
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


class PLspline(nn.Module):
    """
    Piecewise Linear (PL) Spline
    - Monotonic in each dimension
    - Has analytical inverse (no numerical search)
    - x: (ndim, nknot) 2D array of strictly increasing knot positions
    - y: (ndim, nknot) 2D array of strictly increasing knot values
    """

    def __init__(self, ndim, nknot):
        super().__init__()
        self.ndim = ndim
        self.nknot = nknot

        # Create an arange buffer for indexing (avoid repeated arange calls):
        self.register_buffer("dim_arange", torch.arange(self.ndim))

        # Simple random init (you can adjust):
        # We'll pick a random starting x0 (centered near 0),
        # and let each logdx be near log(1.0).
        x0_init = torch.randn(ndim, 1) * 0.1
        y0_init = torch.randn(ndim, 1) * 0.1

        # Initialize all increments to ~exp(0)=1
        logdx_init = torch.zeros(ndim, nknot - 1)
        logdy_init = torch.zeros(ndim, nknot - 1)

        self.x0 = nn.Parameter(x0_init)                   # shape (ndim, 1)
        self.y0 = nn.Parameter(y0_init)                   # shape (ndim, 1)
        self.logdx = nn.Parameter(logdx_init)             # shape (ndim, nknot - 1)
        self.logdy = nn.Parameter(logdy_init)             # shape (ndim, nknot - 1)

    def set_param(self, x, y):
        """
        Optionally, you can manually set the spline's knot locations (x, y).
        x, y: shape (ndim, nknot), strictly increasing in each row.
        """
        dx = x[:, 1:] - x[:, :-1]  # shape (ndim, nknot-1)
        dy = y[:, 1:] - y[:, :-1]
        # (Optional checks):
        # assert (dx > 0).all()
        # assert (dy > 0).all()

        self.x0.data = x[:, 0].view(-1, 1)
        self.y0.data = y[:, 0].view(-1, 1)
        self.logdx.data = torch.log(dx)
        self.logdy.data = torch.log(dy)

    def _prepare(self):
        """
        Compute the actual knot positions xx and yy from the log parameters.
        This is called each forward pass because parameters can change.
        """
        # Recover knot positions in x
        xx_increments = torch.exp(self.logdx)             # shape (ndim, nknot-1)
        xx_cumsum = torch.cumsum(xx_increments, dim=1)    # shape (ndim, nknot-1)
        xx = torch.cat([self.x0, self.x0 + xx_cumsum], dim=1)  # shape (ndim, nknot)

        # Recover knot positions in y
        yy_increments = torch.exp(self.logdy)             # shape (ndim, nknot-1)
        yy_cumsum = torch.cumsum(yy_increments, dim=1)    # shape (ndim, nknot-1)
        yy = torch.cat([self.y0, self.y0 + yy_cumsum], dim=1)  # shape (ndim, nknot)

        return xx, yy

    def forward(self, x):
        """
        Forward transform: x -> y
        x: shape (ndata, ndim)

        Returns:
         y: shape (ndata, ndim)
         logdet: shape (ndata, ndim), the log|dy/dx|
        """
        xx, yy = self._prepare()  # shape (ndim, nknot)

        # 1) Find which interval each x falls into for each dimension
        #    We do searchsorted along each dimension's xx.
        #    x is shape (ndata, ndim), xx is (ndim, nknot).
        #    Torch requires searchsorted of shape (K,) so we do it dimension by dimension.
        #    For vectorization: x.T is shape (ndim, ndata); so we search in xx[i] for x.T[i].
        index_list = []
        for i in range(self.ndim):
            # shape of x.T[i] is (ndata,)
            # shape of xx[i] is (nknot,)
            idx_i = torch.searchsorted(xx[i], x[:, i])  # shape (ndata,)
            index_list.append(idx_i)
        index = torch.stack(index_list, dim=1)  # shape (ndata, ndim)

        # Prepare outputs
        y = torch.zeros_like(x)
        logdet = torch.zeros_like(x)

        # (ndata, ndim) array for picking dimension
        dim_all = self.dim_arange.unsqueeze(0).expand(len(x), self.ndim)

        # ~~~~~ Extrapolation: index == 0 (left of left-most knot) ~~~~~
        select0 = (index == 0)
        if select0.any():
            dim0 = dim_all[select0]    # shape (#selected,)
            # For each selected, slope = (yy[...,1] - yy[...,0]) / (xx[...,1] - xx[...,0])
            # but we pick dimension i for each selected entry
            slope0 = (yy[dim0, 1] - yy[dim0, 0]) / (xx[dim0, 1] - xx[dim0, 0])

            # y = y0 + slope*(x - x0)
            # x0 = xx[dim0,0], y0 = yy[dim0,0]
            xvals_0 = x[select0]
            x0s = xx[dim0, 0]
            y0s = yy[dim0, 0]

            y[select0] = y0s + slope0 * (xvals_0 - x0s)
            logdet[select0] = slope0.log()  # log|dy/dx|

        # ~~~~~ Extrapolation: index == nknot (right of right-most knot) ~~~~~
        select_n = (index == self.nknot)
        if select_n.any():
            dimn = dim_all[select_n]
            slopeN = (yy[dimn, -1] - yy[dimn, -2]) / (xx[dimn, -1] - xx[dimn, -2])

            xvals_n = x[select_n]
            xN_1 = xx[dimn, -1]
            yN_1 = yy[dimn, -1]

            y[select_n] = yN_1 + slopeN * (xvals_n - xN_1)
            logdet[select_n] = slopeN.log()

        # ~~~~~ Piecewise-Linear in between: 0 < index < nknot ~~~~~
        select_mid = ~(select0 | select_n)
        if select_mid.any():
            idx_mid = index[select_mid]  # shape (#selected,)
            dim_mid = dim_all[select_mid]

            xvals_mid = x[select_mid]

            # x_lo = xx[dim_mid, idx_mid-1], x_hi = xx[dim_mid, idx_mid]
            x_lo = xx[dim_mid, idx_mid - 1]
            x_hi = xx[dim_mid, idx_mid]

            # y_lo = yy[dim_mid, idx_mid-1], y_hi = yy[dim_mid, idx_mid]
            y_lo = yy[dim_mid, idx_mid - 1]
            y_hi = yy[dim_mid, idx_mid]

            # slope
            slope_mid = (y_hi - y_lo) / (x_hi - x_lo)

            # forward transform
            y[select_mid] = y_lo + slope_mid * (xvals_mid - x_lo)

            # logdet = log(slope_mid)
            logdet[select_mid] = slope_mid.log()

        return y, logdet

    def inverse(self, y):
        """
        Inverse transform: y -> x
        y: shape (ndata, ndim)

        Returns:
         x: shape (ndata, ndim)
         logdet: shape (ndata, ndim), the log|dx/dy|
        """
        xx, yy = self._prepare()

        # 1) Find which interval each y falls into
        #    We'll do the same dimension-wise approach
        index_list = []
        for i in range(self.ndim):
            idx_i = torch.searchsorted(yy[i], y[:, i])
            index_list.append(idx_i)
        index = torch.stack(index_list, dim=1)

        x = torch.zeros_like(y)
        logdet = torch.zeros_like(y)

        dim_all = self.dim_arange.unsqueeze(0).expand(len(y), self.ndim)

        # ~~~~~ Extrapolation: index == 0 ~~~~~
        select0 = (index == 0)
        if select0.any():
            dim0 = dim_all[select0]
            slope0 = (yy[dim0, 1] - yy[dim0, 0]) / (xx[dim0, 1] - xx[dim0, 0])

            yvals_0 = y[select0]
            x0s = xx[dim0, 0]
            y0s = yy[dim0, 0]

            # x = x0 + (y - y0)/slope
            x[select0] = x0s + (yvals_0 - y0s) / slope0
            logdet[select0] = (-slope0.log())  # log|dx/dy| = log(1/slope) = -log(slope)

        # ~~~~~ Extrapolation: index == nknot ~~~~~
        select_n = (index == self.nknot)
        if select_n.any():
            dimn = dim_all[select_n]
            slopeN = (yy[dimn, -1] - yy[dimn, -2]) / (xx[dimn, -1] - xx[dimn, -2])

            yvals_n = y[select_n]
            xN_1 = xx[dimn, -1]
            yN_1 = yy[dimn, -1]

            x[select_n] = xN_1 + (yvals_n - yN_1) / slopeN
            logdet[select_n] = (-slopeN.log())

        # ~~~~~ Piecewise-Linear in between: 0 < index < nknot ~~~~~
        select_mid = ~(select0 | select_n)
        if select_mid.any():
            idx_mid = index[select_mid]
            dim_mid = dim_all[select_mid]

            yvals_mid = y[select_mid]

            y_lo = yy[dim_mid, idx_mid - 1]
            y_hi = yy[dim_mid, idx_mid]
            x_lo = xx[dim_mid, idx_mid - 1]
            x_hi = xx[dim_mid, idx_mid]

            slope_mid = (y_hi - y_lo) / (x_hi - x_lo)

            x[select_mid] = x_lo + (yvals_mid - y_lo) / slope_mid

            # log|dx/dy| = -log(slope_mid)
            logdet[select_mid] = -slope_mid.log()

        return x, logdet

    
    
class RQspline(nn.Module):
    """
    Rational Quadratic Spline
    See appendix B of https://arxiv.org/pdf/2007.00674.pdf
    - Monotonic in each dimension
    - Has analytical inverse (no binary search)
    - x: (ndim, nknot) 2D array (monotonic in each row)
    - y: (ndim, nknot) 2D array (monotonic in each row)
    - deriv: (ndim, nknot) 2D array (positive)
    """

    def __init__(self, ndim, nknot):
        super().__init__()
        self.ndim = ndim
        self.nknot = nknot

        # Create an arange for indexing (we'll expand it later as needed).
        # This avoids repeatedly calling torch.arange(...) inside forward.
        self.register_buffer("dim_arange", torch.arange(self.ndim))

        # Some initial random initialization (same as your code):
        x0 = torch.rand(ndim, 1) - 4.5
        logdx = torch.log(torch.abs(-2 * x0 / (nknot - 1)))

        # Use log-parameters to ensure monotonicity
        # self.x0 = nn.Parameter(x0)
        # self.y0 = nn.Parameter(x0.clone())
        # self.logdx = nn.Parameter(torch.ones(ndim, nknot - 1) * logdx)
        # self.logdy = nn.Parameter(torch.ones(ndim, nknot - 1) * logdx)
        # self.logderiv = nn.Parameter(torch.zeros(ndim, nknot))

        # self.x0 = x0
        # self.y0 = x0.clone()
        # self.logdx = torch.ones(ndim, nknot - 1) * logdx
        # self.logdy = torch.ones(ndim, nknot - 1) * logdx
        # self.logderiv = torch.zeros(ndim, nknot)


    def get_params(self, params):
        """
        Takes a 1D vector of parameters and splits it into x0, y0, logdx, logdy, logderiv.
        """
        offset = 0
        n = self.ndim
        
        # x0
        x0 = params[offset : offset + n].view(n, 1)
        offset += n

        # y0
        y0 = params[offset : offset + n].view(n, 1)
        offset += n

        # logdx
        size_logdx = n * (self.nknot - 1)
        logdx = params[offset : offset + size_logdx].view(n, self.nknot - 1)
        offset += size_logdx

        # logdy
        size_logdy = n * (self.nknot - 1)
        logdy = params[offset : offset + size_logdy].view(n, self.nknot - 1)
        offset += size_logdy

        # logderiv
        size_logder = n * self.nknot
        logderiv = params[offset : offset + size_logder].view(n, self.nknot)
        offset += size_logder

        return x0, y0, logdx, logdy, logderiv

    def _prepare(self, x0, y0, logdx, logdy, logderiv):
        """
        Computes the actual knot positions (xx, yy) and derivatives (delta)
        from the log-params. This is typically done each forward pass
        because parameters may be updated by backprop.
        """
        # x0 + cumsum(exp(logdx)) => monotonic
        xx = torch.cumsum(torch.exp(logdx), dim=1)
        xx = xx + x0
        xx = torch.cat((x0, xx), dim=1)  # shape (ndim, nknot)

        yy = torch.cumsum(torch.exp(logdy), dim=1)
        yy = yy + y0
        yy = torch.cat((y0, yy), dim=1)  # shape (ndim, nknot)

        delta = torch.exp(logderiv)      # shape (ndim, nknot)
        return xx, yy, delta

    def forward(self, x, params):
        """
        Forward transform: x -> y
        x: shape (ndata, ndim)
        Returns:
          y: shape (ndata, ndim)
          logderiv: shape (ndata, ndim), the log|dy/dx|
        """

        x0, y0, logdx, logdy, logderiv = self.get_params(params)
        xx, yy, delta = self._prepare(x0, y0, logdx, logdy, logderiv)  # shape (ndim, nknot) each

        # 1) We need to find which interval each x falls into
        #    Use searchsorted on each dimension. We do transpose so that
        #    'xx' is shape (ndim, nknot) and 'x' is shape (ndata, ndim).
        #    Then transpose result back.
        #    *Removed the .detach() calls so we keep everything on GPU
        index = torch.searchsorted(xx, x.T.contiguous()).T  # shape (ndata, ndim)
        #index = broadcast_searchsorted_multi(x.to(x.device), xx.to(x.device))
        
        # 2) Prepare output buffers
        y = torch.zeros_like(x)
        new_logderiv = torch.zeros_like(x)

        # 3) We'll create one "dim_all" array (shape (ndata, ndim)) so
        #    we can gather from it for each mask.
        #    This is easier than repeating arange(...) in each block.
        #    self.dim_arange is shape [ndim], so expand to (ndata, ndim):
        dim_all = self.dim_arange.unsqueeze(0).expand(len(x), self.ndim)

        # ~~~~~ Extrapolation: index == 0 ~~~~~
        select0 = (index == 0)
        if select0.any():
            dim0 = dim_all[select0]  # shape (#selected,)
            # y = yy[dim0, 0] + (x - xx[dim0, 0]) * delta[dim0, 0]
            y[select0] = (
                yy[dim0, 0]
                + (x[select0] - xx[dim0, 0]) * delta[dim0, 0]
            )
            new_logderiv[select0] = logderiv[dim0, 0]

        # ~~~~~ Extrapolation: index == nknot ~~~~~
        selectn = (index == self.nknot)
        if selectn.any():
            dimn = dim_all[selectn]
            y[selectn] = (
                yy[dimn, -1]
                + (x[selectn] - xx[dimn, -1]) * delta[dimn, -1]
            )
            new_logderiv[selectn] = logderiv[dimn, -1]

        # ~~~~~ Rational Quadratic Spline: 0 < index < nknot ~~~~~
        select_mid = ~(select0 | selectn)
        if select_mid.any():
            idx_mid = index[select_mid]  # shape (#selected,)
            dim_mid = dim_all[select_mid]

            # Distances in x- and y-knots
            x_lo = xx[dim_mid, idx_mid - 1]
            x_hi = xx[dim_mid, idx_mid]
            y_lo = yy[dim_mid, idx_mid - 1]
            y_hi = yy[dim_mid, idx_mid]
            delta_lo = delta[dim_mid, idx_mid - 1]
            delta_hi = delta[dim_mid, idx_mid]

            # xi in [0,1]
            xval = x[select_mid]
            xi = (xval - x_lo) / (x_hi - x_lo)  # shape (#selected,)

            s = (y_hi - y_lo) / (x_hi - x_lo)   # local slope
            xi1_xi = xi * (1 - xi)
            delta_sum = (delta_hi + delta_lo - 2 * s)
            denominator = s + delta_sum * xi1_xi

            xi2 = xi**2
            # y formula
            y[select_mid] = (
                y_lo
                + (y_hi - y_lo) * (s * xi2 + delta_lo * xi1_xi) / denominator
            )

            # log|dy/dx|
            # logderiv = 2 * log(s) + log(delta_hi*xi^2 + 2*s*xi(1-xi) + delta_lo*(1-xi)^2) - 2*log(denominator)
            log_s = torch.log(s)
            log_num = torch.log(delta_hi * xi2 + 2*s*xi1_xi + delta_lo * (1 - xi)**2)
            log_den = torch.log(denominator)

            new_logderiv[select_mid] = 2 * log_s + log_num - 2 * log_den

        # NOTE: We removed the "assert (discriminant >= 0).all()" 
        #       to avoid sync. If you still want to check for debugging:
        # if torch.any(denominator < 1e-12):
        #     print("Warning: possible negative or zero denominator in RQspline forward")

        return y, new_logderiv

    def inverse(self, y, params):
        """
        Inverse transform: y -> x
        y: shape (ndata, ndim)
        Returns:
          x: shape (ndata, ndim)
          logderiv: shape (ndata, ndim), the log|dx/dy|
        """
        x0, y0, logdx, logdy, logderiv = self.get_params(params)
        xx, yy, delta = self._prepare(x0, y0, logdx, logdy, logderiv)  # shape (ndim, nknot) each
        #xx, yy, delta = self._prepare()  # shape (ndim, nknot)

        #index = broadcast_searchsorted_multi(x, xx)
        index = torch.searchsorted(yy, y.T.contiguous()).T  # shape (ndata, ndim)
        x = torch.zeros_like(y)
        new_logderiv = torch.zeros_like(y)

        # Single "dim_all" array
        dim_all = self.dim_arange.unsqueeze(0).expand(len(y), self.ndim)

        # ~~~~~ Extrapolation: index == 0 ~~~~~
        select0 = (index == 0)
        if select0.any():
            dim0 = dim_all[select0]
            x[select0] = (
                xx[dim0, 0]
                + (y[select0] - yy[dim0, 0]) / delta[dim0, 0]
            )
            new_logderiv[select0] = logderiv[dim0, 0]

        # ~~~~~ Extrapolation: index == nknot ~~~~~
        selectn = (index == self.nknot)
        if selectn.any():
            dimn = dim_all[selectn]
            x[selectn] = (
                xx[dimn, -1]
                + (y[selectn] - yy[dimn, -1]) / delta[dimn, -1]
            )
            new_logderiv[selectn] = logderiv[dimn, -1]

        # ~~~~~ Rational Quadratic Spline: 0 < index < nknot ~~~~~
        select_mid = ~(select0 | selectn)
        if select_mid.any():
            idx_mid = index[select_mid]
            dim_mid = dim_all[select_mid]

            y_lo = yy[dim_mid, idx_mid - 1]
            y_hi = yy[dim_mid, idx_mid]
            x_lo = xx[dim_mid, idx_mid - 1]
            x_hi = xx[dim_mid, idx_mid]

            delta_lo = delta[dim_mid, idx_mid - 1]
            delta_hi = delta[dim_mid, idx_mid]

            s = (y_hi - y_lo) / (x_hi - x_lo)
            delta_2s = (delta_hi + delta_lo - 2 * s)
            yval = y[select_mid]
            y_off = (yval - y_lo) * delta_2s  # factor used in the quartic eq

            # a, b, c in the standard "ax^2 + bx + c = 0" form (see doc)
            # but rearranged for rational quadratic invert.
            # We'll keep the same variable names from your code:
            deltayy = (y_hi - y_lo)
            a = deltayy * (s - delta_lo) + y_off
            b = deltayy * delta_lo - y_off
            c = -s * (yval - y_lo)

            # Solve for xi via the formula:
            #   xi = [ -2c / (b + sqrt(b^2 - 4ac)) ] 
            # (discriminant must be >= 0 for monotonic)
            discriminant = b.pow(2) - 4 * a * c
            # Optional check for debugging if you see NaNs:
            # if torch.any(discriminant < 0):
            #     print("Warning: negative discriminant encountered in RQspline inverse")

            sqrt_disc = torch.sqrt(discriminant)
            xi = -2 * c / (b + sqrt_disc)

            xi1_xi = xi * (1 - xi)

            x[select_mid] = (
                xi * (x_hi - x_lo)
                + x_lo
            )

            # log|dx/dy|
            # formula from rational-quadratic paper
            log_s = torch.log(s)
            log_numer = torch.log(
                delta_hi * xi**2 + 2 * s * xi1_xi + delta_lo * (1 - xi)**2
            )
            denom = s + delta_2s * xi1_xi
            log_denom = torch.log(denom)

            new_logderiv[select_mid] = 2 * log_s + log_numer - 2 * log_denom

        return x, new_logderiv


class PixelwiseNonlinearity(nn.Module):
    """
    A pixelwise transform that applies the same 1D monotonic rational-quadratic spline
    to each pixel value. This is a 'normalizing flow' style transformation:
        - forward(...)  :  x -> y = spline(x)
        - forward(..., reverse=True):  y -> x = spline^{-1}(y)
    """
    def __init__(self, ndim=1, nknot=8):
        """
        nknot: number of knots in the rational-quadratic spline.
        """
        super().__init__()
        
        self.ndim = ndim  
        self.nknot = nknot
        
        # Our RQspline class expects (ndim, nknot). We do (1, nknot).
        self.rqspline = RQspline(ndim=self.ndim, nknot=self.nknot)

        self.num_params = 2 * ndim + 2 * (ndim * nknot - 1) + ndim * nknot
        #self.rqspline = PLspline(ndim=self.ndim, nknot=self.nknot)


    def forward(self, x, params, logdet=None, reverse=False):
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
            y_flat, logabsdet_flat = self.rqspline.forward(x_flat, params)
            # y_flat: (B*N_pixels, 1)
            # logabsdet_flat: (B*N_pixels, 1) or (B*N_pixels,) depending on your RQspline code
        else:
            # inverse transform: x = rqspline^{-1}(y)
            y_flat, logabsdet_flat = self.rqspline.inverse(x_flat, params)
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
