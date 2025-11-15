# srm_frontend.py
# Small SRM-like front-end: applies a bank of high-pass kernels (residual filters),
# returns residual maps suitable for a downstream CNN.
#
# NOTE: This is a practical SRM-like filter bank (5 strong residual filters).
# If you want the full 30-filter SRM bank later, I can provide a kernel file
# and loader. This 5-filter bank already provides a strong residual signal for steganalysis.

import torch
import torch.nn as nn
import torch.nn.functional as F

class SRMFrontend(nn.Module):
    """
    Applies a fixed bank of high-pass filters to grayscale image and returns
    a residual tensor (C_resid x H x W). Non-trainable filters.
    """
    def __init__(self):
        super().__init__()
        # Define a small set of high-pass kernels (3x3 and 5x5 variants)
        # Kernels taken/constructed to approximate SRM-style residuals.
        k1 = torch.tensor([[0,  0,  0],
                           [0,  1, -1],
                           [0,  0,  0]], dtype=torch.float32)  # horizontal diff

        k2 = torch.tensor([[0, 0, 0],
                           [0, 1, 0],
                           [0, -1, 0]], dtype=torch.float32)   # vertical diff

        k3 = torch.tensor([[-1, 2, -1],
                           [2, -4, 2],
                           [-1, 2, -1]], dtype=torch.float32)  # second-derivative (Laplacian-ish)

        k4 = torch.tensor([[0, -1, 0],
                           [-1, 4, -1],
                           [0, -1, 0]], dtype=torch.float32)   # center-surround

        # a 5x5 high-order residual (approximates some SRM larger-kernel behaviour)
        k5 = torch.tensor([
            [0, 0, -1, 0, 0],
            [0, -1, 2, -1, 0],
            [-1, 2, 4, 2, -1],
            [0, -1, 2, -1, 0],
            [0, 0, -1, 0, 0]
        ], dtype=torch.float32)

        kernels = [k1, k2, k3, k4]

        # We'll handle kernel sizes of varying shapes by padding to same shape in forward
        self.register_buffer("kernels_3x3", torch.stack(kernels).unsqueeze(1))  # shape [4,1,3,3]
        self.register_buffer("kernel_5x5", k5.unsqueeze(0).unsqueeze(1))        # shape [1,1,5,5]

    def forward(self, x):
        """
        x: tensor (B,3,H,W) in range [0,1] (float)
        Returns: residual tensor (B, N_filters, H, W)
        """
        # convert to grayscale: simple average
        if x.shape[1] == 3:
            gray = 0.2989 * x[:,0:1,:,:] + 0.5870 * x[:,1:2,:,:] + 0.1140 * x[:,2:3,:,:]
        else:
            gray = x

        # apply 3x3 filters
        resid3 = F.conv2d(gray, self.kernels_3x3, padding=1)  # output shape B x 4 x H x W

        # apply 5x5 with padding=2
        resid5 = F.conv2d(gray, self.kernel_5x5, padding=2)   # B x 1 x H x W

        # Concatenate residuals
        resid = torch.cat([resid3, resid5], dim=1)  # B x 5 x H x W
        # Optionally normalize each residual map to zero mean unit std per sample
        # (helps the network converge)
        B, C, H, W = resid.shape
        # avoid division by zero
        resid = resid.view(B, C, -1)
        mean = resid.mean(dim=2, keepdim=True)
        std  = resid.std(dim=2, keepdim=True).clamp(min=1e-6)
        resid = (resid - mean) / std
        resid = resid.view(B, C, H, W)
        return resid
