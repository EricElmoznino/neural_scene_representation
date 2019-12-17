import torch
from torch import nn
from .conv_lstm import Conv2dLSTMCell


class InferenceCore(nn.Module):
    def __init__(self, c_dim, v_dim, r_dim, h_dim):
        super().__init__()
        self.downsample_x = nn.Conv2d(c_dim, c_dim, kernel_size=4, stride=4, padding=0, bias=False)
        self.upsample_v = nn.ConvTranspose2d(v_dim, v_dim, kernel_size=16, stride=16, padding=0, bias=False)
        self.upsample_r = nn.ConvTranspose2d(r_dim, r_dim, kernel_size=16, stride=16, padding=0, bias=False)
        self.downsample_u = nn.Conv2d(h_dim, h_dim, kernel_size=4, stride=4, padding=0, bias=False)
        self.core = Conv2dLSTMCell(c_dim + v_dim + r_dim + 2 * h_dim, h_dim, kernel_size=5, stride=1, padding=2)

    def forward(self, x, v, r, c_e, h_e, h_g, u):
        x = self.downsample_x(x)
        v = self.upsample_v(v.view(v.size(0), -1, 1, 1))
        if r.size(2) != h_e.size(2):
            r = self.upsample_r(r)
        u = self.downsample_u(u)
        c_e, h_e = self.core(torch.cat((x, v, r, h_g, u), dim=1), (c_e, h_e))

        return c_e, h_e


class GenerationCore(nn.Module):
    def __init__(self, v_dim, r_dim, z_dim, h_dim):
        super().__init__()
        self.upsample_v = nn.ConvTranspose2d(v_dim, v_dim, kernel_size=16, stride=16, padding=0, bias=False)
        self.upsample_r = nn.ConvTranspose2d(r_dim, r_dim, kernel_size=16, stride=16, padding=0, bias=False)
        self.core = Conv2dLSTMCell(v_dim + r_dim + z_dim, h_dim, kernel_size=5, stride=1, padding=2)
        self.upsample_h = nn.ConvTranspose2d(h_dim, h_dim, kernel_size=4, stride=4, padding=0, bias=False)

    def forward(self, v, r, c_g, h_g, u, z):
        v = self.upsample_v(v.view(v.size(0), -1, 1, 1))
        if r.size(2) != h_g.size(2):
            r = self.upsample_r(r)
        c_g, h_g = self.core(torch.cat((v, r, z), dim=1), (c_g, h_g))
        u = self.upsample_h(h_g) + u

        return c_g, h_g, u
