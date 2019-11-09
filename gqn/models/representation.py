import torch
import torch.nn as nn
import torch.nn.functional as F


class RepresentationNetwork(nn.Module):
    def __init__(self, n_channels, v_dim, r_dim=256, pool=True):
        """
        Network that generates a condensed representation
        vector from a joint input of image and viewpoint.

        Employs the tower/pool architecture described in the paper.

        :param n_channels: number of color channels in input image
        :param v_dim: dimensions of the viewpoint vector
        :param r_dim: dimensions of representation
        :param pool: whether to pool representation
        """
        super().__init__()

        # Final representation size
        self.r_dim = k = r_dim
        self.pool = pool

        self.conv1 = nn.Conv2d(n_channels, k, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(k, k, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(k, k // 2, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(k // 2, k, kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(k + v_dim, k, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(k + v_dim, k // 2, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(k // 2, k, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(k, k, kernel_size=1, stride=1)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, v):
        """
        Send an (image, viewpoint) pair into the
        network to generate a representation
        :param x: image
        :param v: viewpoint
        :return: representation
        """
        # First skip-connected conv block
        skip_in = F.relu(self.conv1(x))
        skip_out = F.relu(self.conv2(skip_in))

        x = F.relu(self.conv3(skip_in))
        x = F.relu(self.conv4(x)) + skip_out

        # Second skip-connected conv block (merged)
        v = v.view(v.size(0), -1, 1, 1)
        v = v.repeat(1, 1, x.shape[-2], x.shape[-1])
        skip_in = torch.cat([x, v], dim=1)
        skip_out = F.relu(self.conv5(skip_in))

        x = F.relu(self.conv6(skip_in))
        x = F.relu(self.conv7(x)) + skip_out

        r = F.relu(self.conv8(x))

        if self.pool:
            r = self.avgpool(r)

        return r