import torch.nn as nn

from .generator import GeneratorNetwork
from .representation import RepresentationNetwork


class VAE(nn.Module):
    """
    Single-image VAE version of the Generative Query Network (GQN) as described
    in "Neural scene representation and rendering"
    [Eslami 2018].

    :param c_dim: number of channels in input
    :param r_dim: dimensions of representation
    :param z_dim: latent channels
    :param h_dim: hidden channels in LSTM
    :param l: Number of refinements of density
    """
    def __init__(self, c_dim, r_dim, z_dim=64, h_dim=128, l=8):
        super().__init__()
        self.r_dim = r_dim

        self.generator = GeneratorNetwork(c_dim, r_dim, z_dim, h_dim, l)
        self.representation = RepresentationNetwork(c_dim, r_dim, pool=True)

    def forward(self, x):
        """
        Forward through the GQN.

        :param x: batch of images [b, c, h, w]
        """
        # Representation generated from input image
        r = self.representation(x)

        # Use random (image, viewpoint) pair in batch as query
        x_mu, kl = self.generator(x, r)

        # Return reconstruction and query viewpoint
        # for computing error
        return x_mu, r, kl

    def sample(self, x):
        """
        Sample from the network given some image.

        :param x: image to generate representation
        """
        _,  _, h, w = x.shape

        r = self.representation(x)

        x_mu = self.generator.sample((h, w), r)

        return x_mu
