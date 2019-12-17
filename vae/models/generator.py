"""
The inference-generator architecture is conceptually
similar to the encoder-decoder pair seen in variational
autoencoders. The difference here is that the model
must infer latents from a cascade of time-dependent inputs
using convolutional and recurrent networks.

Additionally, a representation vector is shared between
the networks.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence
from .core import InferenceCore, GenerationCore


class GeneratorNetwork(nn.Module):
    """
    Network similar to a convolutional variational
    autoencoder that refines the generated image
    over a number of iterations.

    :param c_dim: number of channels in input
    :param r_dim: dimensions of representation
    :param z_dim: latent channels
    :param h_dim: hidden channels in LSTM
    :param l: number of density refinements
    :param shared_core: whether to share cores across refinements
    """

    def __init__(self, c_dim, r_dim, z_dim=3, h_dim=128, l=8, shared_core=True):
        super().__init__()

        self.z_dim = z_dim
        self.h_dim = h_dim
        self.l = l

        # Generation network
        self.shared_core = shared_core
        if shared_core:
            self.inference_core = InferenceCore(c_dim, r_dim, h_dim)
            self.generation_core = GenerationCore(r_dim, z_dim, h_dim)
        else:
            self.inference_core = nn.ModuleList([InferenceCore(c_dim, r_dim, h_dim) for _ in range(l)])
            self.generation_core = nn.ModuleList([GenerationCore(r_dim, z_dim, h_dim) for _ in range(l)])

        self.eta_pi = nn.Conv2d(h_dim, 2*z_dim, kernel_size=5, stride=1, padding=2)
        self.eta_g = nn.Conv2d(h_dim, c_dim, kernel_size=1, stride=1, padding=0)
        self.eta_e = nn.Conv2d(h_dim, 2*z_dim, kernel_size=5, stride=1, padding=2)

    def forward(self, x, r):
        """
        Attempt to reconstruct x with corresponding
        context representation r.

        :param x: image to send reconstruct
        :param r: representation for image
        :return reconstruction of x and kl-divergence
        """
        batch_size = x.size(0)

        # Generator initial state
        c_g = x.new_zeros((batch_size, self.h_dim, 16, 16))
        h_g = x.new_zeros((batch_size, self.h_dim, 16, 16))
        u = x.new_zeros((batch_size, self.h_dim, 64, 64))

        # Inference initial state
        c_e = x.new_zeros((batch_size, self.h_dim, 16, 16))
        h_e = x.new_zeros((batch_size, self.h_dim, 16, 16))

        kl = 0
        for l in range(self.l):
            # Prior factor
            mu_pi, logvar_pi = torch.split(self.eta_pi(h_g), self.z_dim, dim=1)
            std_pi = torch.exp(0.5*logvar_pi)
            pi = Normal(mu_pi, std_pi)

            # Inference state update
            if self.shared_core:
                c_e, h_e = self.inference_core(x, r, c_e, h_e, h_g, u)
            else:
                c_e, h_e = self.inference_core[l](x, r, c_e, h_e, h_g, u)

            # Posterior factor
            mu_q, logvar_q = torch.split(self.eta_e(h_e), self.z_dim, dim=1)
            std_q = torch.exp(0.5*logvar_q)
            q = Normal(mu_q, std_q)

            # Posterior sample
            z = q.rsample()

            # Generator state update
            if self.shared_core:
                c_g, h_g, u = self.generation_core(r, c_g, h_g, u, z)
            else:
                c_g, h_g, u = self.generation_core[l](r, c_g, h_g, u, z)

            # KL update
            kl += kl_divergence(q, pi)

        x_mu = self.eta_g(u)
        x_mu = torch.sigmoid(x_mu)

        return x_mu, kl

    def sample(self, r):
        """
        Sample from the prior distribution to generate
        a new image given a representation

        :param r: representation (context)
        """
        batch_size = r.size(0)

        # Initial state
        c_g = r.new_zeros((batch_size, self.h_dim, 16, 16))
        h_g = r.new_zeros((batch_size, self.h_dim, 16, 16))
        u = r.new_zeros((batch_size, self.h_dim, 64, 64))

        for l in range(self.l):
            # Prior factor
            mu_pi, logvar_pi = torch.split(self.eta_pi(h_g), self.z_dim, dim=1)
            std_pi = torch.exp(0.5*logvar_pi)
            pi = Normal(mu_pi, std_pi)

            # Prior sample
            z = pi.sample()

            # State update
            if self.shared_core:
                c_g, h_g, u = self.generation_core(r, c_g, h_g, u, z)
            else:
                c_g, h_g, u = self.generation_core[l](r, c_g, h_g, u, z)

        # Image sample
        x_mu = self.eta_g(u)
        x_mu = torch.sigmoid(x_mu)

        return x_mu
