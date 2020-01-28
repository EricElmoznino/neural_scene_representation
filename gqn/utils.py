from torch.optim.lr_scheduler import _LRScheduler


# Learning rate at training step s with annealing
class AnnealingStepLR(_LRScheduler):
    def __init__(self, optimizer, mu_i, mu_f, n):
        self.mu_i = mu_i
        self.mu_f = mu_f
        self.n = n
        super().__init__(optimizer)

    def get_lr(self):
        return [max(self.mu_f + (self.mu_i - self.mu_f) * (1.0 - self.last_epoch / self.n), self.mu_f)
                for _ in self.base_lrs]


class AnnealingStepSigma(object):
    def __init__(self, sigma_i, sigma_f, n):
        self.sigma_i = sigma_i
        self.sigma_f = sigma_f
        self.n = n
        self.last_epoch = 0
        self.sigma = sigma_i

    def state_dict(self):
        return {"sigma_i": self.sigma_i, "sigma_f": self.sigma_f, "n": self.n, 'sigma': self.sigma}

    def step(self):
        self.sigma = max(self.sigma_f + (self.sigma_i - self.sigma_f) * (1 - self.last_epoch / self.n), self.sigma_f)
        self.last_epoch += 1
