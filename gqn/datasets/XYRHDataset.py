import torch
from gqn.datasets import FoldersDataset


class XYRHDataset(FoldersDataset):
    # position=(x, y), viewpoint=(rotation, horizon)

    def transform_viewpoint(self, v):
        l, y, p = torch.split(v, [2, 1, 1], dim=-1)
        v_hat = [l, torch.cos(y), torch.sin(y), torch.cos(p), torch.sin(p)]
        v_hat = torch.cat(v_hat, dim=-1)
        return v_hat
