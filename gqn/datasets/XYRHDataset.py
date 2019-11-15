import os
import numpy as np
import torch
from gqn.datasets import FoldersDataset


class XYRHDataset(FoldersDataset):
    # position=(x, y), viewpoint=(rotation, horizon)

    def transform_viewpoint(self, v):
        l, y, p = torch.split(v, [2, 1, 1], dim=-1)
        v_hat = [l, torch.cos(y), torch.sin(y), torch.cos(p), torch.sin(p)]
        v_hat = torch.cat(v_hat, dim=-1)
        return v_hat


def normalize_locations(data_dir):
    """
    Utility function to run on the dataset before any training
    in order to normalize the viewpoint locations to range (-1, 1)
    :param data_dir:
    """
    scenes = os.listdir(data_dir)
    scenes = [s for s in scenes if s != '.DS_Store']
    scenes = [os.path.join(data_dir, s) for s in scenes]
    viewpoints = [os.path.join(s, 'viewpoints.npy') for s in scenes]
    max_range = 0
    for path in viewpoints:
        viewpoint = np.load(path)
        r = np.fabs(viewpoint[:, 0:2]).max()
        if r > max_range:
            max_range = r
    for path in viewpoints:
        viewpoint = np.load(path)
        viewpoint[:, 0:2] = viewpoint[:, 0:2] / max_range
        np.save(path, viewpoint)
