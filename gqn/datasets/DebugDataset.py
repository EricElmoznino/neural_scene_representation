import torch
from torch.utils.data import Dataset


class DebugDatset(Dataset):
    """
    Random placeholder dataset for testing
    training loop without loading actual data.
    """
    def __init__(self, data_dir, resolution, max_viewpoints=None):
        super().__init__()
        self.res = resolution
        self.v_dim = 7

    def __len__(self):
        return 128

    def __getitem__(self, idx):
        images = torch.randn(15, 3, self.res, self.res)
        viewpoints = torch.randn(15, self.v_dim)

        return images, viewpoints
