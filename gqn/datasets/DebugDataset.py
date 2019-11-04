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
        self.max_viewpoints = max_viewpoints

    def __len__(self):
        return 128

    def __getitem__(self, idx):
        images = torch.randn(self.max_viewpoints + 1, 3, self.res, self.res)
        viewpoints = torch.randn(self.max_viewpoints + 1, self.v_dim)

        return images, viewpoints
