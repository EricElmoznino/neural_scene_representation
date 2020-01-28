import torch
from torch.utils.data import Dataset


class DebugDatset(Dataset):
    """
    Random placeholder dataset for testing
    training loop without loading actual data.
    """
    def __init__(self, data_dir, resolution):
        super().__init__()
        self.res = resolution

    def __len__(self):
        return 128

    def __getitem__(self, idx):
        images = torch.randn(3, self.res, self.res)
        return images
