import os
import random
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as tr
from torch.utils.data import Dataset


class FoldersDataset(Dataset):
    """
    Dataset that assumes scene data in the following folder format:
    - data_dir
        - scene_1
            - images.npy (all images of the scene from different viewpoints)
            - viewpoints.npy (viewpoint specification for all images)
        - scene_2
            - ...
        - ...
    """
    def __init__(self, data_dir, resolution, max_viewpoints=None):
        super().__init__()
        self.scenes = load_scenes(data_dir)
        self.res = resolution
        self.max_viewpoints = max_viewpoints
        self.v_dim = self.transform_viewpoint(torch.from_numpy(scene_data(self.scenes[0])[1])).shape[-1]

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        images, viewpoints = scene_data(self.scenes[idx])

        if self.max_viewpoints is not None and len(images) > self.max_viewpoints + 1:
            indices = random.sample([i for i in range(len(images))], self.max_viewpoints + 1)
            images, viewpoints = images[indices], viewpoints[indices]

        images = [Image.fromarray(img) for img in images]
        images = [tr.resize(img, [self.res, self.res]) for img in images
                  if img.height != self.res or img.width != self.res]
        images = [tr.to_tensor(img) for img in images]
        images = torch.stack(images)

        viewpoints = torch.from_numpy(viewpoints)
        viewpoints = self.transform_viewpoint(viewpoints)

        return images, viewpoints

    def transform_viewpoint(self, v):
        return v


def load_scenes(data_dir):
    scenes = os.listdir(data_dir)
    scenes = [s for s in scenes if s != '.DS_Store']
    scenes = [os.path.join(data_dir, s) for s in scenes]
    return scenes


def scene_data(scene_dir):
    images = np.load(os.path.join(scene_dir, 'images.npy'))
    viewpoints = np.load(os.path.join(scene_dir, 'viewpoints.npy')).astype(np.float32)
    return images, viewpoints
