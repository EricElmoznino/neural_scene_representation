import os
import random
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as tr
from torch.utils.data import Dataset


class FoldersDataset(Dataset):
    """
    Dataset that assumes images in the following folder format:
    - data_dir
        - scene_1
            - image_1.[jpg/png/etc.] (image file for a given view)
            - ...
            - viewpoints.npy (viewpoint data for all images in the scene sorted alphabetically)
        - scene_2
            - ...
        - ...
    """
    def __init__(self, data_dir, resolution, max_viewpoints=None):
        super().__init__()
        self.scenes = load_data(data_dir)
        self.v_dim = self.scenes[0][1].shape[1]
        self.res = resolution
        self.max_viewpoints = max_viewpoints

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        images, viewpoints = self.scenes[idx]

        if self.max_viewpoints is not None and len(images) > self.max_viewpoints:
            indices = random.sample([i for i in range(len(images))], self.max_viewpoints)
            images, viewpoints = images[indices], viewpoints[indices]

        images = [Image.open(img) for img in images]
        images = [tr.to_tensor(img) for img in images]
        images = [tr.resize(img, [self.res, self.res]) for img in images
                  if img.shape[1] != self.res or img.shape[2] != self.res]
        images = torch.stack(images)

        viewpoints = torch.from_numpy(viewpoints)

        return images, viewpoints


def load_data(data_dir):
    data = []
    scenes = os.listdir(data_dir)
    scenes = [os.path.join(data_dir, s) for s in scenes]
    for s in scenes:
        images = os.listdir(s)
        images = [img for img in images if img != 'views.npy' and img != '.DS_Store']
        images = [os.path.join(s, img) for img in images]
        images = np.array(images)
        images.sort()
        viewpoints = np.load(os.path.join(s, 'viewpoints.npy'))
        data.append([images, viewpoints])
    return data
