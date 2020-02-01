import os
import random
import numpy as np
from PIL import Image
from torchvision.transforms import functional as tr
from torch.utils.data import Dataset


class FoldersDataset(Dataset):
    """
    Dataset that assumes images in the following folder format:
    - data_dir
        - scene_1
            - 00000.[jpg/png/etc.] (image file for first view)
            - 00001.[jpg/png/etc.] (image file for second view)
            - ...
        - scene_2
            - ...
        - ...
    """
    def __init__(self, data_dir, resolution):
        super().__init__()
        self.scenes = load_scenes(data_dir)
        self.res = resolution

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        images = scene_data(self.scenes[idx])
        if len(images) == 0:
            return None

        index = random.sample([i for i in range(len(images))], 1)[0]
        image = images[index]

        image = Image.open(image)
        image = tr.resize(image, [self.res, self.res])
        image = tr.to_tensor(image)

        return image


def load_scenes(data_dir):
    scenes = os.listdir(data_dir)
    scenes = [s for s in scenes if s != '.DS_Store']
    scenes = sorted(scenes)
    scenes = [os.path.join(data_dir, s) for s in scenes]
    return scenes


def scene_data(scene_dir):
    images = os.listdir(scene_dir)
    images = [img for img in images if img != 'viewpoints.npy' and img != '.DS_Store']
    images = [os.path.join(scene_dir, img) for img in images]
    images = np.array(images)
    images.sort()
    return images
