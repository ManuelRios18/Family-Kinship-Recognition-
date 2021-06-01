import os
import glob
import torch
import random
import numpy as np
import pandas as pd
from skimage import io
from skimage.transform import resize
from torch.utils.data import Dataset


class FIWDataset(Dataset):

    def __init__(self, root_dir, pair_type, set_name, transform=None, color_space="rgb"):
        self.root_dir = root_dir
        self.pair_type = pair_type
        self.set_name = set_name
        self.transform = transform
        self.color_space = color_space

        self.labels_df = pd.read_csv(os.path.join("data", f"{self.set_name}-pairs-full.csv"))
        self.labels_df = self.labels_df[self.labels_df.ptype == self.pair_type]

    def __len__(self):
        return len(self.labels_df)

    def get_image_path(self, image_name):
        if self.set_name == 'test':
            n = 'val'
        else:
            n= 'train'
        image_path = os.path.join(self.root_dir, n + '-faces', image_name)

        return image_path

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.labels_df.iloc[idx]
        parent_image_path = self.get_image_path(row.p1)
        children_image_path = self.get_image_path(row.p2)

        parent_image = io.imread(parent_image_path)
        children_image = io.imread(children_image_path)

        if self.color_space == "bgr":
            parent_image = parent_image[:, :, ::-1]
            children_image = parent_image[:, :, ::-1]

        kin = row.label

        if self.transform:
            parent_image = self.transform(parent_image)
            children_image = self.transform(children_image)

        sample = {"parent_image": parent_image,
                  "children_image": children_image,
                  "kin": kin,
                  "parent_image_name": parent_image_path,
                  "children_image_name": children_image_path,
                  "parent_family_id": parent_image_path.split(os.path.sep)[-3],
                  "children_family_id": children_image_path.split(os.path.sep)[-3]}
        return sample
