import os
import torch
from skimage import io
from torch.utils.data import Dataset


class KinFaceDataset(Dataset):

    def __init__(self, labels_df, root_dir, transform=None, color_space="rgb"):
        self.labels_df = labels_df
        self.root_dir = root_dir
        self.transform = transform
        self.color_space = color_space

    def __len__(self):
        return len(self.labels_df)

    def get_image_path(self, image_name):
        prefix = image_name.split('_')[0]
        image_path = ""
        if prefix == "fs":
            image_path = os.path.join(self.root_dir, "images/father-son", image_name)
        elif prefix == "fd":
            image_path = os.path.join(self.root_dir, "images/father-dau", image_name)
        elif prefix == "md":
            image_path = os.path.join(self.root_dir, "images/mother-dau", image_name)
        elif prefix == "ms":
            image_path = os.path.join(self.root_dir, "images/mother-son", image_name)
        return image_path

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        parent_image_path = self.get_image_path(self.labels_df.iloc[idx, 2])
        children_image_path = self.get_image_path(self.labels_df.iloc[idx, 3])

        parent_image = io.imread(parent_image_path)
        children_image = io.imread(children_image_path)

        if self.color_space == "bgr":
            parent_image = parent_image[:, :, ::-1]
            children_image = parent_image[:, :, ::-1]

        fold = self.labels_df.iloc[idx, 0]
        kin = self.labels_df.iloc[idx, 1]

        if self.transform:
            parent_image = self.transform(parent_image)
            children_image = self.transform(children_image)

        parent_image = parent_image.double()
        children_image = children_image.double()

        sample = {"parent_image": parent_image,
                  "children_image": children_image,
                  "kin": kin,
                  "parent_image_name": self.labels_df.iloc[idx, 2],
                  "children_image_name": self.labels_df.iloc[idx, 3],
                  "fold": fold}
        return sample
