import os
import torch
import pandas as pd
from scipy.io import loadmat
from datasets.kinfacew_dataset import KinFaceDataset


class KinFaceWLoaderGenerator:

    def __init__(self, dataset_name, dataset_path, color_space_name):
        assert dataset_name in ["KinFaceW-I", "KinFaceW-II"]
        self.dataset_name = dataset_name
        self.color_space_name = color_space_name
        self.dataset_path = os.path.join(dataset_path, self.dataset_name)
        print(f"Parsing {dataset_name} dataset")
        self.kin_pairs = pd.concat([
            self.parse_kin_pairs(loadmat(os.path.join(self.dataset_path, "meta_data/fs_pairs.mat"))),
            self.parse_kin_pairs(loadmat(os.path.join(self.dataset_path, "meta_data/fd_pairs.mat"))),
            self.parse_kin_pairs(loadmat(os.path.join(self.dataset_path, "meta_data/md_pairs.mat"))),
            self.parse_kin_pairs(loadmat(os.path.join(self.dataset_path, "meta_data/ms_pairs.mat")))
        ])
        self.kin_pairs["pair_type"] = self.kin_pairs["image_1"].apply(lambda x: x.split('_')[0])

    def parse_kin_pairs(self, kin_pair):
        result = pd.DataFrame(columns=["fold", "kin", "image_1", "image_2"])
        for pair in kin_pair["pairs"]:
            pair_data = pair.tolist()
            result.loc[len(result)] = [pair_data[0][0][0], pair_data[1][0][0], pair_data[2][0], pair_data[3][0]]
        return result

    def get_data_loader(self, fold, batch_size, pair_type, train_transformer, test_transformer):

        assert pair_type in ["fs", "fd", "md", "ms"]
        kin_pairs = self.kin_pairs[self.kin_pairs["pair_type"] == pair_type]
        test_data_pairs = kin_pairs[kin_pairs["fold"] == fold]
        train_data_pairs = kin_pairs[kin_pairs["fold"] != fold]
        kinfacew_dataset_test = KinFaceDataset(test_data_pairs, self.dataset_path, test_transformer,
                                               self.color_space_name)
        kinfacew_dataset_train = KinFaceDataset(train_data_pairs, self.dataset_path, train_transformer,
                                                self.color_space_name)

        dataloader_test = torch.utils.data.DataLoader(kinfacew_dataset_test, batch_size=batch_size,
                                                      shuffle=True, num_workers=4)
        dataloader_train = torch.utils.data.DataLoader(kinfacew_dataset_train, batch_size=batch_size,
                                                      shuffle=True, num_workers=4)

        return dataloader_test, dataloader_train

    def get_data_loader_full(self, batch_size, pair_type, transformer):
        assert pair_type in ["fs", "fd", "md", "ms"]
        kin_pairs = self.kin_pairs[self.kin_pairs["pair_type"] == pair_type]
        kinfacew_dataset = KinFaceDataset(kin_pairs, self.dataset_path, transformer, self.color_space_name)

        kinface_dataloader = torch.utils.data.DataLoader(kinfacew_dataset, batch_size=batch_size,
                                                      shuffle=True, num_workers=4)

        return kinface_dataloader
