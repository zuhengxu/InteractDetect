import os
import sys
from typing import Union

sys.path.insert(1, "../../../")  # load the repo root directory

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split


class CISyncDataLoader:
    """
    emd extracter

    example:
    >>> emd = CISyncDataLoader(datadir = "example/synthetic/causalindep/data/obs/")
    >>> Xc = emd._get_control()
    >>> Xp = emd._get_emd("A")
    """

    def __init__(self, datadir: str = "example/synthetic/causalindep/data/obs/"):
        self.datadir = datadir

    def get_control(self):
        pth = os.path.join(self.datadir, "control.npy")
        Xc = np.load(pth)
        return torch.tensor(Xc, dtype=torch.float32)

    def get_perturb(self, perturbation: Union[str, list[str]]):
        if isinstance(perturbation, list):
            pert_name = "_".join(perturbation)
        elif isinstance(perturbation, str):
            pert_name = perturbation
        else:
            raise ValueError("Perturbation must be a string or a list of strings.")

        pth = os.path.join(self.datadir, f"{pert_name}.npy")
        X = np.load(pth)
        return torch.tensor(X, dtype=torch.float32)

class LoopUnroll:
    """
    Example:
    ----------------------------
    di = LoopUnroll([0,1 ,2 ,3])
    di.tb
    di.invtb
    di.get_flatten_idx(0)
    di.get_unflatten_idx(1)
    """
    def __init__(self, single_perts_list):
        self.nsingle_perts = len(single_perts_list)
        self.all_double = [
            (single_perts_list[i], single_perts_list[j])
            for i in range(self.nsingle_perts)
            for j in range(i + 1, self.nsingle_perts)
        ]
        self.all_singles = ["control"] + single_perts_list
        self.nclass = 1 + self.nsingle_perts + len(self.all_double)
        self.all_class = self.all_singles + self.all_double
        self.labels = [i for i in range(self.nclass)]
        self.tb = {i: self.all_class[i] for i in range(self.nclass)}
        self.invtb = {v: k for k, v in self.tb.items()}

    def get_flatten_idx(self, c):
        return self.invtb[c]

    def get_unflatten_idx(self, idx):
        return self.tb[idx]


class NREdata(Dataset):
    """
    Dataset instantiation for training a neural ratio estimator
    """
    def __init__(
        self,
        data_dir="example/synthetic/causalindep/data/obs/dat_all.npy",
        label_dir = "example/synthetic/causalindep/data/obs/labels_all.npy",
    ):
        self.data = torch.tensor(np.load(data_dir), dtype=torch.float32)
        self.labels = torch.LongTensor(np.load(label_dir))
        self.label_set = torch.unique(self.labels)

    def __len__(self):
        assert self.data.shape[0] == self.labels.shape[0]
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class NREdatamodule(pl.LightningDataModule):
    """
    DataModule instantiation for synthetic NRE training
    """
    def __init__(
        self, 
        data_dir="example/synthetic/causalindep/data/obs/dat_all.npy",
        label_dir = "example/synthetic/causalindep/data/obs/labels_all.npy",
        batch_size: int = 128, 
        num_workers: int = 2,
        train_val_test_split=(0.7, 0.3, 0), 
        random_seed=123
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.dataset = NREdata(data_dir, label_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_test_split = train_val_test_split
        self.random_seed = random_seed


    def setup(self, stage=None):
        train_size = int(self.train_val_test_split[0] * len(self.dataset))
        val_size = int(self.train_val_test_split[1] * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, 
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.random_seed)
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset, 
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

if __name__ == "__main__":
    # put all data in one big array 
    obs_dir = "data/obs/"

    label_dict = LoopUnroll([0, 1, 2, 3])
    labels = label_dict.labels
    map_unflatten = label_dict.get_unflatten_idx

    dat_all = []
    labels_all = []
    for l in labels:
        perts = map_unflatten(l)
        if isinstance(perts, tuple):
            perts = "_".join([str(p) for p in perts])
        else:
            perts = str(perts)

        pth = os.path.join(obs_dir, f"{perts}.npy")
        df = np.load(pth)
        dat_all.append(df)
        
        nsample = df.shape[0]
        labels_all.append([l]*nsample)
    
    dat_all = np.concatenate(dat_all, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)
    np.save(obs_dir + "dat_all.npy", dat_all)
    np.save(obs_dir + "labels_all.npy", labels_all)

