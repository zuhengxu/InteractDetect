import sys
from typing import Union

from tqdm import tqdm


import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split

class LabelMap:
    """
    Example:
    ----------------------------
    di = LabelMap([0,1 ,2 ,3])
    di.tb
    di.invtb
    di.get_flatten_idx({1, 2})
    di.get_unflatten_idx(5)
    """
    def __init__(self, single_perts_list):
        self.nsingle_perts = len(single_perts_list)
        self.all_double = [
            (single_perts_list[i], single_perts_list[j])
            for i in range(self.nsingle_perts)
            for j in range(i + 1, self.nsingle_perts)
        ]
        self.all_singles = [("control",)] + [(i, ) for i in single_perts_list] 
        self.nclass = 1 + self.nsingle_perts + len(self.all_double)
        self.all_class = self.all_singles + self.all_double
        self.labels = [i for i in range(self.nclass)]
        self.tb = {i: self.all_class[i] for i in range(self.nclass)}
        self.invtb = {v: k for k, v in self.tb.items()}

    def get_flatten_idx(self, c):
        return self.invtb[tuple(c)]

    def get_unflatten_idx(self, idx):
        return set(self.tb[idx])

#################################
# data module for NRE training
##################################
class Balldata(Dataset):
    """
    Dataset instantiation for training a neural ratio estimator
    """
    def __init__(
        self,
        data_dir="example/ball/data/obs/dat_all.npy",
        label_dir="example/ball/data/obs/label_all.npy",
    ):
        self.data = torch.tensor(np.load(data_dir), dtype=torch.float32).swapaxes(
            1, 3
        )  # put channel dimension to the second
        self.labels = torch.LongTensor(np.load(label_dir))
        self.label_set = torch.unique(self.labels)
        self.labelmap = LabelMap([0, 1, 2, 3])

    def __len__(self):
        assert self.data.shape[0] == self.labels.shape[0]
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    # get control or perts data
    def get_control(self):
        print(f"extracting control group")
        return self.data[self.labels == 0]

    def get_perturb(self, perturbation: Union[int, tuple]):
        if isinstance(perturbation, tuple):
            c1 = self.labelmap.get_flatten_idx(perturbation)
        else:
            c1 = self.labelmap.get_flatten_idx((perturbation,))
        print(f"extracting data for label {c1}")
        return self.data[self.labels == c1]


# dat_obj = Balldata()
# dat_obj.get_perturb(1)
# Xp = dat_obj.get_perturb((1,2))
# Xc = dat_obj.get_control()


class Balldatamodule(pl.LightningDataModule):
    """
    DataModule instantiation for synthetic NRE training
    """

    def __init__(
        self,
        data_dir="example/ball/data/obs/dat_all.npy",
        label_dir="example/ball/data/obs/label_all.npy",
        batch_size: int = 128,
        num_workers: int = 4,
        train_val_test_split=(0.7, 0.3, 0),
        random_seed=123,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.dataset = Balldata(data_dir, label_dir)
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
            generator=torch.Generator().manual_seed(self.random_seed),
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
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


class kl_ball_est:
    def __init__(
        self,
        dataloader: Balldata, 
        pert_list: list = [0,1, 2, 3],
    ):
        self.pert_list = pert_list
        self.dataloader = dataloader
        self.Xc = self.dataloader.get_control()
        self.kls = None

    def _get_perturb(self, perturbation: Union[int, tuple]):
        return self.dataloader.get_perturb(perturbation)

    def kl_single_pair(
        self, perturbation: Union[int, tuple], func_kl_est, **kargs
    ):
        Xp = self._get_perturb(perturbation)
        kl = func_kl_est(self.Xc, Xp, perturbation, **kargs)
        return kl

    def kl_all_pairs(self, func_kl_est, **kargs):
        kl_est_save = []
        labels = []

        # single perturbation
        for g in tqdm(self.pert_list):
            kl = self.kl_single_pair(g, func_kl_est, **kargs)
            kl_est_save.append(kl)
            labels.append(str(g))

        # double perturbation
        nperts = len(self.pert_list)
        for i in tqdm(range(nperts)):
            for j in tqdm(range(i + 1, nperts), leave=False):
                gi = self.pert_list[i]
                gj = self.pert_list[j]

                kl = self.kl_single_pair((gi, gj), func_kl_est, **kargs)

                kl_est_save.append(kl)

                label = f"{gi}_{gj}"
                labels.append(label)

        df = pd.DataFrame({"perturbation": labels, "kl": kl_est_save})

        self.kls = df
        return df

    def kl_reward_mat(self):
        if self.kls is None:
            raise Exception(
                "First run kl_all_pairs to get the KL divergence estimates."
            )

        nperts = len(self.pert_list)
        rwd_mat = np.zeros((nperts, nperts))
        df = self.kls

        for i in range(nperts):
            for j in range(i + 1, nperts):
                gi = self.pert_list[i]
                gj = self.pert_list[j]
                klij = df.loc[df["perturbation"] == f"{gi}_{gj}", "kl"].values[0]
                kli = df.loc[df["perturbation"] == str(gi), "kl"].values[0]
                klj = df.loc[df["perturbation"] == str(gj), "kl"].values[0]
                rwd_mat[i, j] = np.abs(klij - kli - klj)

        rwd_mat[rwd_mat == 0] = np.nan
        return rwd_mat





#############################
# data module for smile training
#############################
class ball_smile_dataset(Dataset):
    """
    Dataset instantiation for training smile estiamtor for smile estimator of estimating KL(control||perturbation)
    """
    def __init__(
        self,
        data_dir="example/ball/data/obs/dat_all.npy",
        label_dir="example/ball/data/obs/label_all.npy",
        perturbation: set = {1},
    ):
        self.data = torch.as_tensor(np.load(data_dir), dtype=torch.float32).swapaxes(
            1, 3
        )  # put channel dimension to the second
        self.labels = torch.LongTensor(np.load(label_dir))
        self.label_set = torch.unique(self.labels)
        self.labelmap = LabelMap([0, 1, 2, 3])
        self.Xc = self.get_control()
        self.Xp = self.get_perturb(perturbation)

    def __len__(self):
        assert self.Xc.shape[0] == self.Xp.shape[0]
        return self.Xc.shape[0]

    def __getitem__(self, idx):
        return self.Xc[idx], self.Xp[idx]

    # get control or perts data
    def get_control(self):
        print("extracting control group")
        return self.data[self.labels == 0]

    def get_perturb(self, perturbation: set):
        c1 = self.labelmap.get_flatten_idx(perturbation)
        print(f"extracting data for label {c1}")
        return self.data[self.labels == c1]


class ball_smile_dataloader(pl.LightningDataModule):
    """
    DataModule instantiation for smile training
    """
    def __init__(
        self,
        data_dir="example/ball/data/obs/dat_all.npy",
        label_dir="example/ball/data/obs/label_all.npy",
        perturbation: set = {1},
        batch_size: int = 128,
        num_workers: int = 4,
        train_val_test_split=(1.0, 0, 0),
        random_seed=123,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.dataset = ball_smile_dataset(data_dir, label_dir, perturbation)
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
            generator=torch.Generator().manual_seed(self.random_seed),
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
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

