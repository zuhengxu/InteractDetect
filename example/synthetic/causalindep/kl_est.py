import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], "../../../"))

from typing import Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from example.synthetic.causalindep.datamodule import CISyncDataLoader

class kl_sync_est:
    def __init__(
        self,
        pert_list: list = [str(i) for i in range(4)],
        dataloader: CISyncDataLoader = CISyncDataLoader(datadir="data/obs/"),
    ):
        self.pert_list = pert_list
        self.dataloader = dataloader
        self.Xc = self.dataloader.get_control()
        self.kls = None

    def _get_perturb(self, perturbation: Union[str, list[str]]):
        return self.dataloader.get_perturb(perturbation)

    def kl_single_pair(self, perturbation: Union[str, list[str]], func_kl_est, **kargs):
        Xp = self._get_perturb(perturbation)
        kl = func_kl_est(self.Xc, Xp, **kargs)
        return kl

    def kl_single_pair_dynamic(self, perturbation: Union[str, list[str]], func_kl_est_dyn, **kargs):
        Xp = self._get_perturb(perturbation)
        kl = func_kl_est_dyn(self.Xc, Xp, perturbation, **kargs)
        return kl

    def kl_all_pairs(self, func_kl_est, dynamic=False, **kargs):
        kl_est_save = []
        labels = []

        # single perturbation
        for g in tqdm(self.pert_list):
            if dynamic:
                kl = self.kl_single_pair_dynamic(g, func_kl_est, **kargs)
            else:
                kl = self.kl_single_pair(g, func_kl_est, **kargs)
            kl_est_save.append(kl)
            labels.append(g)

        # double perturbation
        nperts = len(self.pert_list)
        for i in tqdm(range(nperts)):
            for j in tqdm(range(i + 1, nperts), leave=False):
                gi = self.pert_list[i]
                gj = self.pert_list[j]

                if dynamic:
                    kl = self.kl_single_pair_dynamic([gi, gj], func_kl_est, **kargs)
                else:
                    kl = self.kl_single_pair([gi, gj], func_kl_est, **kargs)

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
                kli = df.loc[df["perturbation"] == gi, "kl"].values[0]
                klj = df.loc[df["perturbation"] == gj, "kl"].values[0]
                rwd_mat[i, j] = np.abs(klij - kli - klj)

        rwd_mat[rwd_mat == 0] = np.nan
        return rwd_mat

