# sys.path.insert(1, "../../../")  # load the repo root directory
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import pytorch_lightning as pl
import submitit
import torch
import wandb
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn, optim

from example.synthetic.causalindep.datamodule import (
    CISyncDataLoader,
    LoopUnroll,
    NREdatamodule,
)
from example.synthetic.causalindep.kl_est import kl_sync_est
from example.utils.plotting import plot_matrix
from inference.CausalIndep.klestimator.neuralratio import NREHermans, NRSmlp

single_perts_list = [0, 1, 2, 3]
label_dict = LoopUnroll(single_perts_list)


# define the klestimator function
def kl_est_nre(Xc, Xp, perts, model):
    if isinstance(perts, list):
        gi, gj = int(perts[0]), int(perts[1])
        c1 = label_dict.get_flatten_idx((gi, gj))
    else:
        c1 = label_dict.get_flatten_idx(int(perts))
    ldrs = model.logdensityratio(Xc, c1=[c1], c0=[0])
    kl = ldrs.mean()
    return torch.abs(kl).item()


def train_nre(max_epochs, batch_size, lr, res_dir_prefix):
    date_stamp = datetime.now().strftime("%Y-%m-%d")
    time_stamp = datetime.now().strftime("%H-%M-%S")
    # initialize wandb
    wandb_logger = pl.loggers.WandbLogger(
        project="NRE-synthetic",
        name=f"{max_epochs}_{batch_size}_{lr}_{date_stamp}_{time_stamp}",
        config={
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "lr": lr,
        },
    )
    logging.info(
        f"training sync NRE model with max_epochs: {max_epochs}, batch_size: {batch_size}, lr: {lr}"
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(res_dir_prefix, "checkpoints/"),
        filename=f"{max_epochs}_{batch_size}_{lr}",
        monitor="val_loss",
        enable_version_counter=False,
    )
    model = NREHermans(
        nclass=11,
        nrs_model=NRSmlp,
        nrsmodel_hparams={
            "inputdim": 3,
            "hiddendim": 128,
            "featuredim": 64,
            "numclass": 11,
        },
        optimizer=optim.Adam,
        lr=lr,
    )

    wp_dir = "example/synthetic/causalindep/"
    data_dir = os.path.join(wp_dir, "data/obs/dat_all.npy")
    label_dir = os.path.join(wp_dir, "data/obs/labels_all.npy")
    datamodule = NREdatamodule(
        data_dir=data_dir, label_dir=label_dir, batch_size=batch_size, num_workers=4
    )
    # logdir = res_dir_prefix + "logging/"
    # logger = loggers.CSVLogger(logdir, name = unpack_perturbation_name(perturbation))
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=1,
        logger=wandb_logger,
        log_every_n_steps=10,
        callbacks=[checkpoint_callback],
        deterministic=True,
        # default_root_dir=logdir,
    )
    trainer.fit(model, datamodule=datamodule)
    logging.info("training completed")

    # compute kl metric using just trained nre
    logging.info("computing kl est using tarined logdensity ratio")
    model.eval()
    perturb_list = [str(i) for i in range(4)]
    sync_dat = CISyncDataLoader(datadir="example/synthetic/causalindep/data/obs/")
    kl_obj = kl_sync_est(pert_list=perturb_list, dataloader=sync_dat)

    kl_df = kl_obj.kl_all_pairs(kl_est_nre, dynamic=True, model=model)
    kl_df.to_csv(
        os.path.join(
            res_dir_prefix, f"{max_epochs}_{batch_size}_{lr}" + "kl_nre_est.csv"
        ),
        index=False,
    )

    kl_mat = kl_obj.kl_reward_mat()
    np.save(
        os.path.join(
            res_dir_prefix, f"{max_epochs}_{batch_size}_{lr}" + "nre_reward_mat.npy"
        ),
        kl_mat,
    )

    # log kl result into wandb
    # import pdb; pdb.set_trace()
    cols = ["perturbation", "kl", "kl_diff"]
    vals = kl_df.to_numpy()

    # flatten kl mat and drop nan
    _kl_diff_flat = kl_mat.flatten()
    kl_diff_flat = [a for a in _kl_diff_flat if not np.isnan(a)]
    inter_score = np.array([0, 0, 0, 0] + kl_diff_flat).reshape(10, 1)
    vals = np.concatenate((vals, inter_score), axis=1)
    wandb_logger.log_table(key="kl_est", columns=cols, data=vals)

    # log image
    fig, ax = plot_matrix(
        kl_mat, [str(i) for i in single_perts_list], title="", cbar_label=""
    )
    fig_dir = os.path.join(res_dir_prefix, "fig/")
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    fig_pth = os.path.join(fig_dir, f"{max_epochs}_{batch_size}_{lr}_" + "nre_mat.pdf")
    plt.savefig(fig_pth)

    wandb_logger.log_image(key="kl_mat", images=[fig])

    # complete wandb
    wandb.finish()


def run_single(max_epochs=1000, batch_size=128, lr=1e-3):
    res_dir_prefix = "example/synthetic/causalindep/result/nre/"
    if not os.path.exists(res_dir_prefix):
        os.makedirs(res_dir_prefix)
    train_nre(max_epochs, batch_size, lr, res_dir_prefix)


def submit_sweep():
    ##################################
    # setup slurm executor
    ##################################
    date_stamp = datetime.now().strftime("%Y-%m-%d")
    time_stamp = datetime.now().strftime("%H-%M-%S")

    res_dir_prefix = "example/synthetic/causalindep/result/nre/"
    if not os.path.exists(res_dir_prefix):
        os.makedirs(res_dir_prefix)
    executor = submitit.AutoExecutor(
        folder=res_dir_prefix + f"outputs/{date_stamp}/{time_stamp}"
    )
    executor.update_parameters(
        slurm_array_parallelism=30,
        mem_gb=80,
        timeout_min=60 * 5,
        name="nre_sync",
        gpus_per_node=1,
        cpus_per_task=4,
        nodes=1,
        tasks_per_node=1,
        # slurm_mem_per_cpu=1000
    )
    ##################################
    # submit jobs
    ##################################
    max_eps_all = [500, 1000]
    bs_all = [512, 1024]
    lr_all = [5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4]
    jobs = []
    args = [(ep, bs, lr) for ep in max_eps_all for bs in bs_all for lr in lr_all]
    jobs = []
    with executor.batch():
        for arg in args:
            ep, bs, lr = arg
            job = executor.submit(run_single, max_epochs=ep, batch_size=bs, lr=lr)
            jobs.append(job)


if __name__ == "__main__":
    # run_single(max_epochs=2, batch_size=256, lr=1e-3)
    submit_sweep()
