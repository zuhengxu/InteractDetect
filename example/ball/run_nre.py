# sys.path.insert(1, "../../../")  # load the repo root directoryrun
import logging
import os
from datetime import datetime
from typing import Union

import numpy as np
import pytorch_lightning as pl
import submitit
import torch
import wandb
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import optim

from example.ball.datamodule import Balldata, Balldatamodule, LabelMap, kl_ball_est
from example.ball.encoder import NRSimageencoder, ldr_image_encoder
from example.utils.plotting import plot_matrix
from inference.CausalIndep.klestimator.neuralratio import NREHermans
from inference.CausalIndep.klestimator.smile import smile_kl_est

single_perts_list = [0, 1, 2, 3]
labelmap = LabelMap(single_perts_list)

# define the klestimator function
def kl_est_nre(Xc, Xp, perts: Union[int, tuple], model: NREHermans):
    if isinstance(perts, tuple):
        gi, gj = int(perts[0]), int(perts[1])
        c1 = labelmap.get_flatten_idx((gi, gj))
    else:
        c1 = labelmap.get_flatten_idx((perts,))

    with torch.no_grad():
        ldrs = model.logdensityratio(Xc, c1=[c1], c0=[0])
        kl = ldrs.mean()
    return torch.abs(kl).item()


def kl_est_smile(Xc, Xp, perts: Union[int, tuple], model: NREHermans, tau=10):
    if isinstance(perts, tuple):
        gi, gj = int(perts[0]), int(perts[1])
        c1 = labelmap.get_flatten_idx((gi, gj))
    else:
        c1 = labelmap.get_flatten_idx((perts,))

    nre = model.score
    trained_ldr = ldr_image_encoder(nre, c1)
    with torch.no_grad():
        kl = smile_kl_est(trained_ldr, Xc, Xp, tau=tau)

    return torch.abs(kl).item()


def wandb_log_after_train(
    trained_model,
    kl_obj,
    wandb_logger,
    table_key="kl_test",
    image_key="kl_mat",
    res_dir_prefix="example/ball/result/nre/",
    csv_name="kl_nre_est.csv",
    mat_name="nre_reward_mat.npy",
    featuredim=128,
    max_epochs=1000,
    batch_size=128,
    lr=1e-3,
):
    # start computing kl
    kl_df = kl_obj.kl_all_pairs(kl_est_nre, model=trained_model)
    kl_df.to_csv(
        os.path.join(
            res_dir_prefix, f"{featuredim}_{max_epochs}_{batch_size}_{lr}" + csv_name
        ),
        index=False,
    )

    kl_mat = kl_obj.kl_reward_mat()
    np.save(
        os.path.join(
            res_dir_prefix, f"{featuredim}_{max_epochs}_{batch_size}_{lr}" + mat_name
        ),
        kl_mat,
    )

    # log kl result into wandb
    cols = ["perturbation", "kl", "kl_diff"]
    vals = kl_df.to_numpy()

    # flatten kl mat and drop nan
    _kl_diff_flat = kl_mat.flatten()
    kl_diff_flat = [a for a in _kl_diff_flat if not np.isnan(a)]
    inter_score = np.array([0, 0, 0, 0] + kl_diff_flat).reshape(10, 1)
    vals = np.concatenate((vals, inter_score), axis=1)

    wandb_logger.log_table(key=table_key, columns=cols, data=vals)

    # log image
    fig, ax = plot_matrix(
        kl_mat, [str(i) for i in single_perts_list], title="", cbar_label=""
    )
    fig_dir = os.path.join(res_dir_prefix, "fig/")
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    fig_pth = os.path.join(
        fig_dir, f"{featuredim}_{max_epochs}_{batch_size}_{lr}_" + f"{image_key}.pdf"
    )
    plt.savefig(fig_pth)

    wandb_logger.log_image(key=image_key, images=[fig])

    return kl_df, kl_mat


def wandb_log_after_train_smile(
    trained_model,
    kl_obj,
    wandb_logger,
    table_key="kl_test",
    image_key="kl_mat",
    res_dir_prefix="example/ball/result/nre/",
    csv_name="kl_nre_est.csv",
    mat_name="smile_reward_mat.npy",
    featuredim=128,
    max_epochs=1000,
    batch_size=128,
    lr=1e-3,
    tau=10,
):
    # start computing kl
    kl_df = kl_obj.kl_all_pairs(kl_est_smile, model=trained_model, tau=tau)
    kl_df.to_csv(
        os.path.join(
            res_dir_prefix, f"{featuredim}_{max_epochs}_{batch_size}_{lr}" + csv_name
        ),
        index=False,
    )

    kl_mat = kl_obj.kl_reward_mat()
    np.save(
        os.path.join(
            res_dir_prefix, f"{featuredim}_{max_epochs}_{batch_size}_{lr}" + mat_name
        ),
        kl_mat,
    )

    # log kl result into wandb
    cols = ["perturbation", "kl", "kl_diff"]
    vals = kl_df.to_numpy()

    # flatten kl mat and drop nan
    _kl_diff_flat = kl_mat.flatten()
    kl_diff_flat = [a for a in _kl_diff_flat if not np.isnan(a)]
    inter_score = np.array([0, 0, 0, 0] + kl_diff_flat).reshape(10, 1)
    vals = np.concatenate((vals, inter_score), axis=1)

    wandb_logger.log_table(key=table_key, columns=cols, data=vals)

    # log image
    fig, ax = plot_matrix(
        kl_mat, [str(i) for i in single_perts_list], title="", cbar_label=""
    )
    fig_dir = os.path.join(res_dir_prefix, "fig/")
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    fig_pth = os.path.join(
        fig_dir, f"{featuredim}_{max_epochs}_{batch_size}_{lr}_" + f"{image_key}.pdf"
    )
    plt.savefig(fig_pth)

    wandb_logger.log_image(key=image_key, images=[fig])

    return kl_df, kl_mat


def train_nre(
    featuredim,
    max_epochs,
    batch_size,
    lr,
    res_dir_prefix,
    tau_inference=[1, 5, 10, 20, 50, 100, 200],
):
    ##############################3
    # setup
    ################################
    pl.seed_everything(42, workers=True)

    date_stamp = datetime.now().strftime("%Y-%m-%d")
    time_stamp = datetime.now().strftime("%H-%M-%S")
    # initialize wandb
    wandb_logger = pl.loggers.WandbLogger(
        project="nre_ball_deterministic",
        name=f"{featuredim}_{max_epochs}_{batch_size}_{lr}_{date_stamp}_{time_stamp}",
        config={
            "featuredim": featuredim,
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "lr": lr,
        },
    )
    logging.info(
        f"training sync NRE model with max_epochs: {max_epochs}, batch_size: {batch_size}, lr: {lr}"
    )

    ##############################3
    # configure checkpoint
    ##############################3
    ckpt_dir = os.path.join(res_dir_prefix, "checkpoints/")

    ckpt_valacc_name = f"valacc_{featuredim}_{max_epochs}_{batch_size}_{lr}"
    ckpt_valacc = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=ckpt_valacc_name,
        monitor="val_acc",
        mode="max",
        enable_version_counter=False,
        # save_on_train_epoch_end=True, # If this is False, then the check runs at the end of the validation.
    )

    ckpt_last_iter_name = f"last_{featuredim}_{max_epochs}_{batch_size}_{lr}"
    ckpt_last_iter = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=ckpt_last_iter_name,
        enable_version_counter=False,
    )

    model = NREHermans(
        nclass=11,
        nrs_model=NRSimageencoder,
        nrsmodel_hparams={
            "featuredim": featuredim,
            "numclass": 11,
        },
        optimizer=optim.Adam,
        lr=lr,
    )

    ##############################3
    # training
    ##############################3
    #  working path of the file (have to run this file in the root directory of the repo)
    wp_dir = "example/ball/"
    data_dir = os.path.join(wp_dir, "data/obs/dat_all.npy")
    label_dir = os.path.join(wp_dir, "data/obs/label_all.npy")
    datamodule = Balldatamodule(
        data_dir=data_dir, label_dir=label_dir, batch_size=batch_size, num_workers=8
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=1,
        logger=wandb_logger,
        log_every_n_steps=10,
        callbacks=[ckpt_valacc, ckpt_last_iter],
        deterministic=True,
    )
    trainer.fit(model, datamodule=datamodule)
    logging.info("training completed")

    # compute kl metric using just trained nre
    logging.info("computing kl est using tarined logdensity ratio")
    model.freeze()
    model.eval()  # this is going to be trained model of the last iter

    ################################
    # load trained model from the checkpoint
    ################################
    model_acc_val = NREHermans.load_from_checkpoint(
        os.path.join(ckpt_dir, ckpt_valacc_name + ".ckpt")
    )
    model_acc_val.eval()
    model_acc_val.freeze()
    # model_acc_val.to("cpu")

    model_last_iter = NREHermans.load_from_checkpoint(
        os.path.join(ckpt_dir, ckpt_last_iter_name + ".ckpt")
    )
    model_last_iter.eval()
    model_last_iter.freeze()
    # model_last_iter.to("cpu")

    ###########################
    # log the result into wandb
    ###########################
    sync_dat = Balldata()
    sync_dat.data = sync_dat.data.to("cuda")
    kl_obj = kl_ball_est(dataloader=sync_dat)

    kl_df_valacc, kl_mat_valacc = wandb_log_after_train(
        model_acc_val,
        kl_obj,
        wandb_logger,
        table_key="kl_test_accval",
        image_key="kl_mat_accval",
        res_dir_prefix=res_dir_prefix,
        csv_name="kl_nre_est_valacc.csv",
        mat_name="nre_reward_mat_valacc.npy",
        featuredim=featuredim,
        max_epochs=max_epochs,
        batch_size=batch_size,
        lr=lr,
    )
    for tau in tau_inference:
        _, _ = wandb_log_after_train_smile(
            model_acc_val,
            kl_obj,
            wandb_logger,
            table_key=f"kl_smile_valacc_{tau}",
            image_key=f"kl_mat_smile_valacc_{tau}",
            res_dir_prefix=res_dir_prefix,
            csv_name=f"kl_smile_valacc_{tau}.csv",
            mat_name=f"smile_reward_mat_valacc_{tau}.npy",
            featuredim=featuredim,
            max_epochs=max_epochs,
            batch_size=batch_size,
            lr=lr,
            tau=tau,
        )

    kl_df_last_iter, kl_mat_last_iter = wandb_log_after_train(
        model_last_iter,
        kl_obj,
        wandb_logger,
        table_key="kl_test_last_iter",
        image_key="kl_mat_last_iter",
        res_dir_prefix=res_dir_prefix,
        csv_name="kl_nre_est_last_iter.csv",
        mat_name="nre_reward_mat_last_iter.npy",
        featuredim=featuredim,
        max_epochs=max_epochs,
        batch_size=batch_size,
        lr=lr,
    )

    for tau in tau_inference:
        _, _ = wandb_log_after_train_smile(
            model_last_iter,
            kl_obj,
            wandb_logger,
            table_key=f"kl_smile_last_{tau}",
            image_key=f"kl_mat_smile_last_{tau}",
            res_dir_prefix=res_dir_prefix,
            csv_name=f"kl_smile_last_iter_{tau}.csv",
            mat_name=f"smile_reward_mat_last_iter_{tau}.npy",
            featuredim=featuredim,
            max_epochs=max_epochs,
            batch_size=batch_size,
            lr=lr,
            tau=tau,
        )

    # complete wandb
    wandb.finish()


def run_single(
    featuredim=128, max_epochs=1000, batch_size=128, lr=1e-3, tau_inference=[1, 5]
):
    res_dir_prefix = "example/ball/result/nre/"
    if not os.path.exists(res_dir_prefix):
        os.makedirs(res_dir_prefix)
    train_nre(featuredim, max_epochs, batch_size, lr, res_dir_prefix, tau_inference)


def submit_sweep():
    ##################################
    # setup slurm executor
    ##################################
    date_stamp = datetime.now().strftime("%Y-%m-%d")
    time_stamp = datetime.now().strftime("%H-%M-%S")

    res_dir_prefix = "example/ball/result/nre/"
    if not os.path.exists(res_dir_prefix):
        os.makedirs(res_dir_prefix)
    executor = submitit.AutoExecutor(
        folder=res_dir_prefix + f"outputs/{date_stamp}/{time_stamp}"
    )
    executor.update_parameters(
        slurm_array_parallelism=100,  # maximal number of jobs running in parallel
        mem_gb=80,
        timeout_min=60 * 12,  # 12hs
        name="nre_ball",
        gpus_per_node=1,
        cpus_per_task=8,
        nodes=1,
        tasks_per_node=1,
        # slurm_mem_per_cpu=1000
    )
    ##################################
    # submit jobs
    ##################################
    fd_all = [64, 128, 256]
    max_eps_all = [50, 100, 200, 500, 1000]
    bs_all = [512, 1024, 2048]
    lr_all = [2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 1e-5, 2e-5, 5e-5]
    jobs = []
    args = [
        (fd, ep, bs, lr)
        for fd in fd_all
        for ep in max_eps_all
        for bs in bs_all
        for lr in lr_all
    ]
    with executor.batch():
        for arg in args:
            fd, ep, bs, lr = arg
            job = executor.submit(
                run_single,
                featuredim=fd,
                max_epochs=ep,
                batch_size=bs,
                lr=lr,
                tau_inference=[1, 5, 10, 20, 50, 100, 500],
            )
            jobs.append(job)


if __name__ == "__main__":
    submit_sweep() # best training acc: 128, 200, 1024, 5e-5, smile val_acc tau = 5
