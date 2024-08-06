import os
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def plot_matrix(
    r2_matrix,
    labels,
    cbar_label="",
    cmap="viridis",
    title="",
    axisfontsize=6,
    titlefontsize=20,
    **kwargs,
):
    fig, ax = plt.subplots(figsize=(15, 15))
    # get perceptually uniform colour palette
    cmap = sns.color_palette(cmap, as_cmap=True)
    # plot_matrix = r2_matrix.toarray()
    r2_matrix[r2_matrix == 0] = np.nan
    cax = ax.matshow(r2_matrix, cmap=cmap, **kwargs)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ngs = r2_matrix.shape[0]
    ax.set_xticklabels(
        [f"{labels[i]}" for i in range(ngs)], rotation=90, fontsize=axisfontsize
    )
    ax.set_yticklabels([f"{labels[i]}" for i in range(ngs)], fontsize=axisfontsize)
    ax.set_title(title, fontsize=titlefontsize)
    fig.colorbar(cax, label=cbar_label, shrink=0.8)
    return fig, ax



# turn mat_selected into a dataframe with row and column labels as labels
def save_df_csv_cb(df: pd.DataFrame, res_dir: str, df_name: str = "df_name.csv"):
    df.to_csv(os.path.join(res_dir, df_name))


def save_fig_cb(fig, ax, fig_dir: str, fig_name: str = "fig_name.pdf"):
    print(f"Saving figure to {fig_dir}")
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    print(fig_name)
    plt.savefig(os.path.join(fig_dir, fig_name))
    plt.close()

def wandb_plot_log_cb(fig, ax, wandb_logger, image_key: str = "fig"):
    wandb_logger.log_image(key=image_key, images=[fig])


def plot_mat_as_df(
    mat,
    labels,
    cols=["pert1", "pert2", "kldiff"],
    figsize=(10, 8),
    fig_kargs={},
    save_df_cb=None,
    df_cb_kargs={
        "res_dir": "example/single_cell/result/nre/",
        "df_name": "selected_heatmap_df.csv",
    },
    save_fig_cb=None,
    fig_cb_kargs={
        "fig_dir": "example/single_cell/result/nre/",
        "fig_name": "nremat.pdf",
    },
    wandb_plot_log_cb=None, 
    wandb_kwargs = {},
):
    df = pd.DataFrame(mat, columns=labels, index=labels)

    # flatten the df and keep only non-nan values
    df_flat = df.stack().reset_index()
    df_flat.columns = cols

    # unflatten the df_flat for heatmap plotting
    df_unflat = df_flat.pivot(index=cols[0], columns=cols[1], values=cols[2])

    # plot the heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        df_unflat, annot=True, cmap="coolwarm", fmt=".2f", cbar=True, ax=ax, **fig_kargs
    )
    plt.tight_layout()

    # callbacks to save the df and fig if needed
    if save_fig_cb is not None:
        save_fig_cb(fig, ax, **fig_cb_kargs)
    if save_df_cb is not None:
        save_df_cb(df, **df_cb_kargs)
    if wandb_plot_log_cb is not None:
        wandb_plot_log_cb(fig, ax, **wandb_kwargs)

    return (fig, ax), (df, df_flat, df_unflat)


def plot_mat_df_flat(
    df_flat, 
    figsize=(10, 8),
    fig_kargs={},
    save_fig_cb=None,
    fig_cb_kargs={
        "fig_dir": "example/single_cell/result/nre/fig",
        "fig_name": "nremat.pdf",
    },
    wandb_plot_log_cb=None, 
    wandb_kwargs = {},
):
    cols = df_flat.columns

    # unflatten the df_flat for heatmap plotting
    df_unflat = df_flat.pivot(index=cols[0], columns=cols[1], values=cols[2])

    # plot the heatmap
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap( df_unflat, annot=True, cmap="coolwarm", fmt=".2f", cbar=True, ax=ax, **fig_kargs)
    sns.set(font_scale=1.5)
    plt.tight_layout()
    
    # callbacks to save the df and fig if needed
    if save_fig_cb is not None:
        save_fig_cb(fig, ax, **fig_cb_kargs)
    if wandb_plot_log_cb is not None:
        wandb_plot_log_cb(fig, ax, **wandb_kwargs)
    return fig, ax

