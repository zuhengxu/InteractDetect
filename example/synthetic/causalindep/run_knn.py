import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], "../../../"))
import numpy as np
import matplotlib.pyplot as plt

from example.synthetic.causalindep.datamodule import CISyncDataLoader
from example.synthetic.causalindep.kl_est import kl_sync_est
from example.utils.plotting import plot_matrix
from inference.CausalIndep.klestimator.klknn import skl_efficient


result_dir = "result/knn/"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

perturb_list = [str(i) for i in range(4)]
sync_dat = CISyncDataLoader(datadir="data/obs/")
kl_obj = kl_sync_est(pert_list=perturb_list, dataloader=sync_dat)

knn_df = kl_obj.kl_all_pairs(skl_efficient, k=1, standardize=False)
knn_df.to_csv(os.path.join(result_dir, "kl_knn_est.csv"), index=False)

knn_mat = kl_obj.kl_reward_mat()
np.save(os.path.join(result_dir, "knn_reward_mat.npy"), knn_mat)

#########################
# plotting the reward mat
#########################
fig_dir = "figure/knn/"
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
labels = [str(i) for i in range(4)]
fig, ax = plot_matrix(
    knn_mat,
    labels,
    title="KL Reward Matrix (KNN)",
    axisfontsize=15,
)
pth = os.path.join(fig_dir, "knn_mat.pdf")
plt.savefig(pth)
