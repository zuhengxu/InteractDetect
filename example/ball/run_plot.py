import numpy as np
import matplotlib.pyplot as plt

from example.utils.plotting import plot_matrix


# load label
labels = ["A", "B", "C", "D"]

rwd_mat = np.load("example/ball/result/nre/128_200_1024_5e-05smile_reward_mat_valacc_5.npy")
fig, ax = plot_matrix(rwd_mat, labels, title="Separability score for image", cbar_label = "", axisfontsize=35, titlefontsize=40)
plt.savefig("example/ball/figure/kl_nre_ball.pdf")
