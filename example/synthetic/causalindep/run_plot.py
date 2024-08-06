import numpy as np
import matplotlib.pyplot as plt

from example.utils.plotting import plot_matrix



#
labels = ["A", "B", "C", "D"]
rwd_mat = np.load("example/synthetic/causalindep/result/nre/1000_512_0.002nre_reward_mat.npy")
fig, ax = plot_matrix(rwd_mat, labels, title="Separability score for tabular (NRE)", cbar_label = "", axisfontsize=35, titlefontsize=40)
plt.savefig("example/synthetic/causalindep/figure/kl_nre_tab.pdf")


rwd_mat = np.load("example/synthetic/causalindep/result/knn/knn_reward_mat.npy")
fig, ax = plot_matrix(rwd_mat, labels, title="Separabiltiy score for tabular (KNN)", cbar_label = "", axisfontsize=35, titlefontsize=40)
plt.savefig("example/synthetic/causalindep/figure/kl_knn_tab.pdf")



rwd_mat = np.load("example/synthetic/causalindep/result/nre/1000_512_0.002nre_reward_mat.npy")
fig, ax = plot_matrix(rwd_mat, labels, title="KL score for sync. tabular (NRE)", cbar_label = "", axisfontsize=35, titlefontsize=40)
plt.savefig("example/synthetic/causalindep/figure/kl_nre_tab_rebut.pdf")


rwd_mat = np.load("example/synthetic/causalindep/result/knn/knn_reward_mat.npy")
fig, ax = plot_matrix(rwd_mat, labels, title="KL score for sync. tabular (KNN)", cbar_label = "", axisfontsize=35, titlefontsize=40)
plt.savefig("example/synthetic/causalindep/figure/kl_knn_tab_rebut.pdf")
