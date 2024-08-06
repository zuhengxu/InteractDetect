import warnings
import numpy as np
from sklearn.neighbors import NearestNeighbors

#################################################################################################
# copy from https://github.com/nhartland/KL-divergence-estimators/blob/master/src/knn_divergence.py
#################################################################################################
""" KL-Divergence estimation through K-Nearest Neighbours

    This module provides four implementations of the K-NN divergence estimator of
        Qing Wang, Sanjeev R. Kulkarni, and Sergio Verdú.
        "Divergence estimation for multidimensional densities via
        k-nearest-neighbor distances." Information Theory, IEEE Transactions on
        55.5 (2009): 2392-2405.

    The implementations are through:
        scikit-learn (skl_estimator / skl_efficient)

    No guarantees are made w.r.t the efficiency of these implementations.

    # usage to est KL(p||q)
    # both X_p and X_q are numpy arrays of shape (n, d)

    import torch
    X_p = torch.randn(1000, 2)
    X_q = torch.randn(1000, 2)
    kl_pq = skl_efficient(X_p, X_q, k=1)
"""

def skl_efficient(s1, s2, k=1, standardize=False):
    """An efficient version of the scikit-learn estimator by @LoryPack
    s1: (N_1,D) Sample drawn from distribution P
    s2: (N_2,D) Sample drawn from distribution Q
    k: Number of neighbours considered (default 1)
    return: estimated D(P|Q)

    Contributed by Lorenzo Pacchiardi (@LoryPack)
    """
    verify_sample_shapes(s1, s2, k)

    n, m = len(s1), len(s2)
    d = float(s1.shape[1])

    if standardize:
        means1, stds1 = s1.mean(axis=0), s1.std(axis=0)
        s1 = (s1 - means1) / stds1
        s2 = (s2 - means1) / stds1

    s1_neighbourhood = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree").fit(s1)
    s2_neighbourhood = NearestNeighbors(n_neighbors=k, algorithm="kd_tree").fit(s2)

    s1_distances, indices = s1_neighbourhood.kneighbors(s1, k + 1)
    s2_distances, indices = s2_neighbourhood.kneighbors(s1, k)
    rho = s1_distances[:, -1]
    nu = s2_distances[:, -1]
    if np.any(rho == 0):
        warnings.warn(
            f"The distance between an element of the first dataset and its {k}-th NN in the same dataset "
            f"is 0; this causes divergences in the code, and it is due to elements which are repeated "
            f"{k + 1} times in the first dataset. Increasing the value of k usually solves this.",
            RuntimeWarning,
        )
    D = np.sum(np.log(nu / rho))

    return (d / n) * D + np.log(
        m / (n - 1)
    )  # this second term should be enough for it to be valid for m \neq n

def verify_sample_shapes(s1, s2, k):
    # Expects [N, D]
    assert len(s1.shape) == len(s2.shape) == 2
    # Check dimensionality of sample is identical
    assert s1.shape[1] == s2.shape[1]
