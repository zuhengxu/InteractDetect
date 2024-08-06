import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import pickle

import gpytorch
import pandas as pd
import torch
from example.utils.extract import extract_feature

from inference.TwoSample.mmd import MMD_square


genes = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
def MMD_reward_compute(df, kernel, bw=1.52):
    x0 = extract_feature(df.loc[df["perturbation"] == "control"], "x")
    mmd_mat = torch.zeros(len(genes), len(genes)).to("cuda")
    for i in range(len(genes)):
        for j in range(i+1, len(genes)):
            xi = df.loc[df["perturbation"] == genes[i]]
            xj = df.loc[df["perturbation"] == genes[j]]
            xij = df.loc[df["perturbation"] == genes[i]+genes[j]]

            xi = extract_feature(xi, "x")
            xj = extract_feature(xj, "x")
            xij = extract_feature(xij, "x")

            xx = torch.cat([x0, xij],0).to("cuda")
            yy = torch.cat([xi, xj],0).to("cuda")

            bw = float(torch.median(torch.cdist(xx, yy)))
            print(f"Bandwidth: {bw}")
            with torch.no_grad():
                # kernel = gpytorch.kernels.MaternKernel(nu=2.5).to("cuda")
                kernel.lengthscale = bw
                mmd = MMD_square(xx, yy, kernel)

            mmd_mat[i,j] = mmd
            torch.cuda.empty_cache()
            print(f"MMD between {genes[i]} and {genes[j]}: {mmd}")

    return mmd_mat

def MMD_rwd_on_transform(df, transform, kernel, bw = 0.5):
    x0 = extract_feature(df.loc[df["perturbation"] == "control"], "x")
    mmd_mat = torch.zeros(len(genes), len(genes)).to("cuda")
    for i in range(len(genes)):
        for j in range(i+1, len(genes)):
            xi = df.loc[df["perturbation"] == genes[i]]
            xj = df.loc[df["perturbation"] == genes[j]]
            xij = df.loc[df["perturbation"] == genes[i]+genes[j]]

            xi = extract_feature(xi, "x")
            xj = extract_feature(xj, "x")
            xij = extract_feature(xij, "x")

            _xx = torch.cat([x0, xij],0)
            _yy = torch.cat([xi, xj],0)

            xx = transform(_xx).to("cuda")
            yy = transform(_yy).to("cuda")

            bw = float(torch.median(torch.cdist(xx, yy))) print(f"Bandwidth: {bw}")
            with torch.no_grad():
                # kernel = gpytorch.kernels.MaternKernel(nu=2.5).to("cuda")
                kernel.lengthscale = bw
                mmd = MMD_square(xx, yy, kernel)

            mmd_mat[i,j] = mmd
            torch.cuda.empty_cache()
            print(f"MMD between {genes[i]} and {genes[j]}: {mmd}")

    return mmd_mat


def l2d_on_emd(df, emd, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    x0 = extract_feature(df.loc[df["perturbation"] == "control"], "x").to(device)
    rwd_mat = torch.zeros(len(genes), len(genes)).to(device)
    for i in range(len(genes)):
        for j in range(i+1, len(genes)):
            xi = df.loc[df["perturbation"] == genes[i]]
            xj = df.loc[df["perturbation"] == genes[j]]
            xij = df.loc[df["perturbation"] == genes[i]+genes[j]]

            xi = extract_feature(xi, "x")
            xj = extract_feature(xj, "x")
            xij = extract_feature(xij, "x")

            # xx = torch.cat([x0, xij],0).to(device)
            # yy = torch.cat([xi, xj],0).to(device)

            e0 = emd(x0).mean(dim = 0)
            ei = emd(xi).mean(dim = 0)
            ej = emd(xj).mean(dim = 0)
            eij = emd(xij).mean(dim = 0)

            l2d = torch.norm(e0 + eij - ei - ej) 
            rwd_mat[i,j] = l2d
            print(f"l2 feature dist. between {genes[i]} and {genes[j]}: {l2d}")

    return rwd_mat


if __name__ == "__main__":

    from example.utils.transform import leaky_map

    maternkernel = gpytorch.kernels.MaternKernel(nu=2.5).to("cuda")
    rbfkernel = gpytorch.kernels.RBFKernel().to("cuda")

    df_l = pd.read_pickle('example/synthetic/disjointness/data/latent_df.pkl')
    mmd_mat_latent = MMD_reward_compute(df_l, maternkernel, bw=2.0)
    with open("example/synthetic/disjointness/result/mmd_mat_latent.pkl", "wb") as f:
        pickle.dump(mmd_mat_latent.cpu().numpy(), f)

    g = leaky_map(2, 10, slope = 0.25, stretch = 2.0, add_noise = True, noise_level = 10)
    mmd_mat_leaky = MMD_rwd_on_transform(df_l, g, maternkernel, bw=2.0)
    with open("example/synthetic/disjointness/result/mmd_mat_obs_leaky.pkl", "wb") as f:
        pickle.dump(mmd_mat_leaky.cpu().numpy(), f)


    g_noise = leaky_map(2, 10, slope = 0.25, stretch = 2.0, add_noise = True, noise_level = 20)
    mmd_mat_leaky_noise = MMD_rwd_on_transform(df_l, g_noise, maternkernel, bw=2.0)
    with open("example/synthetic/disjointness/result/mmd_mat_obs_leaky_noise.pkl", "wb") as f:
        pickle.dump(mmd_mat_leaky_noise.cpu().numpy(), f)



