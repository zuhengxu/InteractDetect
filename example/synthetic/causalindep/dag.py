import numpy as np
import torch
from torch import distributions

class DAGModel:
    def __init__(self, seed=None):
        self.P = [
            distributions.Normal(loc=0, scale=1),
            distributions.Normal(loc=3, scale=1),
        ]
        self.adjacency_matrix = torch.tensor( # parents x children (T x Z)
            [
                [1, 0, 0], 
                [1, 1, 0], 
                [0, 0, 1],
                [0, 0, 1]
            ])
        self.num_parents = self.adjacency_matrix.shape[0]
        self.num_children = self.adjacency_matrix.shape[1]
        self.seed = seed

    def sample(self, num_samples, intervene=None):
        if self.seed is not None:
            torch.manual_seed(self.seed)

        intervene = None if intervene == "control" else intervene

        if intervene is None: 
            for j in range(self.num_children):
                samples = self.P[0].sample((num_samples,self.num_children))
                return samples

        samples = torch.zeros((num_samples, self.num_children))
        print(f'Intervening on {intervene}')
        for j in range(self.num_children):
            parents = set(np.where(self.adjacency_matrix[:, j] == 1)[0])
            if parents & intervene != set():
                print(f'Changed dist of z{j} since parents {parents} are in intervene set')
                samples[:, j] = self.P[1].sample((num_samples,))
            else:
                samples[:, j] = self.P[0].sample((num_samples,))
        
        return samples

    def log_density_z(self, z, intervene=None):
        intervene = None if intervene == "control" else intervene

        if intervene is None:
            return self.P[0].log_prob(z).sum(dim=1)
        log_prob = torch.zeros(z.shape[0])
        print(f'Intervening on {intervene}')
        for j in range(self.num_children):
            parents = set(np.where(self.adjacency_matrix[:, j] == 1)[0])
            if parents & intervene != set():
                print(f'Intervening on z{j} since parents {parents} are in intervene set')
                log_prob += self.P[1].log_prob(z[:, j])
            else:
                log_prob += self.P[0].log_prob(z[:, j])
        return log_prob


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(1, "../../../")  # load the repo root directory
    import numpy as np
    from example.utils.transform import leaky_map

    # 5 layer leaky ReLU map with random orthogonal matrices
    g = leaky_map(inputdims=3, nlayers=5, slope=0.25, stretch=2.0, add_noise=False)

    # numpy set seed
    np.random.seed(0)

    # make directory for latent data
    latent_dir = "data/latent/"
    obs_dir = "data/obs/"
    for dir in [latent_dir, obs_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    # generate latent data
    model = DAGModel()
    num_samples = 30000

    # control group
    dat_control = np.array(model.sample(num_samples=num_samples, intervene=None))
    np.save(latent_dir + "control.npy", dat_control)

    obs_control = g(torch.tensor(dat_control, dtype=torch.float32)).detach().numpy()
    np.save(obs_dir + "control.npy", obs_control)

    # single gene perturbation (["A", "B", "C", "D", "E", "F"])
    gene_list = [0, 1, 2, 3]
    for i in range(len(gene_list)):
        gene = gene_list[i]
        latent_single = model.sample(num_samples=num_samples, intervene={gene})
        np.save(latent_dir + f"{gene}.npy", latent_single)

        obs_single = (
            g(torch.tensor(latent_single, dtype=torch.float32)).detach().numpy()
        )
        np.save(obs_dir + f"{gene}.npy", obs_single)

    # double gene perturbation (["A", "B", "C", "D", "E", "F"])
    for i in range(len(gene_list)):
        for j in range(i + 1, len(gene_list)):
            # pert_double.append(gene_list[i] + gene_list[j])
            perts = [str(gene_list[i]), str(gene_list[j])]
            name = "_".join(perts)
            dat_double = model.sample(
                num_samples=num_samples, intervene={gene_list[i], gene_list[j]}
            )
            # dat_double_set.append(dat_double)
            np.save(latent_dir + f"{name}.npy", dat_double)

            obs_double = (
                g(torch.tensor(dat_double, dtype=torch.float32)).detach().numpy()
            )
            np.save(obs_dir + f"{name}.npy", obs_double)
