import torch
from torch import distributions

class DAGmixture:
    def __init__(self):
        # independent ones
        self.P0 = distributions.Normal(loc=0.0, scale=1.0)
        self.P1 = [
            distributions.Normal(loc=5.0, scale=1.0),
        ]
        self.P2 = [
            distributions.Normal(loc=10.0, scale=1.0),
        ]
        self.P3 = [
            distributions.Normal(loc=-5, scale=5.0),
        ]

        # interacting group
        self.P4 = [
            distributions.Normal(loc=0.0, scale=5.0), # D
            distributions.Normal(loc=-10.0, scale=5.0), # E 
            distributions.Normal(loc=10.0, scale=5.0), # DE
        ]
        self.P5 = [
            distributions.Cauchy(loc=15.0, scale=1.0), # F
            distributions.Cauchy(loc=-15.0, scale=1.0), # G 
            distributions.Cauchy(loc=20.0, scale=1.0), # FG
        ]
        self.mixingweights = [1/8, 1/8, 1/8, 1/8, 1/4, 1/4] # mixing weights for each mix component
        self.perturbation_dict = [[], ["A"], ["B"], ["C"], ["D", "E", "DE"], ["F", "G", "FG"]]
        self.component_dist = [self.P0, self.P1, self.P2, self.P3, self.P4, self.P5]

    def search_index(self, intervene):
        """
        Search for the index of the perturbation_dict that contains the intervention
        """
        for i in range(len(self.perturbation_dict)):
            if intervene in self.perturbation_dict[i]:
                return i
        return None

    def _build_mixture(self, intervene):
        # asserts that intervene is not none and is a string
        assert intervene is not None and isinstance(intervene, str)

        index = self.search_index(intervene)

        # only perturbed one of the component dist
        if index:
            Ps = [self.P0, self.component_dist[index][0]]
            ws = [1 - self.mixingweights[index], self.mixingweights[index]]
        else:
            intervene_1, intervene_2 = intervene
            assert intervene_1 is not None and intervene_2 is not None

            index_1 = self.search_index(intervene_1)
            index_2 = self.search_index(intervene_2)

            Ps = [self.P0, self.component_dist[index_1][0], self.component_dist[index_2][0]]

            w0 = 1.0 - self.mixingweights[index_1] - self.mixingweights[index_2]
            w1 = self.mixingweights[index_1]
            w2 = self.mixingweights[index_2]
            ws = [w0, w1, w2]

        return Ps, torch.tensor(ws)

    def latent_sample(self, num_samples, dim, intervene=None):
        if intervene is None:
            samples = self.P0.sample((num_samples, dim))
            return samples
        
        else:
            Ps, ws = self._build_mixture(intervene)
            # sample catorical distribution from ws
            cat_dist = distributions.Categorical(probs=ws)

            # sample from the categorical distribution
            mix_component = cat_dist.sample((num_samples, ))
            # count the number of samples in each component
            counts = mix_component.bincount(minlength=len(ws))

            # sample from each component
            samples = []
            for i in range(len(Ps)):
                samples.append(Ps[i].sample((counts[i], dim)))
                
            samples = torch.cat(samples, dim=0)
            return samples 
    #
    # def obs_sample(self, num_samples, dim, g, intervene=None):
    #     latent_samples = self.latent_sample(num_samples, dim, intervene)
    #     obs_samples = g(latent_samples)
    #     return obs_samples

if __name__ == "__main__":
    # build dataframe that save all data
    import pandas as pd
    import pickle as pkl

    # set seed 
    torch.manual_seed(0)
    genes = ["A", "B", "C", "D", "E", "F", "G"]
    
    m = DAGmixture()
    num_samples = 20000
    dim = 2

    latent_samples = []
    latent_control = m.latent_sample(num_samples, dim)
    latent_samples.append(latent_control)

    # all single perturbations 
    for g in genes:
        lsample = m.latent_sample(num_samples, dim, intervene=g)
        latent_samples.append(lsample)
        
    labels = ["control"] + genes
    
    for i in range(len(genes)):
        for j in range(i+1, len(genes)):
            lsample = m.latent_sample(num_samples, dim, intervene=genes[i]+genes[j])
            latent_samples.append(lsample)
            labels.append(genes[i]+genes[j])

        

    # build dataframe 
    latent_samples = torch.cat(latent_samples, dim=0)
    latent_samples = latent_samples.detach().numpy()

    latent_df = pd.DataFrame(latent_samples, columns=["x_"+str(i) for i in range(dim)])

    # perturbation cols = repeat labels for num_samples times
    perturbation_col = []
    for label in labels:
        perturbation_col += [label]*num_samples
    # for each pd, add a column that indicates the perturbation 
    latent_df["perturbation"] = perturbation_col

    # pickle the dataframes 
    with open("example/synthetic/disjointness/data/latent_df.pkl", "wb") as f:
        pkl.dump(latent_df, f)

