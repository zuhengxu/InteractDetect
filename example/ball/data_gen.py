############################################
# The following code is adapted from https://github.com/facebookresearch/CausalRepID/blob/main/data/balls_dataset.py
############################################

import os
from typing import Callable, Optional

import numpy as np
import pygame
import torch

from torch import distributions
from pygame import gfxdraw
from matplotlib import pyplot as plt
from example.ball.datamodule import LabelMap


if "SDL_VIDEODRIVER" not in os.environ:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

COLOURS_ = [
    [2, 156, 154],
    [222, 100, 100],
    [149, 59, 123],
    [74, 114, 179],
    [27, 159, 119],
    [218, 95, 2],
    [117, 112, 180],
    [232, 41, 139],
    [102, 167, 30],
    [231, 172, 2],
    [167, 118, 29],
    [102, 102, 102],
]
SCREEN_DIM = 64


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


#########################################
# DAG model for generating locations
#######################################
class LocationDAG:
    def __init__(self, seed=None):
        self.P = [
            distributions.Normal(loc=-4, scale=2),
            distributions.Normal(loc=5, scale=2),
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


















# we dont need square for this experiment
def circle(
    x_,  # x coordinate of the ball
    y_,  # y coordinate of the ball
    surf,
    color=(204, 204, 0),
    radius=0.02,
    screen_width=SCREEN_DIM,
    y_shift=0.0,
    offset=None,
):
    if offset is None:
        offset = screen_width / 2
    scale = screen_width
    x = scale * x_ + offset
    y = scale * y_ + offset
    gfxdraw.aacircle(
        surf, int(x), int(y - offset * y_shift), int(radius * scale), color
    )
    gfxdraw.filled_circle(
        surf, int(x), int(y - offset * y_shift), int(radius * scale), color
    )


class Balls(torch.utils.data.Dataset):
    ball_rad = 0.02
    screen_dim = 128

    def __init__(
        self,
        transform: Optional[Callable] = None,
        n_balls: int = 1,
        use_noise: bool = False,
    ):
        super(Balls, self).__init__()
        if transform is None:

            def transform(x):
                return x

        self.transform = transform
        pygame.init()
        self.screen1 = pygame.display.set_mode((self.screen_dim, self.screen_dim))
        self.surf1 = pygame.Surface((self.screen_dim, self.screen_dim))
        self.screen2 = pygame.display.set_mode((self.screen_dim, self.screen_dim))
        self.surf2 = pygame.Surface((self.screen_dim, self.screen_dim))
        self.n_balls = n_balls
        self.use_noise = use_noise

    def __len__(self) -> int:
        # arbitrary since examples are generated online.
        return 20000

    def draw_scene1(self, z):
        cmap_bg1 = plt.get_cmap("summer")
        cmap_ball = plt.get_cmap("cool")
        self.surf1.fill((255, 255, 255))
        if z.ndim == 1:
            z = z.reshape((1, 2))
        for i in range(25):
            c = tuple(
                np.array(np.array(cmap_bg1(np.random.rand())) * 255, dtype=int)[:3]
            )
            circle(
                np.random.rand(),
                np.random.rand(),
                self.surf1,
                color=c,
                radius=0.3,
                screen_width=self.screen_dim,
                y_shift=0.0,
                offset=0.0,
            )

        # import pdb; pdb.set_trace()
        ball_colors = [0.1, 0.5, 0.9]
        for i in range(z.shape[0]):
            # color = np.random.rand()
            color = ball_colors[i]
            c = tuple(np.array(np.array(cmap_ball(color))*255 , dtype=int)[:3])
            circle(
                z[i, 0],
                z[i, 1],
                self.surf1,
                color=c,
                radius=self.ball_rad,
                screen_width=self.screen_dim,
                y_shift=0.0,
                offset=0.0,
            )
        self.surf1 = pygame.transform.flip(self.surf1, False, True)
        self.screen1.blit(self.surf1, (0, 0))
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen1)), axes=(1, 0, 2)
        )

    def __getitem__(self, item):
        raise NotImplemented()



class BlockOffset(Balls):
    def __init__(
        self,
        transform: Optional[Callable] = None,
        n_balls: int = 3, # 3 balls because we are going to use the same latent space as the synthetic dag
        use_noise: bool = False,
    ):
        super().__init__(transform=transform, n_balls=n_balls, use_noise=use_noise)
        # self.dataset_size = 20000 # whatever---it will be generated online
        # self.latent_dim = (
        #     self.n_balls * 2
        # )  # location (z1, z2) of each ball as the latent
        self.latent_dag = LocationDAG()
        self.labelmap = LabelMap([0, 1, 2, 3])
        self.labels = self.labelmap.labels
        self.perts = self.labelmap.all_class

    def label_flatten(self, perturbation):
        return self.labelmap.get_flatten_idx(perturbation)

    def label_unflatten(self, label):
        return self.labelmap.get_unflatten_idx(label)

    # just to generate the latent space, which is the location of the balls
    def generate_locations(self, label):
        # map class label to pair/single perturbation
        if label == 0:
            perturbation = None
        else:
            perturbation = self.label_unflatten(label)

        # what ever data generating process for z, and make sure z is in the range of [0.1, 0.9]
        # just do the same as synthetic and apply sigmoid transformation to it.
        _z = self.latent_dag.sample(num_samples=2, intervene=perturbation).t()
        z = torch.sigmoid(_z)*0.8 + 0.1 # transform to [0.1, 0.9]
        return z.numpy()


    def single_image_gen(self, label):
        # shape: (n_balls, 2)
        z = self.generate_locations(label) 

        # this is the g: z -> image
        _x = self.draw_scene1(z)
        x = self.transform(_x)
        return x, label, z.flatten()

    def image_gen(self, label, num_samples):
        xs = []
        ys = []
        zs = []

        for i in range(num_samples):
            x, y, z = self.single_image_gen(label)
            x = np.expand_dims(x, axis=0)
            y = np.expand_dims(y, axis=0)
            z = np.expand_dims(z, axis=0)
            
            xs.append(x)
            ys.append(y)
            zs.append(z)

        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), np.concatenate(zs, axis=0)

    def image_gen_all(self, nsample_eachclass):
        xs = []
        ys = []
        zs = []

        for c in self.labels:
            x, y, z = self.image_gen(c, nsample_eachclass)
            xs.append(x)
            ys.append(y)
            zs.append(z)

        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), np.concatenate(zs, axis=0)


if __name__ == "__main__":
    # generate the data
    # data_dir = "data/"
    obs_dir = "example/ball/data/obs/"
    latent_dir = "example/ball/data/latent/"
    data_dirs = [obs_dir, latent_dir]
    for pth in data_dirs:
        if not os.path.exists(pth):
            os.mkdir(pth)


    # generate the data
    nsample_eachclass = 20000
    model = BlockOffset()
    all_class = model.labels

    dat_obs, label_all, dat_lat = model.image_gen_all(nsample_eachclass)
    np.save(os.path.join(obs_dir, "dat_all.npy"), dat_obs)
    np.save(os.path.join(obs_dir, "label_all.npy"), label_all)
    np.save(os.path.join(latent_dir, "dat_all.npy"), dat_lat)
    np.save(os.path.join(latent_dir, "label_all.npy"), label_all)

    # df = np.load(os.path.join(obs_dir, "dat_all.npy"))
    # lab = np.load(os.path.join(latent_dir, "label_all.npy"))
