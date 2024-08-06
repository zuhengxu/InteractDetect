import torch
import scipy


# U = torch.tensor(scipy.stats.ortho_group.rvs(2)).to(dtype = torch.float32)
class leaky_map:
    def __init__(self, inputdims, nlayers, slope = 0.5, stretch = 1.5, add_noise = False, noise_level = 0.5):
        self.linear_maps = [torch.tensor(scipy.stats.ortho_group.rvs(inputdims)).to(dtype = torch.float32) for i in range(nlayers)]
        self.slope = slope
        self.stretch = stretch
        self.add_noise = add_noise
        self.noise_level = noise_level
        
    
    def __call__(self, x):
        for U in self.linear_maps:
            x = torch.matmul(x, U)
            x = self.stretch*torch.nn.LeakyReLU(self.slope)(x)
            if self.add_noise:
                x += self.noise_level*torch.randn_like(x)
        return x

def warp(z):
    x, y = z[:, 0], z[:, 1]
    r = torch.norm(z, dim=-1, keepdim=False)

    theta = torch.atan2(y, x)
    theta -= r / 2

    new_x = r * torch.cos(theta)
    new_y = r * torch.sin(theta)

    new_x = new_x.unsqueeze(-1)
    new_y = new_y.unsqueeze(-1)
    z_transformed = torch.cat((new_x, new_y), dim = -1)
    return z_transformed

class warp_map:
    def __init__(self, inputdims, nlayers, stretch = torch.tensor([1.0, 0.12]), add_noise = False, noise_level = 0.5):
        self.linear_maps = [torch.tensor(scipy.stats.ortho_group.rvs(inputdims)).to(dtype = torch.float32) for i in range(nlayers)]
        self.stretch = stretch
        self.add_noise = add_noise
        self.noise_level = noise_level
        
    
    def __call__(self, x):
        for U in self.linear_maps:
            x = torch.matmul(x, U) * self.stretch
            x = warp(x)
            if self.add_noise:
                x += self.noise_level*torch.randn_like(x)
        return x

