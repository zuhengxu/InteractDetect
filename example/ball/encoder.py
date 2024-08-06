import torch
from torch import nn
from inference.CausalIndep.klestimator.neuralratio import CondNeuralRatioScore, LogDensityRatioEstimator

class ImageEncoder(torch.nn.Module):
    def __init__(self, latent_dim):
        super(ImageEncoder, self).__init__()        
        
        self.latent_dim = latent_dim

        ## Image encoder from https://github.com/uhlerlab/cross-modal-autoencoders
        modules = []
        hidden_dims = [32, 64, 128, 256, 512]
        in_channels = 3
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.feat_net = nn.Sequential(*modules)
        self.fc_net = nn.Linear(hidden_dims[-1] * 4 * 4, latent_dim)
        

        
    def forward(self, x):
        x= self.feat_net(x)
        x = torch.flatten(x, start_dim=1)
        x= self.fc_net(x)
        return x



# define the neural ratio estimator with mlp encoder
class NRSimageencoder(CondNeuralRatioScore):
    """
    NRS using a image encoder as the feature layer NN(x)

    For ImageEncoder:
    -----------------------------------------------------------
    input shape: (bs, num of channel, image_hight, image_width), here num of channel = 3
    output shape: (bs, featuredim)


    Example:
    ----------------------------
    Stest = NRSimageencoder(256, 3)
    x = torch.randn(bs, 3, 128, 128)
    label = torch.LongTensor([1, 1, 2, 0, 0])
    Stest(x, label) # compute logp(x|c=label)
    Stest.score_fix_label(x, [1]) # compute log p(x|c=1)/p(x)
    Stest.logdensityratio(x, [1], [0]) # compute log p(x|c=1)/p(x|c= 0)
    """

    def __init__(self, featuredim, numclass):
        encoder = ImageEncoder(featuredim)
        super().__init__(encoder, featuredim, numclass)

class ldr_image_encoder(LogDensityRatioEstimator):
    """
    Example:
    ---------
    model = NREHermans.load_from_checkpoint(ckpt_file)
    nre = model.score
    T = ldr_image_encoder(nre, [1])
    """

    def __init__(self, nre: NRSimageencoder, pert_label):
        super().__init__(nre)
        self.pert_label = [pert_label]

    def ldr(self, x):
        return self.nre.logdensityratio(x, [0], self.pert_label)

    def forward(self, x):
        return self.ldr(x)

