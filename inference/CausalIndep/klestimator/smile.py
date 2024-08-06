import pytorch_lightning as pl
import torch
import math

from inference.CausalIndep.klestimator.neuralratio import LogDensityRatioEstimator


# implement customized grad estimate for MINE
# using EMA for the exponnenetial moving average for the denominator to reduce bias
# https://github.com/gtegner/mine-pytorch/blob/master/mine/models/mine.py

EPS=1e-9
# define the expoennetial moving averaged grad estimator for the logsumexp
class EMALoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, running_ema):
        ctx.save_for_backward(input, running_ema)
        input_log_sum_exp = input.exp().mean().log()

        return input_log_sum_exp

    @staticmethod
    def backward(ctx, grad_output):
        input, running_mean = ctx.saved_tensors
        grad = grad_output * input.exp().detach() / \
            (running_mean + EPS) / input.shape[0]
        return grad, None


def ema(mu, alpha, past_ema):
    return alpha * mu + (1.0 - alpha) * past_ema


def ema_loss(x, running_mean, alpha):
    t_exp = torch.exp(torch.logsumexp(x, 0) - torch.log(x.shape[0])).detach()
    if running_mean == 0:
        running_mean = t_exp
    else:
        running_mean = ema(t_exp, alpha, running_mean.item())
    t_log = EMALoss.apply(x, running_mean)

    # Recalculate ema
    return t_log, running_mean


def smile_kl_est(T, xp, xq, tau=10):
    """
    This est KL(p||q) using Eq.(17) from https://arxiv.org/pdf/1910.06222
    KL(p||q) = sup_T E_p[T] - log(E_q[exp(T)])
    """
    term1 = T(xp).mean()
    qs = torch.clamp(T(xq), min = -tau, max = tau)
    term2 = torch.logsumexp(qs, dim=0) - math.log(qs.shape[0])
    return term1 - term2

class SMILE(pl.LightningModule):
    """
    SMILE computation for KL estimation KL(P||Q)
    check eq (17) of https://arxiv.org/pdf/1910.06222
    T in the paper will be the learned logdensity ratio log p(x)/q(x)
    which is going to be a neural network
    """
    def __init__(
        self, 
        neural_ldr:LogDensityRatioEstimator,
        clip_thres = 10, 
        optimizer=torch.optim.Adam,
        lr=1e-3,
        opt_kwargs=None,
    ):
        super().__init__()
        self.T = neural_ldr
        self.lr = lr
        self.tau = clip_thres
        self._optimizer = optimizer
        self.lr = lr
        self.opt_kwargs = opt_kwargs if opt_kwargs is not None else {}

        # log hyperparameters
        self.save_hyperparameters(ignore=["neural_ldr"])

    def smile_kl_est(self, xp, xq, tau=10):
        """
        This est KL(p||q) using Eq.(17) from https://arxiv.org/pdf/1910.06222
        KL(p||q) = sup_T E_p[T] - log(E_q[exp(T)])
        """
        term1 = self.T(xp).mean()
        qs = torch.clamp(self.T(xq), min = -tau, max = tau)
        term2 = torch.logsumexp(qs, dim=0) - math.log(qs.shape[0])
        return term1 - term2

    def dv_loss(self, xp, xq):
        return -self.smile_kl_est(xp, xq, self.tau)

    def training_step(self, batch, batch_idx):
        xp, xq = batch
        loss = self.dv_loss(xp, xq)

        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        xp, xq = batch
        loss = self.dv_loss(xp, xq)

        self.log("val_loss", loss, on_step=True, on_epoch=True, logger=True)
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = self._optimizer(self.parameters(), lr=self.lr, **self.opt_kwargs)
        return optimizer
        # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = self.lr, pct_start = 0.1)
        # return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
