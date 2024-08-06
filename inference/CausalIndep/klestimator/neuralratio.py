import math

import pytorch_lightning as pl
import torch

batched_dot = torch.func.vmap(torch.dot)


class CondNeuralRatioScore(torch.nn.Module):
    """
    base class for neural ratio estimator: score(x, c)
    with the goal of learning the logdensity ratio log p(x|c)/p(x)

    score(x, c) = NN(x) * W(c), where:
    - NN(x) is an encoder that outputs a vector of dimension featuredim
    - W is a vector of dimension featuredim,
    - c is in the finite set of discrete labels
    """

    def __init__(self, encoder, featuredim, numclass):
        super().__init__()
        self.encoder = encoder
        self.label_embedding = torch.nn.Embedding(numclass, featuredim)

    def forward(self, x, label):
        feature = self.encoder(x)
        return batched_dot(feature, self.label_embedding(label))

    def score_fix_label(self, xs, single_label):
        device = next(self.parameters()).device
        batch_size = xs.size(0)
        labels = torch.as_tensor(
            single_label * batch_size, dtype=torch.long, device=device
        )
        return self.forward(xs, labels)

    def logdensityratio(self, xs, c1, c0):
        """
        ideally the trained score(x, c) conv to log p(x|c)/p(x), hence
        logdensityratio(x, c1, c0) = log p(x|c1) - log p(x|c0)
        """
        # return log p(x|c1) - log p(x|c0)
        return self.score_fix_label(xs, c1) - self.score_fix_label(xs, c0)


# define the neural ratio estimator with mlp encoder
class NRSmlp(CondNeuralRatioScore):
    """
    NRS using a 3-layer MLP as the encoder NN(x)

    Example:
    ----------------------------
    Stest = NRSmlp(10, 10, 5, 3)
    x = torch.randn(5, 10)
    label = torch.LongTensor([1, 1, 2, 0, 0])

    Stest(x, label) # compute logp(x|c=label)
    Stest.score_fix_label(x, [1]) # compute log p(x|c=1)/p(x)
    Stest.logdensityratio(x, [1], [0]) # compute log p(x|c=1)/p(x|c= 0)
    """

    def __init__(self, inputdim, hiddendim, featuredim, numclass):
        encoder = torch.nn.Sequential(
            torch.nn.Linear(inputdim, hiddendim),
            torch.nn.ReLU(),
            torch.nn.Linear(hiddendim, hiddendim),
            torch.nn.ReLU(),
            torch.nn.Linear(hiddendim, featuredim),
        )
        super().__init__(encoder, featuredim, numclass)


class NRSfancyMLP(CondNeuralRatioScore):
    def __init__(self, inputdim, hdims, featuredim, numclass, dropout_rate=0.3):
        # Define the layers
        layers = []
        in_dim = inputdim
        for h_dim in hdims:
            layers.append(torch.nn.Linear(in_dim, h_dim))
            layers.append(torch.nn.BatchNorm1d(h_dim))  # Apply BatchNorm
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout_rate))  # Apply Dropout
            in_dim = h_dim

        # Output layer
        layers.append(torch.nn.Linear(in_dim, featuredim))

        # Combine layers in encoder
        encoder = torch.nn.Sequential(*layers)
        super().__init__(encoder, featuredim, numclass)


class LogDensityRatioEstimator(torch.nn.Module):
    """
    a wrapper for learned logdensity ratio: T(x) = log p(x)/q(x)

    The SMILE estimator requires wrapping the neural ratio estimator into this class


    Example:
    ----------------------------
    ## wrap NREmlp into LogDensityRatioEstimator
    # class ldrmlp(LogDensityRatioEstimator):
    #     def __init__(self, nre:torch.nn.Module):
    #         super().__init__(nre)

    #     def ldr(self, x):
    #         return self.nre.score_fix_label(x, [0])

    nre = NRSmlp(10, 10, 5, 3)
    T = ldrmlp(nre)
    x = torch.randn(5, 10)
    T(x)
    """

    def __init__(self, nre: torch.nn.Module):
        super().__init__()
        self.nre = nre

    def ldr(self, x):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class ldr_nre(LogDensityRatioEstimator):
    """
    Example:
    ---------
    model = NREHermans.load_from_checkpoint(ckpt_file)
    nre = model.score
    T = ldr_nre(nre, [1])
    """

    def __init__(self, nre: CondNeuralRatioScore, pert_label: int):
        super().__init__(nre)
        self.pert_label = [pert_label]

    def ldr(self, x):
        return self.nre.logdensityratio(x, [0], self.pert_label)

    def forward(self, x):
        return self.ldr(x)



class NREHermans(pl.LightningModule):
    """
    A binary classifier to distinguish (θ, x) pairs drawn dependently p(θ, x) from those drawn independently p(θ)p(x)
    see Eq.(3) in the paper https://openreview.net/pdf?id=kOIaB1hzaLe for details

    Attributes:
    - score: CondNeuralRatioScore: the neural ratio estimator to be trained, fw(θ, x) in the paper

    Reference:
    J. Hermans, V. Begy, and G. Louppe. Likelihood-free mcmc with amortized approximate ratio estimators. 2020
    """

    def __init__(
        self,
        nclass: int,
        nrs_model: CondNeuralRatioScore,
        nrsmodel_hparams,
        optimizer=torch.optim.Adam,
        lr=1e-3,
        opt_kwargs=None,
    ):
        super().__init__()
        # self.score = CondNeuralRatioScore(inputdim, hiddendim, featuredim, numclass)
        self.nrshparams = nrsmodel_hparams if nrsmodel_hparams is not None else {}
        self.score = nrs_model(**self.nrshparams)  # instantiate the nrs class within
        self._optimizer = optimizer
        self.lr = lr
        self.opt_kwargs = opt_kwargs if opt_kwargs is not None else {}
        self.nclass = nclass

        # log hyperparameters
        self.save_hyperparameters()

    def forward(self, x, label):
        return self.score(x, label)

    def logdensityratio(self, x, c1, c0):
        # return log p(x|c1) - log p(x|c0)
        return self.score.logdensityratio(x, c1, c0)

    def logits(self, x_joint, label_joint, x_marg, label_marg):
        assert (
            x_joint.size(0)
            == x_marg.size(0)
            == label_joint.size(0)
            == label_marg.size(0)
        )  # ensure equal batchsize

        # here we use the notion "postive/nagative scores" to coincide with typical language used in NCE literature
        # (classifying postive samples from the union of both)
        score_pos = self.score(x_joint, label_joint)
        score_neg = self.score(x_marg, label_marg)

        logits_pos = torch.nn.functional.logsigmoid(score_pos)
        logits_neg = torch.nn.functional.logsigmoid(score_neg)
        return logits_pos, logits_neg, score_pos, score_neg

    def nreloss_and_acc(self, x_joint, label_joint, x_marg, label_marg):
        # score_pos = self.score(x_joint, label_joint)
        # score_neg = self.score(x_marg, label_marg)
        # lln_pos = torch.nn.functional.logsigmoid(score_pos).mean()
        # lln_neg = -score_neg.mean() + torch.nn.functional.logsigmoid(score_neg).mean()

        logits_pos, logits_neg, score_pos, score_neg = self.logits(
            x_joint, label_joint, x_marg, label_marg
        )
        ll_pos = logits_pos.mean()
        ll_neg = -score_neg.mean() + logits_neg.mean()
        # compute loss
        loss = -ll_pos / 2 - ll_neg / 2

        # compute accuracy
        ncorrect = (logits_pos > math.log(0.5)).sum() + (
            logits_neg < math.log(0.5)
        ).sum()
        batch_size = 2 * len(label_joint)
        acc = ncorrect / batch_size

        return loss, acc

    def training_step(self, batch, batch_idx):
        x_joint, label_joint = batch
        label_marg = torch.randint_like(label_joint, 0, self.nclass)
        loss, acc = self.nreloss_and_acc(x_joint, label_joint, x_joint, label_marg)

        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, logger=True)
        return {"loss": loss, "acc": acc}

    def validation_step(self, batch, batch_idx):
        x_joint, label_joint = batch
        label_marg = torch.randint_like(label_joint, 0, self.nclass)
        loss, acc = self.nreloss_and_acc(x_joint, label_joint, x_joint, label_marg)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return {"loss": loss, "acc": acc}

    def test_step(self, batch, batch_idx):
        x_joint, label_joint = batch
        label_marg = torch.randint_like(label_joint, 0, self.nclass)
        loss, acc = self.nreloss_and_acc(x_joint, label_joint, x_joint, label_marg)

        self.log("test_loss", loss, prog_bar=True)
        return {"loss": loss, "acc": acc}

    def configure_optimizers(self):
        optimizer = self._optimizer(self.parameters(), lr=self.lr, **self.opt_kwargs)
        return optimizer
        # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = self.lr, pct_start = 0.1)
        # return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]


# class infoNCE(pl.LightningModule):
#     def __init__(self, inputdim, hiddendim, featuredim, numclass, lr=1e-3):
#         super().__init__()
#         self.lr = lr
#         self.score = CondNeuralRatioScore(inputdim, hiddendim, featuredim, numclass)
#
#     def forward(self, x, label):
#         return self.model(x, label)
#
#     def infoloss(self, x, xn, label):
#         score_pos = self.score(x, label)
#         score_neg = self.score.score_fix_negsamples(xn, label)
#         return
#
#     def training_step(self, batch, batch_idx):
#         x, label = batch
#         y = self.model(x, label)
#         loss = self.loss(x, y, label)
#         acc = self.accuracy(y, label)
#         self.log("train_loss", loss)
#         self.log("train_acc", acc)
#         return loss
#
#     def validation_step(self, batch, batch_idx):
#         x, xn, label = batch  # postive sample, negative samples, labels
#         y = self.model(x, label)
#         loss = self.loss(y, label)
#         acc = self.accuracy(y, label)
#         self.log("val_loss", loss)
#         self.log("val_acc", acc)
#         return loss
#
#     def configure_optimizers(self):
#         return optim.Adam(self.model.parameters(), lr=self.lr)
