import torch


def kl_divergence_gaussian(m1, S1, m2, S2):
    """
    Compute the KL divergence between two multivariate Gaussians.

    Args:
    m1 (torch.Tensor): Mean vector of the first Gaussian distribution.
    S1 (torch.Tensor): Covariance matrix of the first Gaussian distribution.
    m2 (torch.Tensor): Mean vector of the second Gaussian distribution.
    S2 (torch.Tensor): Covariance matrix of the second Gaussian distribution.

    Returns:
    torch.Tensor: Scalar representing the KL divergence.
    """
    # Ensure the covariance matrices are positive definite
    S1 = S1 + 1e-6 * torch.eye(S1.size(0))
    S2 = S2 + 1e-6 * torch.eye(S2.size(0))

    # Calculate the necessary components for the KL divergence formula
    inv_S2 = torch.linalg.inv(S2)
    diff_mu = m2 - m1
    term1 = torch.trace(torch.matmul(inv_S2, S1))
    term2 = torch.dot(diff_mu, torch.matmul(inv_S2, diff_mu))
    log_det_ratio = torch.logdet(S2) - torch.logdet(S1)

    # Dimensionality of the Gaussian distributions
    k = m1.size(0)

    # Calculate KL divergence
    kl_div = (term1 + term2 - k + log_det_ratio) / 2

    return kl_div


def kl_mvn_est(x1, x2):
    m1, S1 = x1.mean(dim=0), x1.T.cov()
    m2, S2 = x2.mean(dim=0), x2.T.cov()
    return kl_divergence_gaussian(m1, S1, m2, S2)


# # Example usage
# m1 = torch.randn(3)
# S1 = torch.rand(3, 3)
# S1 = torch.mm(S1, S1.t())  # Make it symmetric and positive definite
# m2 = torch.randn(3)
# S2 = torch.rand(3, 3)
# S2 = torch.mm(S2, S2.t())  # Make it symmetric and positive definite

# kl_div = kl_divergence_gaussian(m1, S1, m2, S2)
