import torch

def compute_kernel_matrix(x, y, kernel):
    kernelMat = kernel(x, y).to_dense()
    return kernelMat

def MMD_square(x, y, kernel):
    n1 = x.size(0)
    n2 = y.size(0)

    # The three constants used in the test.
    a00 = 1. / (n1 * (n1 - 1))
    a11 = 1. / (n2 * (n2 - 1))
    a01 = - 2. / (n1 * n2)

    K_xx = compute_kernel_matrix(x, x, kernel)
    K_yy = compute_kernel_matrix(y, y, kernel)
    K_xy = compute_kernel_matrix(x, y, kernel)

    mmd = (a01*K_xy.sum() + 
        a00 * (K_xx.sum() - torch.trace(K_xx)) + 
        a11 * (K_yy.sum() - torch.trace(K_yy))
    )
    return mmd

