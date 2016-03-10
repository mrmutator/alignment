import numpy as np

def forward(J, I, start_prob, dist_mat, trans_mat, alphas, scale_coeffs):
    alphas[0, :] = np.multiply(trans_mat[:, 0],start_prob[:])
    Z = np.sum(alphas[0, :])
    alphas[0, :] = alphas[0, :] / Z
    scale_coeffs[0] = Z

    for j in xrange(1, J):
        for i in xrange(I):
            alphas[j, i] = trans_mat[i, j] * np.sum(np.multiply(alphas[j-1, :], dist_mat[:, i]))
        Z = np.sum(alphas[j, :])
        alphas[j, :] = alphas[j, :] / Z
        scale_coeffs[j] = Z

def backward(J, I, dist_mat, trans_mat, betas, scale_coeffs):
    for j in xrange(J-2, -1, -1):
        for i in xrange(I):
            betas[j][i] = np.sum(np.multiply( dist_mat[i, :], np.multiply(trans_mat[ :, j+1], betas[j+1, :]))) / scale_coeffs[j+1]