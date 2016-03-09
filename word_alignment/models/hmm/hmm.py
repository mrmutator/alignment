import numpy as np

def random_start_prob(I):
    p = np.random.random(I)
    Z = np.sum(p)
    return p / Z

def random_dist_prob(I):
    p = np.random.random((I, I))
    Z = np.sum(p, axis=1)
    return p / Z[:, np.newaxis]

def random_emission_prob(I, J):
    p = np.random.random((I, J))
    Z = np.sum(p, axis=1)
    return p / Z[:, np.newaxis]


def forward(J, I, start_prob, dist_mat, trans_mat, alphas, scale_coeffs):
    for i in xrange(I):
        alphas[0, i] = trans_mat[i, 0] * start_prob[i]
    Z = np.sum(alphas[0, :])
    alphas[0, :] = alphas[0, :] / Z
    scale_coeffs[0] = Z

    for j in xrange(1, J):
        for i in xrange(I):
            alphas[j, i] = trans_mat[i, j] * np.sum(np.multiply(alphas[j-1, :], dist_mat[:, i]))
        Z = np.sum(alphas[j, :])
        alphas[j, :] = alphas[j, :] / Z
        scale_coeffs[j] = Z



# I = 5
# J = 6
#
# start_prob = random_start_prob(I)
# dist_prob = random_dist_prob(I)
# trans_prob = random_emission_prob(I, J)
#
# for _ in xrange(100000):
#     alphas = np.zeros((J, I))
#     scale_coffs = np.zeros(J)
#
#     forward(J, I, start_prob, dist_prob, trans_prob, alphas, scale_coffs)