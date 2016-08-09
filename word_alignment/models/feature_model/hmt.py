import numpy as np


def upward_downward(J, I, heads, translation_matrix, dist_probs, marginals):
    # I already includes the NULL word extension
    # heads = list of dim J, where each position specifies the index of the head in the list (heads[0] = 0)
    # dist_probs = 2I X 2I matrix (rows = from i', cols = to i)
    # trans_probs = J x 2I for the translation probabilities
    # the gamma here is the xi in the Kondo paper (single posterior)
    # the xi here is the p(aj|a_pa(j)) (double posterior) in the Kondo paper
    SMALL_CONST = 0.0000000000000000001
    # compute marginals

    for j in xrange(1, J):
        marginals[j] = np.dot(marginals[heads[j]], dist_probs[j-1])

    # upward recursion betas
    betas = np.zeros((J, I))
    betas_p = np.zeros((J, I))
    prod = np.ones((J, I), dtype=np.longfloat)
    log_likelihood = 0
    for j in xrange(J - 1, 0, -1):
        numerator = prod[j] * translation_matrix[j] * marginals[j]
        N_j = np.sum(numerator)
        if N_j == 0:
            N_j = SMALL_CONST
        log_likelihood += np.log(N_j)
        betas[j] = np.divide(numerator, N_j)
        betas_p[j] = np.dot(dist_probs[j - 1], (betas[j] / marginals[j]))
        prod[heads[j]]*= betas_p[j]

    # j=0
    numerator = prod[0] * translation_matrix[0] * marginals[0]
    N_j = np.sum(numerator)
    if N_j == 0:
        N_j = SMALL_CONST
    log_likelihood += np.log(N_j)
    betas[0] = np.divide(numerator, N_j)

    # downward recursion gammas and xis
    gammas = np.zeros((J, I))
    gammas[0] = betas[0]
    xis = [gammas[0]]
    for j in xrange(1, J):
        parent = heads[j]
        gammas[j] = (betas[j] / marginals[j]) * np.dot((gammas[parent] / betas_p[j]), dist_probs[j-1])
        xi = np.outer((gammas[parent] / betas_p[j]), (betas[j] / marginals[j])) * dist_probs[j-1]
        xis.append(xi)
        # xi is xi[i_p, i] like dist_matrix

    return gammas, xis, log_likelihood
