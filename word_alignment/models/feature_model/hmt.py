import numpy as np


def upward_downward(J, I, heads, translation_matrix, dist_probs, marginals):
    # I already includes the NULL word extension
    # heads = list of dim J, where each position specifies the index of the head in the list (heads[0] = 0)
    # dist_probs = 2I X 2I matrix (rows = from i', cols = to i)
    # trans_probs = J x 2I for the translation probabilities
    # the gamma here is the xi in the Kondo paper (single posterior)
    # the xi here is the p(aj|a_pa(j)) (double posterior) in the Kondo paper

    children = [set() for _ in xrange(J)]
    for j, h in enumerate(heads[1:]):
        children[h].add(j + 1)

    # compute marginals

    for j in xrange(1, J):
        marginals[j] = np.dot(marginals[heads[j]], dist_probs[j-1])

    # upward recursion betas
    betas = np.zeros((J, I))
    betas_p = np.zeros((J, I))
    log_likelihood = 0
    for j in xrange(J - 1, -1, -1):
        prod = np.ones(I, dtype=np.longfloat)
        for c in children[j]:
            # compute betas_p for j,c
            betas_p_c = np.dot(dist_probs[c-1], (betas[c] / marginals[c]))
            prod *= betas_p_c
            betas_p[c] = betas_p_c
        numerator = prod * translation_matrix[j] * marginals[j]
        N_j = np.sum(numerator)
        log_likelihood += np.log(N_j)
        betas[j] = np.divide(numerator, N_j)

    # downward recursion gammas and xis
    gammas = np.zeros((J, I))
    gammas[0] = betas[0]
    xis = [None]
    for j in xrange(1, J):
        parent = heads[j]
        gammas[j] = (betas[j] / marginals[j]) * np.dot((gammas[parent] / betas_p[j]), dist_probs[j-1])
        xi = np.outer((gammas[parent] / betas_p[j]), (betas[j] / marginals[j])) * dist_probs[j-1]
        xis.append(xi)
        # xi is xi[i_p, i] like dist_matrix

    return gammas, xis, log_likelihood
