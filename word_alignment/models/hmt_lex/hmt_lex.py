import numpy as np


def upward_downward(f_toks, e_toks, heads, head_pos,  trans_params, dist_probs, start_probs):
    # I already includes the NULL word extension
    # heads = list of dim J, where each position specifies the index of the head in the list (heads[0] = 0)
    # dist_probs = 2I X 2I matrix (rows = from i', cols = to i)
    # trans_probs = J x 2I for the translation probabilities
    # the gamma here is the xi in the Kondo paper (single posterior)
    # the xi here is the p(aj|a_pa(j)) (double posterior) in the Kondo paper

    I = len(e_toks)
    J = len(f_toks)

    children = [set() for _ in xrange(J)]
    for j, h in enumerate(heads[1:]):
        children[h].add(j + 1)

    # compute marginals

    marginals = np.zeros((J, I))

    marginals[0] = start_probs

    for j in xrange(J - 1):
        p = head_pos[j+1]
        marginals[j + 1] = np.dot(marginals[j], dist_probs[p])

    # upward recursion betas
    betas = np.zeros((J, I))
    betas_p = np.zeros((J, I))
    log_likelihood = 0
    for j in range(J - 1, -1, -1):
        prod = np.ones(I, dtype=np.longfloat)
        for c in children[j]:
            p = head_pos[c] # is the same for each c
            # compute betas_p for j,c
            betas_p_c = np.dot(dist_probs[p], (betas[c] / marginals[c]))
            prod *= betas_p_c
            betas_p[c] = betas_p_c
        t_j = np.array([trans_params.get((e_tok, f_toks[j]), 0.00000001) for e_tok in e_toks])
        numerator = prod * t_j * marginals[j]
        N_j = np.sum(numerator)
        log_likelihood += np.log(N_j)
        betas[j] = np.divide(numerator, N_j)

    # downward recursion gammas and xis
    gammas = np.zeros((J, I))
    gammas[0] = betas[0]
    xis = [None]
    for j in range(1, J):
        parent = heads[j]
        p = head_pos[j]
        gammas[j] = (betas[j] / marginals[j]) * np.dot((gammas[parent] / betas_p[j]), dist_probs[p])
        xi = np.outer((gammas[parent] / betas_p[j]), (betas[j] / marginals[j])) * dist_probs[p]
        xis.append(xi)
        # xi is xi[i_p, i] like dist_matrix

    return gammas, xis, log_likelihood
