import numpy as np

def upward_downward(f_toks, e_toks, heads, trans_params, dist_probs, start_probs):
    # I already includes the NULL word extension
    # heads = list of dim J, where each position specifies the index of the head in the list (heads[0] = 0)
    # dist_probs = 2I X 2I matrix (rows = from i', cols = to i)
    # trans_probs = J x 2I for the translation probabilities

    I = len(e_toks)
    J = len(f_toks)

    children = [set() for _ in xrange(J)]
    for j, h in enumerate(heads[1:]):
        children[h].add(j+1)

    # compute marginals

    marginals = np.zeros((J, I))

    marginals[0] = start_probs

    for j in xrange(J-1):
        marginals[j+1] = np.dot(marginals[j], dist_probs)


    # upward recursion betas
    betas = np.zeros((J, I))
    betas_p = np.zeros((J, I))
    log_likelihood = 0
    for j in range(J-1, -1, -1):
        prod = np.ones(I, dtype=np.longfloat)
        for c in children[j]:
            # compute betas_p for j,c
            betas_p_c = np.dot((betas[c] / marginals[c]), dist_probs)
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
        gammas[j] = (betas[j] / marginals[j]) * np.dot(dist_probs, (gammas[parent] / betas_p[j]))
        xi = np.outer((betas[j] / marginals[j]), (gammas[parent] / betas_p[j])) * dist_probs
        xis.append(xi)
        # xi and gamma can probably be computed in main method such that counts can be updated directly

    return gammas, xis, log_likelihood