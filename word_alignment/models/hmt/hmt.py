import numpy as np

def upward_downward(heads, trans_probs, dist_probs, start_probs, J, I):
    # I already includes the NULL word extension
    # heads = list of dim J, where each position specifies the index of the head in the list (heads[0] = 0)
    # dist_probs = 2I X 2I matrix (rows = from i', cols = to i)
    # trans_probs = J x 2I for the translation probabilities

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
    for j in range(J-1, -1, -1):
        prod = np.ones(I)
        for c in children[j]:
            # compute betas_p for j,c
            betas_p_c = np.dot((betas[c] / marginals[c]), dist_probs)
            prod *= betas_p_c
            betas_p[c] = betas_p_c
        numerator = prod * trans_probs[j] * marginals[j]
        N_j = np.sum(numerator)
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

    print bla

if __name__ == "__main__":


    def random_start_prob(I):
        p = np.random.random(I)
        Z = np.sum(p)
        return p / Z

    def random_dist_prob(I):
        p = np.random.random((I, I))
        Z = np.sum(p, axis=1)
        return p / Z[:, np.newaxis]

    def random_emission_prob(J, I):
        p = np.random.random((J, I))
        Z = np.sum(p, axis=0) # np.random.random()
        return p / Z[np.newaxis, :]

    I = 5
    J = 6
    heads = [0, 0, 0, 1, 1, 2]

    start_prob = random_start_prob(I)
    dist_prob = random_dist_prob(I)
    trans_prob = random_emission_prob(J, I)

    for _ in xrange(1):
        upward_downward(heads, trans_prob, dist_prob, start_prob, J, I)