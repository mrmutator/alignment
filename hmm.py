from models import AlignmentModel, random_prob
from Corpus_Reader import Corpus_Reader
from collections import defaultdict, Counter
import numpy as np
import sys

class HMMAlignment(AlignmentModel):

    def __init__(self, trans_prob=None, al_prob=None):
        # initialize t(f|e) randomly
        # t(f|e) = trans_prob[e][f]
        if not trans_prob:
            self.trans_prob = defaultdict(lambda: defaultdict(random_prob))
        else:
            self.trans_prob = trans_prob

        # initialize p(a_j|a_j-1, I) randomly
        # = al_prob[(a_j-1,I)][a_j]
        # initial probs (-1,I)[a_j]
        if not al_prob:
            self.al_prob = defaultdict(lambda: defaultdict(random_prob))
        else:
            self.al_prob = al_prob = al_prob


    def train(self, corpus, max_iterations=np.inf, convergence_ll=0.01, log_file = sys.stdout):
        """
        Trains the model using EM parameter estimation.
        """
        iteration = 0
        delta_ll = np.inf
        data_ll = - np.inf
        while(iteration < max_iterations and delta_ll > convergence_ll):
            iteration += 1
            print "Iteration ", iteration
            # set all counts to zero
            counts_e_f = Counter()
            counts_e = Counter()
            pi_counts = Counter() # (i, I)
            pi_denom = Counter() # I
            xi_sums = Counter() # (i, i_p, I)
            gamma_sums = Counter() # (i_p, I)
            for e_toks, f_toks in corpus:
                e_toks = [None] + e_toks # introduce NULL-token
                I = len(e_toks) -1
                J = len(f_toks)
                alphas = np.zeros((J, I+1))
                betas = np.zeros((J, I+1))
                for j, f_tok in enumerate(f_toks):
                    for i, e_tok in enumerate(e_toks):
                        if j == 0:
                            alpha_j_i = self.trans_prob[e_tok][f_tok] * self.al_prob[-1, I][i]
                        else:
                            alpha_j_i = self.trans_prob[e_tok][f_tok] * np.sum([ alphas[j-1][k]*self.al_prob[(k, I)][i] for k in range(I+1)])
                        alphas[j][i] = alpha_j_i

                        # lexical translation parameters
                        delta_t = self.trans_prob[e_tok][f_tok] / np.sum([self.trans_prob[k_tok][f_tok] for k_tok in [None] + e_toks])
                        counts_e_f[(e_tok, f_tok)] += delta_t
                        counts_e[e_tok] += delta_t

                for j, f_tok in reversed(list(enumerate(f_toks))):
                    for i, e_tok in enumerate(e_toks):
                        if j == J-1:
                            beta_j_i = 1
                        else:
                            beta_j_i = np.sum([betas[j+1][k] * self.trans_prob[e_k][f_toks[j+1]] * self.al_prob[(i, I)][k] for k, e_k in enumerate(e_toks)])
                        betas[j][i] = beta_j_i

                # posteriors
                multiplied = np.multiply(alphas, betas)
                denom_sums = 1.0 / np.sum(multiplied, axis=1)

                gammas = multiplied * denom_sums[:, np.newaxis]

                for j in range(1, J):
                    t_f_e = np.array([self.trans_prob[e_tok][f_toks[j]] for e_tok in e_toks])
                    beta_t_j_i = np.multiply(betas[j], t_f_e)
                    j_p = j-1
                    alpha_j_p = alphas[j_p]
                    denom = np.sum(np.multiply(alphas[j_p], betas[j_p]))
                    for i_p in range(I+1):
                        alpha_j_p_i_p = alpha_j_p[i_p]
                        al_i_p = self.al_prob[i_p, I]
                        gamma_sums[(i_p, I)] += gammas[j_p][i_p]
                        for i in range(i+1):
                            xi = (al_i_p[i]  * alpha_j_p_i_p * beta_t_j_i[i]) / denom
                            xi_sums[(i, i_p, I)] += xi
                # add counts
                for i in range(I+1):
                    pi_counts[(i, I)] += gammas[0][i]
                pi_denom[I] += 1


            # update parameters

            for (e,f), count in counts_e_f.items():
                self.trans_prob[e][f] = count / counts_e[e]

            denoms = Counter()
            for (i, i_p, I), count in xi_sums.items():
                self.al_prob[(i_p, I)][i] = count / gamma_sums[(i_p, I)]

            for (i, I), count in pi_counts.items():
                self.al_prob[(-1, I)][i] = count / pi_denom[I]

            old_ll = data_ll
            data_ll =  self.calculate_data_log_likelihood(corpus)
            log_file.write(str(data_ll) + "\n")
            print data_ll
            delta_ll =  data_ll - old_ll

        if delta_ll <= convergence_ll:
            return True
        else:
            return False

if __name__ == "__main__":
    # create corpus instance
    corpus = Corpus_Reader("../ALT_Lab1/data/file.en", "../ALT_Lab1/data/file.de", alignment_order=("f", "e"), limit=10)

    # create model, omit parameters for random initialization
    model = HMMAlignment()

    # train model until convergence delta is small enough or max_iterations is reached
    model.train(corpus, max_iterations=10, convergence_ll=0.001)


