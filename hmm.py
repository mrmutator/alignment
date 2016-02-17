from models import AlignmentModel, random_prob
from Corpus_Reader import Corpus_Reader
from collections import defaultdict, Counter
import numpy as np
import sys
import random

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
                # e_toks = [None] + e_toks # introduce NULL-token
                I = len(e_toks)
                J = len(f_toks)
                alphas = np.zeros((J, I))
                betas = np.zeros((J, I))
                for j, f_tok in enumerate(f_toks):
                    for i, e_tok in enumerate(e_toks):
                        if j == 0:
                            alpha_j_i = self.trans_prob[e_tok][f_tok] * self.al_prob[-1, I][i]
                        else:
                            alpha_j_i = self.trans_prob[e_tok][f_tok] * np.sum([ alphas[j-1][k]*self.al_prob[(k, I)][i] for k in range(I)])
                        alphas[j][i] = alpha_j_i

                        # lexical translation parameters
                        delta_t = self.trans_prob[e_tok][f_tok] / np.sum([self.trans_prob[k_tok][f_tok] for k_tok in e_toks])
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
                    for i_p in range(I):
                        alpha_j_p_i_p = alpha_j_p[i_p]
                        al_i_p = self.al_prob[i_p, I]
                        gamma_sums[(i_p, I)] += gammas[j_p][i_p]
                        for i in range(I):
                            xi = (al_i_p[i]  * alpha_j_p_i_p * beta_t_j_i[i]) / denom
                            xi_sums[(i, i_p, I)] += xi
                # add counts
                for i in range(I):
                    pi_counts[(i, I)] += gammas[0][i]
                pi_denom[I] += 1


            # update parameters

            for (e,f), count in counts_e_f.items():
                self.trans_prob[e][f] = count / counts_e[e]

            for (i, i_p, I), count in xi_sums.items():
                self.al_prob[(i_p, I)][i] = count / gamma_sums[(i_p, I)]

            for (i, I), count in pi_counts.items():
                self.al_prob[(-1, I)][i] = count / pi_denom[I]

            #self.test_probs()

            old_ll = data_ll
            data_ll =  self.calculate_data_log_likelihood(corpus)
            log_file.write(str(data_ll) + "\n")
            print data_ll
            delta_ll =  data_ll - old_ll

        if delta_ll <= convergence_ll:
            return True
        else:
            return False

    def calculate_pair_log_likelihood(self, f_toks, e_toks):
        """
        Calculates the log-likelihood of a translation pair log p(f|e). Note that it is only proportional.
        """
        J = len(f_toks)
        I = len(e_toks)
        alphas = np.zeros((J, I))
        for j, f_tok in enumerate(f_toks):
            for i, e_tok in enumerate(e_toks):
                if j == 0:
                    alpha_j_i = self.trans_prob[e_tok][f_tok] * self.al_prob[-1, I][i]
                else:
                    alpha_j_i = self.trans_prob[e_tok][f_tok] * np.sum([ alphas[j-1][k]*self.al_prob[(k, I)][i] for k in range(I)])
                alphas[j][i] = alpha_j_i

        sent_prob = np.sum(alphas[J-1])
        return np.log(sent_prob)

    def get_viterbi_alignment(self, e_toks, f_toks):
        alignments = []
        J = len(f_toks)
        I = len(e_toks)
        chart = np.zeros((J, I))
        best = np.zeros((J, I))
        # initialize
        for i, e_tok in enumerate(e_toks):
            chart[0][i] = self.trans_prob[e_tok][f_toks[0]] * self.al_prob[(-1, I)][i]

        # compute chart
        for j, f_tok in enumerate(f_toks[1:]):
            j= j+1
            for i, e_tok in enumerate(e_toks):
                values = []
                for i_p in range(I):
                    values.append(chart[j-1][i_p]*self.al_prob[(i_p, I)][i])
                best_i = np.argmax(values)
                chart[j][i] = values[best_i] * self.trans_prob[e_tok][f_tok]
                best[j][i] = best_i

        # get max path
        best_path = []
        best_end = np.argmax(chart[J-1])
        best_path.append(best_end)
        for j in reversed(range(1, J)):
            best_end = best[j][best_end]
            best_path.append(best_end)

        best_path = list(reversed(best_path))
        alignments = [(int(best_path[j]), j) for j in range(J)]
        return alignments

    def test_probs(self):
        test_es = random.sample(self.trans_prob.keys(), 10)
        for e in test_es:
            assert abs(sum(self.trans_prob[e].values()) - 1.0) < 0.000001

        test_als = random.sample(self.al_prob.keys(), 10)
        for al in test_als:
            assert abs(sum(self.al_prob[al].values()) - 1.0) < 0.0000001



if __name__ == "__main__":
    # create corpus instance
    corpus = Corpus_Reader("../ALT_Lab1/data/file.en", "../ALT_Lab1/data/file.de", alignment_order=("f", "e"), limit=1000)

    # create model, omit parameters for random initialization
    model = HMMAlignment()

    # train model until convergence delta is small enough or max_iterations is reached
    model.train(corpus, max_iterations=5, convergence_ll=0.001)
    alignments = model.get_all_viterbi_alignments(corpus)
    print alignments
    outfile = open("test.al", "w")
    for als in alignments:
        als = [str(al[0]) + "-" + str(al[1]) for al in als]
        outfile.write(" ".join(als) + "\n")
    outfile.close()

