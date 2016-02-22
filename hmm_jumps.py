from models import AlignmentModel, random_prob
from Corpus_Reader import Corpus_Reader
from collections import defaultdict, Counter
import numpy as np
import sys
import random
import cPickle as pickle
import multiprocessing as mp

class HMMAlignment(AlignmentModel):

    def __init__(self, trans_prob=None, al_prob=None, p_0=0.2):
        self.p_0 = p_0 # null word transition probability
        # initialize t(f|e) randomly
        # t(f|e) = trans_prob[e][f]
        if not trans_prob:
            self.trans_prob = defaultdict(lambda: defaultdict(random_prob))
        else:
            self.trans_prob = dict()
            self.load_trans_prob(trans_prob)

        # initialize p(a_j|a_j-1, I) randomly
        # = al_prob[(a_j-1,I)][a_j]
        # initial probs (-1,I)[a_j]
        if not al_prob:
            self.al_prob = defaultdict(lambda: defaultdict(random_prob))
        else:
            self.al_prob = dict()
            self.load_al_prob(al_prob)

    def load_trans_prob(self, trans_probs):
        for (e,f), v in trans_probs.iteritems():
            if e not in self.trans_prob:
                self.trans_prob[e] = dict()
            self.trans_prob[e][f] = v

    def load_al_prob(self, al_probs):
        for (cond, jmp), v in al_probs.iteritems():
            if cond not in self.al_prob:
                self.al_prob[cond] = dict()
            self.al_prob[cond][jmp] = v


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
            counts_e_f, counts_e, gamma_sums, xi_sums, pi_counts, pi_denom = self.train_iteration(corpus)

            # update parameters

            for (e,f), count in counts_e_f.items():
                self.trans_prob[e][f] = count / counts_e[e]

            jmp_dict = defaultdict(lambda: defaultdict(int))
            for (i, i_p, I), count in xi_sums.items():
                jmp_dict[I][i-i_p] += count / gamma_sums[(i_p, I)]
            for I_p in jmp_dict:
                norm_c = np.sum(jmp_dict[I_p].values())
                for jmp in jmp_dict[I_p]:
                    self.al_prob[I_p][jmp] = jmp_dict[I_p][jmp] / norm_c

            for (i, I), count in pi_counts.items():
                self.al_prob[(None, I)][i] = count / pi_denom[I]

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

    def train_iteration(self, corpus):
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
            alphas = np.zeros((J, 2*I))
            betas = np.zeros((J, 2*I))
            scale_coeffs = np.zeros(J)
            for j, f_tok in enumerate(f_toks):
                for i, e_tok in enumerate(e_toks + [0] * I):
                    t_f_e = self.trans_prob[e_tok][f_tok]
                    if e_tok == 0:
                        # i is NULL
                        if j == 0:
                            alphas[j][i] = t_f_e * self.p_0
                        else:
                            # go to NULL_i is only possible from NULL_i or from NULL_i - I
                            alphas[j][i] = t_f_e * np.sum([alphas[j - 1][k] * self.p_0 for k in [i-I, i]])
                    else:
                        # i is not null
                        if j == 0:
                            alphas[j][i] = t_f_e * self.al_prob[(None, I)][i]
                        else:
                            alphas[j][i] = t_f_e * (np.sum([alphas[j - 1][k] * self.al_prob[I][i - k]
                                                            for k in xrange(I)]) + np.sum([alphas[j-1][k] *
                                                                    self.al_prob[I][i - k + I] for k in range(I, 2*I)]))


                    # lexical translation parameters
                    delta_t = t_f_e / np.sum([self.trans_prob[k_tok][f_tok] for k_tok in e_toks + [0]*I])
                    counts_e_f[(e_tok, f_tok)] += delta_t
                    counts_e[e_tok] += delta_t
                # rescale alphas for numerical stability
                Z = np.sum(alphas[j])
                alphas[j] = alphas[j] / Z
                scale_coeffs[j] = Z

            for j, f_tok in reversed(list(enumerate(f_toks))):
                for i, e_tok in enumerate(e_toks + [0] * I):
                    if e_tok == 0:
                        # i is NULL
                        if j == J-1:
                            betas[j][i] = 1
                        else:
                            betas[j][i] = np.sum([betas[j+1][k] * self.trans_prob[e_k][f_toks[j+1]] *
                                                  self.al_prob[I][k-i+I] for k, e_k in enumerate(e_toks)]) +\
                                          (betas[j+1][i] * self.trans_prob[0][f_toks[j+1]] *
                                                  self.p_0)
                    else:
                        # i is not NULL
                        if j == J-1:
                            betas[j][i] = 1
                        else:
                            betas[j][i] = np.sum([betas[j+1][k] * self.trans_prob[e_k][f_toks[j+1]] *
                                                  self.al_prob[I][k-i] for k, e_k in enumerate(e_toks)]) + \
                                          (betas[j+1][i+I] * self.trans_prob[0][f_toks[j+1]] *
                                                  self.p_0)
                if j != J-1:
                    # rescale betas for numerical stability
                    betas[j] = betas[j] / scale_coeffs[j+1]

            # posteriors
            # multiplied = np.multiply(alphas, betas)
            # denom_sums = 1.0 / np.sum(multiplied, axis=1)
            #
            # gammas = multiplied * denom_sums[:, np.newaxis]

            gammas = np.multiply(alphas, betas)

            for j in xrange(1, J):
                t_f_e = np.array([self.trans_prob[e_tok][f_toks[j]] for e_tok in e_toks + [0]*I])
                beta_t_j_i = np.multiply(betas[j], t_f_e)
                j_p = j-1
                alpha_j_p = alphas[j_p]
                # denom = np.sum(np.multiply(alpha_j_p, betas[j_p]))
                for i_p in range(2*I):
                    alpha_j_p_i_p = alpha_j_p[i_p]
                    gamma_sums[(i_p, I)] += gammas[j_p][i_p]
                    for i in range(2*I):
                        if i >= I:
                            if i - I == i_p or i == i_p: # i is NULL
                                xi = (self.p_0  * alpha_j_p_i_p * beta_t_j_i[i]) / scale_coeffs[j]
                                xi_sums[(i,i_p, I)] += xi
                        else:
                            if i_p < I:
                                xi = (self.al_prob[I][i - i_p]  * alpha_j_p_i_p * beta_t_j_i[i]) / scale_coeffs[j]
                                xi_sums[(i,i_p, I)] += xi
                            else:
                                xi = (self.al_prob[I][i - i_p + I]  * alpha_j_p_i_p * beta_t_j_i[i]) / scale_coeffs[j]
                                xi_sums[(i,i_p-I, I)] += xi
            # add counts
            for i in range(2*I):
                pi_counts[(i, I)] += gammas[0][i]
            pi_denom[I] += 1

        return counts_e_f, counts_e, gamma_sums, xi_sums, pi_counts, pi_denom



    def calculate_pair_log_likelihood(self, f_toks, e_toks):
        """
        Calculates the log-likelihood of a translation pair log p(f|e). Note that it is only proportional.
        """
        J = len(f_toks)
        I = len(e_toks)
        alphas = np.zeros((J, 2*I))
        for j, f_tok in enumerate(f_toks):
            for i, e_tok in enumerate(e_toks + [0]*I):
                t_f_e = self.trans_prob[e_tok][f_tok]
                if e_tok == 0:
                    # i is NULL
                    if j == 0:
                        alphas[j][i] = t_f_e * self.p_0
                    else:
                        # go to NULL_i is only possible from NULL_i or from NULL_i - I
                        alphas[j][i] = t_f_e * np.sum([alphas[j - 1][k] * self.p_0 for k in [i-I, i]])
                else:
                    # i is not null
                    if j == 0:
                        alphas[j][i] = t_f_e * self.al_prob[(None, I)][i]
                    else:
                        alphas[j][i] = t_f_e * (np.sum([alphas[j - 1][k] * self.al_prob[I][i - k]
                                                        for k in xrange(I)]) + np.sum([alphas[j-1][k] *
                                                                self.al_prob[I][i - k + I] for k in range(I, 2*I)]))

        sent_prob = np.sum(alphas[J-1])
        return np.log(sent_prob)

    def get_viterbi_alignment(self, e_toks, f_toks):
        J = len(f_toks)
        I = len(e_toks)
        chart = np.zeros((J, 2*I))
        best = np.zeros((J, 2*I))
        # initialize
        for i, e_tok in enumerate(e_toks):
            chart[0][i] = self.trans_prob[e_tok][f_toks[0]] * self.al_prob[(None, I)][i]
        for i in range(I, I*2):
            chart[0][i] = self.trans_prob[0][f_toks[0]] * self.p_0

        # compute chart
        for j, f_tok in enumerate(f_toks[1:]):
            j= j+1
            for i, e_tok in enumerate(e_toks + [0]*I):
                values = []
                for i_p in range(2*I):
                    if i >= I:
                        if i - I == i_p or i == i_p: # i is NULL
                            values.append(chart[j-1][i_p] * self.p_0)
                        else:
                            values.append(0)
                    else:
                        if i_p < I:
                            values.append(chart[j-1][i_p]*self.al_prob[I][i-i_p])
                        else:
                            values.append(chart[j-1][i_p]*self.al_prob[I][i-i_p + I])
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
        alignments = [(int(best_path[j]), j) for j in range(J) if int(best_path[j]) < I]
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
    corpus = Corpus_Reader("test/tp.1.e", "test/tp.1.f", limit=100)
    #trans_params, al_params = pickle.load(open("test/tp.1.prms", "rb"))
    # create model, omit parameters for random initialization
    model = HMMAlignment(al_prob=None, trans_prob=None)

    # train model until convergence delta is small enough or max_iterations is reached
    model.train(corpus, max_iterations=5, convergence_ll=0.001)
    alignments = model.get_all_viterbi_alignments(corpus)
    print alignments
    # outfile = open("test.al", "w")
    # for als in alignments:
    #     als = [str(al[0]) + "-" + str(al[1]) for al in als]
    #     outfile.write(" ".join(als) + "\n")
    # outfile.close()
