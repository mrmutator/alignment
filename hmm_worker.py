from __future__ import division
from Corpus_Reader import Corpus_Reader
from collections import defaultdict, Counter
import numpy as np
import cPickle as pickle
import multiprocessing as mp
import argparse


def train_iteration(corpus, trans_prob, jump_params, start_prob, p_0, results, alpha=0):
    # set all counts to zero
    counts_e_f = Counter()
    counts_e = Counter()
    pi_counts = Counter() # (i, I)
    pi_denom = Counter() # I
    xi_sums = Counter() # (i, i_p, I)
    gamma_sums = Counter() # (i_p, I)
    ll = 0
    for e_toks, f_toks in corpus:
        I = len(e_toks)
        J = len(f_toks)
        al_prob = jump_params[I] # alignment probabilities are precomputed for each sentence length I

        # initialize alphas, betas and scaling coefficients
        alphas = np.zeros((J, 2*I))
        betas = np.zeros((J, 2*I))
        scale_coeffs = np.zeros(J)

        # f_0, special initialization for alphas
        # also compute lexical probabilities for f_0
        f_0 = f_toks[0]
        t_f_e_0 = trans_prob[0][f_0]
        t_denom_f0 = np.sum([trans_prob[k_tok][f_0] for k_tok in e_toks]) + I * t_f_e_0
        for i, e_tok in enumerate(e_toks):
            alphas[0][i] = trans_prob[e_tok][f_0] * start_prob[I][i]
            delta_t = trans_prob[e_tok][f_0] / t_denom_f0
            counts_e_f[(e_tok, f_0)] += delta_t
            counts_e[e_tok] += delta_t

        # initialize alphas for NULL words
        for i in range(I, 2*I):
            alphas[0][i] = t_f_e_0 * p_0
        # lexical translation deltas for NULL words
        delta_t = trans_prob[0][f_0] / t_denom_f0
        counts_e_f[(0, f_0)] += delta_t * I
        counts_e[0] += delta_t * I

        # rescale alphas and store scaling coefficient
        Z = np.sum(alphas[0])
        alphas[0] = alphas[0] / Z
        scale_coeffs[0] = Z

        # iterate over f_2, ..., f_J
        for j_, f_tok in enumerate(f_toks[1:]):
            j = j_ + 1
            t_denom_f_tok = np.sum([trans_prob[k_tok][f_tok] for k_tok in e_toks])  \
                            + I * trans_prob[0][f_tok]
            for i, e_tok in enumerate(e_toks + [0] * I):
                t_f_e = trans_prob[e_tok][f_tok]
                if e_tok == 0:
                    # i is NULL
                    # go to NULL_i is only possible from NULL_i or from NULL_i - I
                    alphas[j][i] = t_f_e * (alphas[j - 1][i-I] + alphas[j-1][i]) * p_0
                else:
                    # i is not null
                    alphas[j][i] = t_f_e * (np.sum([alphas[j - 1][k] * al_prob[k][i] for k in xrange(I)]) +
                                            np.sum([alphas[j-1][k] *
                                                                al_prob[k-I][i] for k in range(I, 2*I)]))
                # lexical translation parameters
                delta_t = t_f_e / t_denom_f_tok
                counts_e_f[(e_tok, f_tok)] += delta_t
                counts_e[e_tok] += delta_t

            # rescale alphas for numerical stability
            Z = np.sum(alphas[j])
            alphas[j] = alphas[j] / Z
            scale_coeffs[j] = Z


        # betas
        # special initialization for last betas
        betas[J-1] = np.ones(I*2)

        # other betas
        for j, f_tok in reversed(list(enumerate(f_toks[:-1]))):
            for i, e_tok in enumerate(e_toks + [0] * I):
                if e_tok == 0:
                    # i is NULL
                    betas[j][i] = np.sum([betas[j+1][k] * trans_prob[e_k][f_toks[j+1]] *
                                              al_prob[i-I][k] for k, e_k in enumerate(e_toks)]) +\
                                      (betas[j+1][i] * trans_prob[0][f_toks[j+1]] *
                                              p_0)
                else:
                    # i is not NULL
                    betas[j][i] = np.sum([betas[j+1][k] * trans_prob[e_k][f_toks[j+1]] *
                                              al_prob[i][k] for k, e_k in enumerate(e_toks)]) + \
                                      (betas[j+1][i+I] * trans_prob[0][f_toks[j+1]] *
                                              p_0)

            # rescale betas for numerical stability
            betas[j] = betas[j] / scale_coeffs[j+1]

        gammas = np.multiply(alphas, betas)

        for j in xrange(1, J):
            t_f_e = np.array([trans_prob[e_tok][f_toks[j]] for e_tok in e_toks + [0]*I])
            beta_t_j_i = np.multiply(betas[j], t_f_e)
            j_p = j-1
            alpha_j_p = alphas[j_p]
            for i_p in range(2*I):
                alpha_j_p_i_p = alpha_j_p[i_p]
                if i_p < I:
                    gamma_sums[i_p] += gammas[j_p][i_p]
                else:
                    gamma_sums[i_p-I] += gammas[j_p][i_p]
                for i in range(I):
                    if i_p < I:
                        xi = (al_prob[i_p][i]  * alpha_j_p_i_p * beta_t_j_i[i]) / scale_coeffs[j]
                        xi_sums[(i,i_p)] += xi
                    else:
                        xi = (al_prob[i_p-I][i]  * alpha_j_p_i_p * beta_t_j_i[i]) / scale_coeffs[j]
                        xi_sums[(i,i_p-I)] += xi
        # add start counts
        for i in range(2*I):
            pi_counts[(I, i)] += gammas[0][i]
        pi_denom[I] += 1

        ll += np.sum(np.log(scale_coeffs))

    results.put([counts_e_f, counts_e, gamma_sums, xi_sums, pi_counts, pi_denom, ll])


def load_probs(trans_probs):
    probs = dict()
    for (cond, x), v in trans_probs.iteritems():
        if cond not in probs:
            probs[cond] = dict()
        probs[cond][x] = v
    return probs


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-e", required=True)
arg_parser.add_argument("-f", required=True)
arg_parser.add_argument("-params", required=True)
arg_parser.add_argument("-num_workers", required=False, type=int, default=1)
arg_parser.add_argument("-p_0", required=False, type=float, default=0.2)

args = arg_parser.parse_args()

e_file = args.e
f_file = args.f
params_file = args.params
output_exp_file = params_file + ".counts"
num_workers = args.num_workers
p_0 = args.p_0

corpus = Corpus_Reader(e_file, f_file)
trans_params, jump_params, start_params = pickle.load(open(params_file, "rb"))

trans_prob = load_probs(trans_params)
start_prob = load_probs(start_params)


corpus = list(corpus)
n= int(np.ceil(float(len(corpus)) / num_workers))
data = [corpus[i:i+n] for i in range(0, len(corpus), n)]

results = mp.Queue()

processes = [mp.Process(target=train_iteration, args=(data[i], trans_prob, jump_params, start_prob, p_0, results)) for i in xrange(num_workers)]
for p in processes:
    p.start()

initial = results.get()
total = initial[:-1]
total_ll = initial[-1]

for p in processes[1:]:
    counts = results.get()
    for i, c in enumerate(counts[:-1]):
        total[i].update(c)
    total_ll += counts[-1]
for p in processes:
    a = p.join()


expectations = {'counts_e_f': total[0], 'counts_e':total[1], 'gamma_sums': total[2], 'xi_sums': total[3],
                'pi_counts':total[4], 'pi_denom':total[5], 'll':total_ll}

pickle.dump(expectations, open(output_exp_file, "wb"))