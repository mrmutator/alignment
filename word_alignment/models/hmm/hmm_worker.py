from __future__ import division
from word_alignment.utils.Corpus_Reader import Corpus_Reader
from collections import Counter
import numpy as np
import cPickle as pickle
import multiprocessing as mp
import argparse


def train_iteration(corpus, trans_prob, jump_params, start_prob, p_0, results):
    # set all counts to zero
    lex_counts = Counter() #(e,f)
    lex_norm = Counter() # e
    start_counts = Counter() # (I, i)
    start_norm = Counter() # I
    al_counts = Counter() # (i_p, i)
    al_norm = Counter() # (i_p)
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
        for i, e_tok in enumerate(e_toks):
            alphas[0][i] = trans_prob[e_tok][f_0] * start_prob[I][i]

        # initialize alphas for NULL words
        t_f_e_0 = trans_prob[0][f_0]
        for i in range(I, 2*I):
            alphas[0][i] = t_f_e_0 * p_0

        # rescale alphas[0] and store scaling coefficient
        Z = np.sum(alphas[0])
        alphas[0] = alphas[0] / Z
        scale_coeffs[0] = Z

        # iterate over f_2, ..., f_J
        for j_, f_tok in enumerate(f_toks[1:]):
            j = j_ + 1
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

        # update counts

        # add start counts and counts for lex f_0
        f_0 = f_toks[0]
        for i, e_tok in enumerate(e_toks + [0]*I):
            start_counts[(I, i)] += gammas[0][i]
            lex_counts[(e_tok, f_0)] += gammas[0][i]
            lex_norm[e_tok] += gammas[0][i]
        start_norm[I] += 1

        for j_p, f_tok in enumerate(f_toks[1:]):
            j = j_p + 1
            t_f_e = np.array([trans_prob[e_tok][f_toks[j]] for e_tok in e_toks + [0]*I]) # array of t(f_j|e) for all e
            beta_t_j_i = np.multiply(betas[j], t_f_e)
            alpha_j_p = alphas[j_p]
            gammas_0_j = np.sum(gammas[j][I:])
            lex_counts[(0, f_tok)] += gammas_0_j
            lex_norm[0] += gammas_0_j
            for i, e_tok in enumerate(e_toks):
                lex_counts[(e_tok, f_tok)] += gammas[j][i]
                lex_norm[e_tok] += gammas[j][i]
                for i_p in range(2*I):
                    if i_p < I:
                        al_prob_ip = al_prob[i_p]
                        actual_i_p = i_p
                    else:
                        al_prob_ip = al_prob[i_p-I]
                        actual_i_p = i_p - I
                    xi = (al_prob_ip[i]  * alpha_j_p[i_p] * beta_t_j_i[i]) / scale_coeffs[j]
                    al_counts[(actual_i_p, i)] += xi
                    al_norm[actual_i_p] += gammas[j_p][i_p]


        ll += np.sum(np.log(scale_coeffs))

    results.put([lex_counts, lex_norm, al_norm, al_counts, start_counts, start_norm, ll])


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
num_workers = len(data)

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


expectations = {'lex_counts': total[0], 'lex_norm':total[1], 'al_norm': total[2], 'al_counts': total[3],
                'start_counts':total[4], 'start_norm':total[5], 'll':total_ll}

pickle.dump(expectations, open(output_exp_file, "wb"))