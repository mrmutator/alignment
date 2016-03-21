from __future__ import division
from word_alignment.utils.Corpus_Reader import Corpus_Reader
from collections import Counter
import numpy as np
import cPickle as pickle
import multiprocessing as mp
import argparse
import hmm

def train_iteration(corpus, trans_prob, dist_params, start_params, p_0, results):
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
        I_double = 2 * I
        J = len(f_toks)
        start_prob = np.hstack((start_params[I], np.ones(I) * (p_0/I)))

        tmp = np.hstack((dist_params[I], np.identity(I)))
        dist_mat = np.vstack((tmp, tmp))

        # initialize alphas, betas and scaling coefficients

        alphas = np.zeros((J, I_double))
        betas = np.ones((J, I_double))
        scale_coeffs = np.zeros(J)

        trans_mat = np.zeros((I_double, J))
        for j, f_tok in enumerate(f_toks):
            for i, e_tok in enumerate(e_toks + [0] * I):
                trans_mat[i, j] = trans_prob[(e_tok,f_tok)]

        hmm.forward(J, I_double, start_prob, dist_mat, trans_mat, alphas, scale_coeffs)
        hmm.backward(J, I_double, dist_mat, trans_mat, betas, scale_coeffs)

        gammas = np.multiply(alphas, betas)

        # update counts

        # add start counts and counts for lex f_0
        f_0 = f_toks[0]
        for i, e_tok in enumerate(e_toks):
            start_counts[(I, i)] += gammas[0][i]
            lex_counts[(e_tok, f_0)] += gammas[0][i]
            lex_norm[e_tok] += gammas[0][i]
        start_norm[I] += 1
        zero_sum = np.sum(gammas[0][I:])
        lex_counts[(0, f_0)] += zero_sum
        lex_norm[0] += zero_sum

        for j_p, f_tok in enumerate(f_toks[1:]):
            j = j_p + 1
            t_f_e = np.array([trans_prob[(e_tok,f_toks[j])] for e_tok in e_toks + [0]*I]) # array of t(f_j|e) for all e
            beta_t_j_i = np.multiply(betas[j], t_f_e)
            alpha_j_p = alphas[j_p]
            gammas_0_j = np.sum(gammas[j][I:])
            lex_counts[(0, f_tok)] += gammas_0_j
            lex_norm[0] += gammas_0_j
            for i, e_tok in enumerate(e_toks):
                lex_counts[(e_tok, f_tok)] += gammas[j][i]
                lex_norm[e_tok] += gammas[j][i]
                for i_p in range(I_double):
                    if i_p < I:
                        al_prob_ip = dist_params[I][i_p]
                        actual_i_p = i_p
                    else:
                        al_prob_ip = dist_params[I][i_p-I]
                        actual_i_p = i_p - I
                    xi = (al_prob_ip[i]  * alpha_j_p[i_p] * beta_t_j_i[i]) / scale_coeffs[j]
                    al_counts[(actual_i_p, i)] += xi
                    al_norm[actual_i_p] += gammas[j_p][i_p]


        ll += np.sum(np.log(scale_coeffs))

    results.put([lex_counts, lex_norm, al_norm, al_counts, start_counts, start_norm, ll])



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
trans_params, dist_params, start_params = pickle.load(open(params_file, "rb"))

corpus = list(corpus)
n= int(np.ceil(float(len(corpus)) / num_workers))
data = [corpus[i:i+n] for i in range(0, len(corpus), n)]
num_workers = len(data)

results = mp.Queue()

processes = [mp.Process(target=train_iteration, args=(data[i], trans_params, dist_params, start_params, p_0, results)) for i in xrange(num_workers)]
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