from Corpus_Reader import Corpus_Reader
from collections import defaultdict, Counter
import numpy as np
import cPickle as pickle
import multiprocessing as mp
import argparse

def train_iteration(corpus, trans_prob, al_prob, results):
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
        alphas = np.zeros((J, I))
        betas = np.zeros((J, I))
        scale_coeffs = np.zeros(J)
        for j, f_tok in enumerate(f_toks):
            for i, e_tok in enumerate(e_toks):
                t_f_e = trans_prob[e_tok][f_tok]
                if j == 0:
                    alphas[j][i] = t_f_e * al_prob[(None, I)][i]
                else:
                    alphas[j][i] = t_f_e * np.sum([alphas[j - 1][k] * al_prob[I][i - k] for k in xrange(I)])

                # lexical translation parameters
                delta_t = t_f_e / np.sum([trans_prob[k_tok][f_tok] for k_tok in e_toks])
                counts_e_f[(e_tok, f_tok)] += delta_t
                counts_e[e_tok] += delta_t
            # rescale alphas for numerical stability
            Z = np.sum(alphas[j])
            alphas[j] = alphas[j] / Z
            scale_coeffs[j] = Z

        for j, f_tok in reversed(list(enumerate(f_toks))):
            for i, e_tok in enumerate(e_toks):
                if j == J-1:
                    betas[j][i] = 1
                else:
                    betas[j][i] = np.sum([betas[j+1][k] * trans_prob[e_k][f_toks[j+1]] * al_prob[I][k-i] for k, e_k in enumerate(e_toks)])
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
            t_f_e = np.array([trans_prob[e_tok][f_toks[j]] for e_tok in e_toks])
            beta_t_j_i = np.multiply(betas[j], t_f_e)
            j_p = j-1
            alpha_j_p = alphas[j_p]
            # denom = np.sum(np.multiply(alpha_j_p, betas[j_p]))
            for i_p in range(I):
                alpha_j_p_i_p = alpha_j_p[i_p]
                gamma_sums[(i_p, I)] += gammas[j_p][i_p]
                for i in range(I):
                    # xi = (al_prob[I][i - i_p]  * alpha_j_p_i_p * beta_t_j_i[i]) / denom
                    xi = (al_prob[I][i - i_p]  * alpha_j_p_i_p * beta_t_j_i[i]) / scale_coeffs[j]
                    xi_sums[(i,i_p, I)] += xi
        # add counts
        for i in range(I):
            pi_counts[(i, I)] += gammas[0][i]
        pi_denom[I] += 1

        # ll += np.log(np.sum(alphas[J-1]))
        ll += np.sum(np.log(scale_coeffs))

    print ll
    results.put([counts_e_f, counts_e, gamma_sums, xi_sums, pi_counts, pi_denom, ll])


def load_probs(trans_probs):
    probs = dict()
    for (cond,x), v in trans_probs.iteritems():
        if cond not in probs:
            probs[cond] = dict()
        probs[cond][x] = v
    return probs


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-e", required=True)
arg_parser.add_argument("-f", required=True)
arg_parser.add_argument("-params", required=True)
arg_parser.add_argument("-num_workers", required=False, type=int, default=1)

args = arg_parser.parse_args()

e_file = args.e
f_file = args.f
params_file = args.params
output_exp_file = params_file + ".counts"
num_workers = 6

corpus = Corpus_Reader(e_file, f_file)
trans_params, al_params = pickle.load(open(params_file, "rb"))

trans_prob = load_probs(trans_params)
al_prob = load_probs(al_params)

corpus = list(corpus)
n= int(np.ceil(float(len(corpus)) / num_workers))
data = [corpus[i:i+n] for i in range(0, len(corpus), n)]

results = mp.Queue()

processes = [mp.Process(target=train_iteration, args=(data[i], trans_prob, al_prob, results)) for i in xrange(num_workers)]
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