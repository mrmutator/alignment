from __future__ import division
from utils.Corpus_Reader import Corpus_Reader
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
    for e_toks, f_pairs in corpus:
        I = len(e_toks)
        J = len(f_pairs)

        # iterate over f_1, ..., f_J
        for j, (f_tok, f_head) in enumerate(f_pairs):
            t_denom_f_tok = np.sum([trans_prob[k_tok][f_tok] for k_tok in [0] + e_toks])
            for i, e_tok in enumerate([0] + e_toks):
                t_f_e = trans_prob[e_tok][f_tok]
                delta_t = t_f_e / t_denom_f_tok
                counts_e_f[(e_tok, f_tok)] += delta_t
                counts_e[e_tok] += delta_t



        # ll += np.sum(np.log(scale_coeffs))

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

corpus = Corpus_Reader(e_file, f_file, f_)
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