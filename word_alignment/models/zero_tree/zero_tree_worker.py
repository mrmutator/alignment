from __future__ import division
from word_alignment.utils.Corpus_Reader import Corpus_Reader
from collections import Counter
import numpy as np
import cPickle as pickle
import multiprocessing as mp
import argparse


def train_iteration(corpus, trans_prob, al_prob, results):
    # set all counts to zero
    lex_counts = Counter()
    lex_norm = Counter()
    al_counts = Counter()
    al_norm = Counter()
    ll = 0
    for e_toks, f_pairs in corpus:
        I = len(e_toks)
        J = len(f_pairs)

        # iterate over f_1, ..., f_J
        for j, (f_tok, f_head) in enumerate(f_pairs):
            delta_denom = np.sum([trans_prob[k_tok][f_tok] * al_prob[(I, J, f_head, j)][k] for k, k_tok in enumerate([0] + e_toks)])
            partial_ll = 0
            for i, e_tok in enumerate([0] + e_toks):
                t_f_e = trans_prob[e_tok][f_tok]
                a_i_j_t_j_I = al_prob[(I, J, f_head, j)][i]
                prod_t_a = t_f_e * a_i_j_t_j_I
                partial_ll += prod_t_a
                delta = prod_t_a / delta_denom
                lex_counts[(e_tok, f_tok)] += delta
                lex_norm[e_tok] += delta
                al_counts[(I, J, f_head, j, i)] += delta
                al_norm[(I, J, f_head, j)] += delta


            ll += np.sum(np.log(partial_ll))

    results.put([lex_counts, lex_norm, al_counts, al_norm, ll])


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

corpus = Corpus_Reader(e_file, f_file, source_dep=True)
trans_params, al_params = pickle.load(open(params_file, "rb"))

trans_prob = load_probs(trans_params)
al_prob = load_probs(al_params)


corpus = list(corpus)
n= int(np.ceil(float(len(corpus)) / num_workers))
data = [corpus[i:i+n] for i in range(0, len(corpus), n)]
num_workers = len(data)

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


expectations = {'lex_counts': total[0], 'lex_norm':total[1], 'al_counts': total[2], 'al_norm': total[3], 'll':total_ll}

pickle.dump(expectations, open(output_exp_file, "wb"))