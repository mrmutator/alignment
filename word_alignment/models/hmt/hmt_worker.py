from __future__ import division
from collections import Counter
import numpy as np
import multiprocessing as mp
import argparse
from CorpusReader import SubcorpusReader
import logging
import hmt

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s  %(message)s')
logger = logging.getLogger()


class Corpus_Buffer(object):
    def __init__(self, corpus, buffer_size=20):
        self.buffer_size = buffer_size
        self.corpus = corpus

    def __iter__(self):
        self.corpus.reset()
        buffer = []
        c = 0
        for el in self.corpus:
            buffer.append(el)
            c += 1
            if c == self.buffer_size:
                yield buffer
                buffer = []
                c = 0
        if c > 0:
            yield buffer


def train_iteration(corpus, t_params_s, d_params_s, s_params_s, alpha, p_0, queue):
    # set all counts to zero
    lex_counts = Counter()  # (e,f)
    lex_norm = Counter()  # e
    start_counts = Counter()  # (I, i)
    start_norm = Counter()  # I
    al_counts = Counter()  # (i_p, i)
    al_norm = Counter()  # (i_p)
    ll = 0
    start_norm_coeff = 1.0 - p_0
    # get all t-params of buffer
    required_ts= set()
    required_Is = set()
    required_cons_j = set()
    for e_toks, f_toks, f_heads, cons in corpus:
        for e_tok in e_toks + [0]:
            for f_tok in f_toks:
                required_ts.add((e_tok, f_tok))
        I = len(e_toks)
        required_Is.add(I)
        for con in cons[1:]:
            for jmp in xrange(-I +1, I):
                required_cons_j.add((con, jmp))

    # get a copy from shared dicts
    trans_params = {ef: t_params_s[ef] for ef in required_ts if ef in t_params_s}
    s_params = {I_: np.copy(s_params_s[I_]) for I_ in required_Is}
    d_params = {cj: d_params_s[cj] for cj in required_cons_j}

    for e_toks, f_toks, f_heads, cons in corpus:
        I = len(e_toks)
        I_double = 2 * I

        s_probs = s_params[I]
        start_prob = np.hstack((s_probs, np.ones(I) * (p_0 / I)))
        cons_set = set()
        for con in cons[1:]:
            cons_set.add(con)

        d_probs = dict()
        for p in cons_set:
            tmp_prob = np.zeros((I, I))
            jumps = {j: d_params[p, j] for j in xrange(-I+1, I)}
            for i_p in xrange(I):
                norm = np.sum([jumps[i_pp - i_p] for i_pp in xrange(I)]) + p_0
                tmp_prob[i_p, :] = np.array(
                    [((jumps[i - i_p] / norm) * (1 - alpha)) + (alpha * (1.0 / I)) for i in xrange(I)])
            tmp = np.hstack((tmp_prob, np.identity(I)*p_0))
            dist_mat = np.vstack((tmp, tmp))
            d_probs[p] = dist_mat


        gammas, xis, pair_ll = hmt.upward_downward(f_toks, e_toks + [0] * I, f_heads, cons, trans_params, d_probs,
                                                       start_prob)

        # update counts

        # add start counts and counts for lex f_0
        f_0 = f_toks[0]
        for i, e_tok in enumerate(e_toks):
            start_counts[(I, i)] += gammas[0][i] * start_norm_coeff
            start_norm[I] += gammas[0][i]
            if (e_tok, f_0) in trans_params:
                lex_counts[(e_tok, f_0)] += gammas[0][i]
                lex_norm[e_tok] += gammas[0][i]
        if (0, f_0) in trans_params:
            zero_sum = np.sum(gammas[0][I:])
            lex_counts[(0, f_0)] += zero_sum
            lex_norm[0] += zero_sum

        for j_p, f_tok in enumerate(f_toks[1:]):
            j = j_p + 1
            p = cons[j]
            if (0, f_tok) in trans_params:
                gammas_0_j = np.sum(gammas[j][I:])
                lex_counts[(0, f_tok)] += gammas_0_j
                lex_norm[0] += gammas_0_j
            for i, e_tok in enumerate(e_toks):
                if (e_tok, f_tok) in trans_params:
                    lex_counts[(e_tok, f_tok)] += gammas[j][i]
                    lex_norm[e_tok] += gammas[j][i]
                for i_p in range(I_double):
                    if i_p < I:
                        actual_i_p = i_p
                    else:
                        actual_i_p = i_p - I
                    al_counts[(p, actual_i_p, i)] += xis[j][i_p][i]
                    al_norm[p, actual_i_p] += gammas[j_p][i_p]

        ll += pair_ll

    queue.put((lex_counts, lex_norm, al_counts, al_norm, start_counts, start_norm, ll))


def load_params(t_params, d_params, s_params, file_name):
    lengths = set()
    temp_start_params = dict()
    infile = open(file_name, "r")
    for line in infile:
        els = line.strip().split(" ")
        p_type = els[0]
        if p_type == "t":
            e = int(els[1])
            f = int(els[2])
            p = float(els[3])
            t_params[(e, f)] = p
        elif p_type == "j":
            pos = int(els[1])
            j = int(els[2])
            p = float(els[3])
            d_params[(pos, j)] = p
        elif p_type == "s":
            I = int(els[1])
            i = int(els[2])
            p = float(els[3])
            temp_start_params[(I, i)] = p
            lengths.add(I)
        else:
            raise Exception("Should not happen.")
    infile.close()


    for I in lengths:
        tmp2_prob = np.zeros(I)
        for i in xrange(I):
            tmp2_prob[i] = temp_start_params[I, i]
        s_params[I] = tmp2_prob




def aggregate_counts(queue, num_work, counts_file):
    # total = [Counter(), Counter(), Counter(), Counter(), Counter(), Counter()]
    # total_ll = 0
    initial = queue.get()
    total = initial[:-1]
    total_ll = initial[-1]
    for _ in xrange(num_work - 1):
        counts = queue.get()
        for i, c in enumerate(counts[:-1]):
            total[i].update(c)
        total_ll += counts[-1]

    logger.info("Writing counts to file.")

    # store counts
    # types = ["lex_counts", "lex_norm", "al_counts", "al_norm", "start_counts", "start_norm"]
    types = map(str, range(6))
    with open(counts_file, "w") as outfile:
        outfile.write("LL:\t" + str(total_ll) + "\n")
        for i, counter in enumerate(total):
            t = types[i]
            for k, v in counter.iteritems():
                if isinstance(k, tuple):
                    k = " ".join(map(str, k))
                else:
                    k = str(k)
                v = str(v)
                outfile.write("\t".join([t, k, v]) + "\n")


#############################################
# main

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-corpus", required=True)
arg_parser.add_argument("-params", required=True)
arg_parser.add_argument("-num_workers", required=False, type=int, default=1)
arg_parser.add_argument("-p_0", required=False, type=float, default=0.2)
arg_parser.add_argument("-alpha", required=False, type=float, default=0.0)
arg_parser.add_argument("-buffer_size", required=False, type=int, default=20)

args = arg_parser.parse_args()

counts_file_name = args.params + ".counts"

update_queue = mp.Queue()
process_queue = mp.Queue()
manager = mp.Manager()
t_params = manager.dict()
d_params = manager.dict()
s_params = manager.dict()


def worker_wrapper(process_queue):
    while True:
        buffer = process_queue.get()
        train_iteration(buffer, t_params, d_params, s_params, args.alpha, args.p_0, update_queue)


corpus = SubcorpusReader(args.corpus)
corpus_length = corpus.get_length()
num_work = int(np.ceil(float(corpus_length) / args.buffer_size))

pool = mp.Pool(args.num_workers - 1, worker_wrapper, (process_queue,))
logger.info("Loading parameters.")
load_params(t_params, d_params, s_params, args.params)

updater = mp.Process(target=aggregate_counts, args=(update_queue, num_work, counts_file_name))
updater.start()

corpus_buffer = Corpus_Buffer(corpus, buffer_size=args.buffer_size)
logger.info("Loading corpus.")
for buffer in corpus_buffer:
    process_queue.put(buffer)
logger.info("Waiting to finish")
updater.join()
