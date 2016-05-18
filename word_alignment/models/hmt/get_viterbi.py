from __future__ import division
import numpy as np
import multiprocessing as mp
import argparse
from CorpusReader import SubcorpusReader
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s  %(message)s')
logger = logging.getLogger()


class Corpus_Buffer(object):
    def __init__(self, corpus, buffer_size=20, limit=0):
        self.buffer_size = buffer_size
        self.corpus = corpus
        self.num = 0
        self.limit = limit

    def __iter__(self):
        self.corpus.reset()
        buffer = []
        c = 0
        for el in self.corpus:
            buffer.append(el)
            c += 1
            if c == self.buffer_size:
                yield self.num, buffer
                self.num += 1
                buffer = []
                c = 0
        if c > 0:
            yield self.num, buffer


def get_all_viterbi_alignments(corpus, t_params, d_params, s_params, alpha, p_0, fertility, queue, worker_num):
    all_alignments = []
    for e_toks, f_toks, f_heads, cons, order in corpus:
        I = len(e_toks)
        J = len(f_toks)
        I_double = 2 * I

        t_probs = {(e_tok, f_tok): t_params.get((e_tok, f_tok), 0.0000001) for f_tok in f_toks for e_tok in
                   e_toks + [0]}

        cons_set = set()
        for con in cons[1:]:
            cons_set.add(con)

        d_probs = dict()
        for p in cons_set:
            tmp_prob = np.zeros((I, I))
            jumps = {j: d_params[p, j] for j in xrange(-I+1, I)}
            if fertility:
                jumps[0] = 0.0
            for i_p in xrange(I):
                norm = np.sum([jumps[i_pp - i_p] for i_pp in xrange(I)]) + p_0 + fertility
                tmp_prob[i_p, :] = np.array(
                    [((jumps[i - i_p] / norm) * (1 - alpha)) + (alpha * (1.0 / I)) if not fertility or  i != i_p
                     else fertility * (1-alpha) + (alpha * (1.0 / I)) for i in xrange(I)])
            tmp = np.hstack((tmp_prob, np.identity(I)*p_0))
            dist_mat = np.vstack((tmp, tmp))
            d_probs[p] = dist_mat


        s_probs = np.hstack((np.array(s_params[I]), np.ones(I)* (float(p_0)/I)))

        e_toks = e_toks + [0] * I
        dependencies = [set() for _ in xrange(J)]
        for j, h in enumerate(f_heads[1:]):
            dependencies[h].add(j+1)

        f = np.zeros((J,I_double))
        best = np.zeros((J, I_double), dtype=int)

        for j in reversed(range(1, J)):
            p = cons[j]
            f_j_in = np.zeros(I_double)
            for dep in dependencies[j]:
                f_j_in += f[dep]

            for i_p in range(I_double):
                values = np.zeros(I_double)
                for i in range(I_double):
                    values[i] = (np.log(t_probs[(e_toks[i], f_toks[j])]) + np.log(d_probs[p][i_p, i])) + f_j_in[i]

                best_i = np.argmax(values)
                best[j][i_p] = best_i
                f[j][i_p] = values[best_i]

        f[0] = np.array([np.log(s_probs[i]) + np.log(t_probs[(e_toks[i], f_toks[0])]) for i in range(I_double)])

        for dep in dependencies[0]:
            f[0] += f[dep]
        last_best = np.argmax(f[0])
        alignment = [int(last_best)]
        for j in range(1, J):
            head = f_heads[j]
            if head == 0:
                alignment.append(best[j][last_best])
            else:
                alignment.append(best[j][alignment[head]])

        alignment = [(al, order[j]) for j, al in enumerate(alignment) if al < I]
        all_alignments.append(alignment)

    queue.put((worker_num, all_alignments))


def load_params(t_params, d_params, s_params, file_name, p_0=0.2):
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


#############################################
# main

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-corpus", required=True)
arg_parser.add_argument("-params", required=True)
arg_parser.add_argument("-out_file", required=True)
arg_parser.add_argument("-num_workers", required=False, type=int, default=1)
arg_parser.add_argument("-p_0", required=False, type=float, default=0.2)
arg_parser.add_argument("-alpha", required=False, type=float, default=0.0)
arg_parser.add_argument("-fertility", required=False, type=float, default=0.0)
arg_parser.add_argument("-buffer_size", required=False, type=int, default=20)
arg_parser.add_argument("-limit", required=False, type=int, default=0)

args = arg_parser.parse_args()

results_queue = mp.Queue()
process_queue = mp.Queue()
manager = mp.Manager()
t_params = manager.dict()
d_params = manager.dict()
s_params = manager.dict()


def worker_wrapper(process_queue):
    while True:
        worker_num, buffer = process_queue.get()
        get_all_viterbi_alignments(buffer, t_params, d_params, s_params, args.alpha, args.p_0, args.fertility,
                                   results_queue, worker_num)


corpus = SubcorpusReader(args.corpus, limit=args.limit, return_order=True)
corpus_length = corpus.get_length()
num_work = int(np.ceil(float(corpus_length) / args.buffer_size))

pool = mp.Pool(args.num_workers, worker_wrapper, (process_queue,))
logger.info("Loading parameters.")
load_params(t_params, d_params, s_params, args.params, p_0=args.p_0)

corpus_buffer = Corpus_Buffer(corpus, buffer_size=args.buffer_size)
logger.info("Loading corpus.")
for num_buffer in corpus_buffer:
    process_queue.put(num_buffer)
logger.info("Finish processing.")

results = [None] * num_work
for _ in xrange(num_work):
    i, als = results_queue.get()
    results[i] = als
logger.info("Writing alignment file.")
outfile = open(args.out_file, "w")
for group in results:
    for als in group:
        als = [str(al[0]) + "-" + str(al[1]) for al in als]
        outfile.write(" ".join(als) + "\n")
outfile.close()
logger.info("Done.")