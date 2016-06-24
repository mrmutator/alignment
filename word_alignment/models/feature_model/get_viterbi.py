from __future__ import division
import numpy as np
import multiprocessing as mp
import argparse
from CorpusReader import SubcorpusReader, Corpus_Buffer
import logging
from feature_model_worker import load_cons, load_params, load_weights

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s  %(message)s')
logger = logging.getLogger()

def get_all_viterbi_alignments(buffer, p_0, dist_cons, dist_weights, queue):
    worker_num, corpus, t_probs = buffer
    all_alignments = []
    norm_coeff = 1.0 - p_0
    feature_dim = len(dist_weights)
    SMALL_PROB_CONST = 0.0000001
    for e_toks, f_toks, f_heads, feature_ids in corpus:
        I = len(e_toks)
        J = len(f_toks)
        I_double = 2 * I

        # start probs
        # i_p is 0 for start_probs
        feature_matrix = np.zeros((I, feature_dim))
        for sid in dist_cons[feature_ids[0][0][0]]:  # static feature set
            feature_matrix[:, sid] = 1.0
        for i in xrange(I):
            for did in dist_cons[feature_ids[0][0][1][i]]:  # dynamic feature set
                feature_matrix[i, did] = 1.0

        numerator = np.exp(np.dot(feature_matrix, dist_weights))
        Z = np.sum(numerator)
        s_probs = (numerator / Z) * norm_coeff
        start_prob = np.hstack((s_probs, np.ones(I) * (p_0 / I)))

        # dist probs
        d_probs = dict()
        for j in xrange(1, len(f_toks)):
            tmp_prob = np.zeros((I, I))
            for i_p in xrange(I):
                feature_matrix = np.zeros((I, feature_dim))
                for sid in dist_cons[feature_ids[j][i_p][0]]:
                    feature_matrix[:, sid] = 1.0
                for i in xrange(I):
                    for did in dist_cons[feature_ids[j][i_p][1][i]]:
                        feature_matrix[i, did] = 1.0

                numerator = np.exp(np.dot(feature_matrix, dist_weights))
                Z = np.sum(numerator)
                tmp_prob[i_p] = numerator / Z
            tmp_prob = tmp_prob * norm_coeff
            tmp = np.hstack((tmp_prob, np.identity(I) * p_0))
            dist_mat = np.vstack((tmp, tmp))
            d_probs[j] = dist_mat

        e_toks = e_toks + [0] * I
        dependencies = [set() for _ in xrange(J)]
        for j, h in enumerate(f_heads[1:]):
            dependencies[h].add(j + 1)

        f = np.zeros((J, I_double))
        best = np.zeros((J, I_double), dtype=int)

        for j in reversed(range(1, J)):
            f_j_in = np.zeros(I_double)
            for dep in dependencies[j]:
                f_j_in += f[dep]

            for i_p in range(I_double):
                values = np.zeros(I_double)
                for i in range(I_double):
                    values[i] = (np.log(t_probs.get((e_toks[i], f_toks[j]), SMALL_PROB_CONST)) + np.log(d_probs[j][i_p, i])) + \
                                f_j_in[i]

                best_i = np.argmax(values)
                best[j][i_p] = best_i
                f[j][i_p] = values[best_i]

        f[0] = np.array(
            [np.log(start_prob[i]) + np.log(t_probs.get((e_toks[i], f_toks[0]), SMALL_PROB_CONST)) for i in range(I_double)])

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

        # alignment = [(al, order[j]) for j, al in enumerate(alignment) if al < I]
        alignment = [(al, j) for j, al in enumerate(alignment) if al < I]
        all_alignments.append(alignment)

    queue.put((worker_num, all_alignments))


#############################################
# main
if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-corpus", required=True)
    arg_parser.add_argument("-params", required=True)
    arg_parser.add_argument("-cons", required=True)
    arg_parser.add_argument("-weights", required=True)
    arg_parser.add_argument("-out_file", required=True)
    arg_parser.add_argument("-num_workers", required=False, type=int, default=1)
    arg_parser.add_argument("-p_0", required=False, type=float, default=0.2)
    arg_parser.add_argument("-buffer_size", required=False, type=int, default=20)
    arg_parser.add_argument("-limit", required=False, type=int, default=0)

    args = arg_parser.parse_args()

    results_queue = mp.Queue()
    process_queue = mp.Queue(maxsize=int(np.ceil((args.num_workers) / 4)))

    cond_ids = load_cons(args.cons)
    d_weights = load_weights(args.weights)


    def worker_wrapper(process_queue):
        cons = dict(cond_ids)
        weights = np.array(d_weights)
        while True:
            buffer = process_queue.get()
            if buffer is None:
                return
            get_all_viterbi_alignments(buffer, args.p_0, cons, weights, results_queue)


    corpus = SubcorpusReader(args.corpus, limit=args.limit)
    pool = []
    for w in xrange(args.num_workers):
        p = mp.Process(target=worker_wrapper, args=(process_queue,))
        p.start()
        pool.append(p)

    logger.info("Loading parameters.")
    t_params = load_params(args.params)


    corpus_buffer = Corpus_Buffer(corpus, buffer_size=args.buffer_size)
    logger.info("Loading corpus.")

    for num_buffer, buff in enumerate(corpus_buffer):
        # get all t-params of buffer
        required_ts = set()
        for (e_toks, f_toks, f_heads, feature_ids) in buff:
            I = len(e_toks)
            for e_tok in e_toks + [0]:
                for f_tok in f_toks:
                    required_ts.add((e_tok, f_tok))

        t_probs = {ef: t_params[ef] for ef in required_ts if ef in t_params}
        process_queue.put((num_buffer, buff, t_probs))


    for _ in xrange(args.num_workers):
        process_queue.put(None)

    logger.info("Entire corpus loaded.")

    results = [None] * (num_buffer + 1)
    for _ in xrange(num_buffer+1):
        res = results_queue.get()
        i, als = res
        results[i] = als

    logger.info("Writing alignment file.")
    outfile = open(args.out_file, "w")
    for group in results:
        for als in group:
            als = [str(al[0]) + "-" + str(al[1]) for al in als]
            outfile.write(" ".join(als) + "\n")
    outfile.close()
    logger.info("Done.")
