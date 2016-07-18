from __future__ import division
import numpy as np
import multiprocessing as mp
import argparse
from CorpusReader import SubcorpusReader
import logging
from feature_model_worker import load_vecs, load_params, load_weights
from scipy.sparse import lil_matrix

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s  %(message)s')
logger = logging.getLogger()

def get_viterbi_alignment(process_queue, queue):
    dist_vecs = dict(vec_ids)
    dist_weights = np.array(d_weights)
    p_0 = args.p_0
    norm_coeff = 1.0 - p_0
    feature_dim = len(dist_weights)
    SMALL_PROB_CONST = 0.0000001


    while True:
        buffer = process_queue.get()
        if buffer is None:
            return

        e_toks, f_toks, f_heads, feature_ids, t_probs, pair_num = buffer
        I = len(e_toks)
        J = len(f_toks)
        I_double = 2 * I

        # start probs
        # i_p is 0 for start_probs
        feature_matrix = lil_matrix((I, feature_dim))
        for i in xrange(I):
            features_i =  dist_vecs[feature_ids[0][0][1][i]]
            feature_matrix.rows[i] = features_i
            feature_matrix.data[i] = [1.0] * len(features_i)
        feature_matrix = feature_matrix.tocsr()
        numerator = np.exp(feature_matrix.dot(dist_weights))
        s_probs = (numerator / np.sum(numerator)) * norm_coeff
        start_prob = np.hstack((s_probs, np.ones(I) * (p_0 / I)))

        # dist probs
        d_probs = np.zeros((J-1, I_double, I_double))
        tmp = np.hstack((np.zeros((I, I)), np.identity(I) * p_0))
        template = np.vstack((tmp, tmp))
        for j in xrange(1, J):
            for i_p in xrange(I):
                feature_matrix = lil_matrix((I, feature_dim))
                for i in xrange(I):
                    features_i = dist_vecs[feature_ids[j][i_p][1][i]]
                    feature_matrix.rows[i] = features_i
                    feature_matrix.data[i] = [1.0] * len(features_i)
                feature_matrix = feature_matrix.tocsr()
                num = np.exp(feature_matrix.dot(dist_weights))
                d_probs[j-1, i_p, :I] = num
                d_probs[j-1, i_p +I, :I] = num

        d_probs = ((d_probs / np.sum(d_probs, axis=2)[:, :, np.newaxis]) * norm_coeff) + template

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
                    values[i] = (np.log(t_probs.get((e_toks[i], f_toks[j]), SMALL_PROB_CONST)) + np.log(d_probs[j-1][i_p, i])) + \
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

        queue.put((pair_num, alignment))

def aggregate_alignments(queue):
    outfile = open(args.out_file, "w")
    buffer_dict = dict()
    next_c = 0
    while True:
        if next_c in buffer_dict:
            alignment = buffer_dict[next_c]
            alignment = [str(al[0]) + "-" + str(al[1]) for al in alignment]
            outfile.write(" ".join(alignment) + "\n")
            del buffer_dict[next_c]
            next_c += 1
            continue

        obj = queue.get()
        if obj is None:
            break
        num, alignment = obj
        if num == next_c:
            alignment = [str(al[0]) + "-" + str(al[1]) for al in alignment]
            outfile.write(" ".join(alignment) + "\n")
            next_c += 1
        else:
            buffer_dict[num] = alignment

    while len(buffer_dict) > 0:
        alignment = buffer_dict[next_c]
        alignment = [str(al[0]) + "-" + str(al[1]) for al in alignment]
        outfile.write(" ".join(alignment) + "\n")
        del buffer_dict[next_c]
        next_c += 1

    outfile.close()

#############################################
# main
if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-corpus", required=True)
    arg_parser.add_argument("-params", required=True)
    arg_parser.add_argument("-vecs", required=True)
    arg_parser.add_argument("-weights", required=True)
    arg_parser.add_argument("-out_file", required=True)
    arg_parser.add_argument("-num_workers", required=False, type=int, default=1)
    arg_parser.add_argument("-p_0", required=False, type=float, default=0.2)
    arg_parser.add_argument("-buffer_size", required=False, type=int, default=20)
    arg_parser.add_argument("-limit", required=False, type=int, default=0)

    args = arg_parser.parse_args()
    num_workers = max(1, args.num_workers-1)

    results_queue = mp.Queue()
    process_queue = mp.Queue(maxsize=num_workers)

    vec_ids = load_vecs(args.vecs)
    d_weights = load_weights(args.weights)

    corpus = SubcorpusReader(args.corpus, limit=args.limit)
    pool = []
    for w in xrange(num_workers):
        p = mp.Process(target=get_viterbi_alignment, args=(process_queue, results_queue))
        p.start()
        pool.append(p)

    aggregator = mp.Process(target=aggregate_alignments, args=(results_queue,))
    aggregator.start()

    logger.info("Loading parameters.")
    t_params = load_params(args.params)


    logger.info("Loading corpus.")

    for pair_num, (e_toks, f_toks, f_heads, feature_ids) in enumerate(corpus):
        # get all t-params of buffer
        required_ts = set()
        I = len(e_toks)
        for e_tok in e_toks + [0]:
            for f_tok in f_toks:
                required_ts.add((e_tok, f_tok))

        t_probs = {ef: t_params[ef] for ef in required_ts if ef in t_params}
        process_queue.put((e_toks, f_toks, f_heads, feature_ids, t_probs, pair_num))

    logger.info("Entire corpus loaded.")
    for _ in pool:
        process_queue.put(None)

    for p in pool:
        p.join()

    results_queue.put(None)
    aggregator.join()

    logger.info("Done.")
