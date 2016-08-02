from __future__ import division
from collections import Counter
import numpy as np
import multiprocessing as mp
import argparse
from CorpusReader import SubcorpusReader
import logging
import hmt
from scipy.sparse import lil_matrix

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s  %(message)s')
logger = logging.getLogger()


def train_iteration(process_queue, queue):
    dist_vecs = dict(vec_ids)
    dist_weights = np.array(d_weights)
    p_0 = args.p_0
    feature_dim = len(dist_weights)
    norm_coeff = 1.0 - p_0
    SMALL_PROB_CONST = 0.00000001
    while True:
        buffer = process_queue.get()
        if buffer is None:
            return

        e_toks, f_toks, f_heads, feature_ids = buffer
        # set all counts to zero
        lex_counts = Counter()  # (e,f)
        lex_norm = Counter()  # e
        al_counts = Counter()  # (static_cond, dynamic_cond)

        I = len(e_toks)
        I_double = 2 * I
        J = len(f_toks)

        translation_matrix = np.zeros((J, I * 2))
        marginals = np.zeros((J, I_double))

        # start probs
        # i_p is 0 for start_probs
        feature_matrix = lil_matrix((I, feature_dim))
        #f_0 = f_toks[0]
        t_params_j = t_params.get(f_toks[0], {})
        translation_matrix[0][I:] = t_params_j.get(0, SMALL_PROB_CONST)
        static = dist_vecs[feature_ids[0]]
        for i in xrange(I):
            translation_matrix[0][i] = t_params_j.get(e_toks[i], SMALL_PROB_CONST)
            features_i = static[i]
            feature_matrix.rows[i] = features_i
            feature_matrix.data[i] = [1.0] * len(features_i)
        feature_matrix = feature_matrix.tocsr()
        numerator = np.exp(feature_matrix.dot(dist_weights))
        s_probs = (numerator / np.sum(numerator)) * norm_coeff
        marginals[0] = np.hstack((s_probs, np.ones(I) * (p_0 / I)))

        # dist probs
        d_probs = np.zeros((J - 1, I_double, I_double))
        tmp = np.hstack((np.zeros((I, I)), np.identity(I) * p_0))
        template = np.vstack((tmp, tmp))
        for j in xrange(1, J):
            #f_j = f_toks[j]
            t_params_j = t_params.get(f_toks[j], {})
            translation_matrix[j][I:] = t_params_j.get(0, SMALL_PROB_CONST)
            for i_p in xrange(I):
                translation_matrix[j][i_p] = t_params_j.get(e_toks[i_p], SMALL_PROB_CONST)
                feature_matrix = lil_matrix((I, feature_dim))
                static = dist_vecs[feature_ids[j][i_p]]
                for i in xrange(I):
                    features_i = static[i-i_p]
                    feature_matrix.rows[i] = features_i
                    feature_matrix.data[i] = [1.0] * len(features_i)
                feature_matrix = feature_matrix.tocsr()
                num = np.exp(feature_matrix.dot(dist_weights))
                d_probs[j - 1, i_p, :I] = num
                d_probs[j - 1, i_p + I, :I] = num

        dist_probs = ((d_probs / np.sum(d_probs, axis=2)[:, :, np.newaxis]) * norm_coeff) + template

        gammas, xis, log_likelihood = hmt.upward_downward(J, I_double, f_heads, translation_matrix, dist_probs,
                                                   marginals)

        # update counts

        # add start counts and counts for lex f_0
        for j_p, f_tok in enumerate(f_toks[1:]):
            j = j_p+1
            gammas_0_j = np.sum(gammas[j][I:])
            if translation_matrix[j, I] > SMALL_PROB_CONST:
                lex_counts[(0, f_tok)] += gammas_0_j
            for i, e_tok in enumerate(e_toks):
                if translation_matrix[j, i] > SMALL_PROB_CONST:
                    lex_counts[(e_tok, f_tok)] += gammas[j][i]
                for i_p in range(I):
                    static_cond = feature_ids[j][i_p]
                    al_counts[(static_cond, i-i_p)] += xis[j][i_p][i] + xis[j][i_p+I][i]

        f_0 = f_toks[0]
        if translation_matrix[0, I] > SMALL_PROB_CONST:
            lex_counts[(0, f_0)] += np.sum(gammas[0][I:])
        e_norm = np.sum(gammas, axis=0)
        static_cond = feature_ids[0]
        for i, e_tok in enumerate(e_toks):
            lex_norm[e_tok] += e_norm[i]
            al_counts[(static_cond, i)] += gammas[0][i]
            if translation_matrix[0, i] > SMALL_PROB_CONST:
                lex_counts[(e_tok, f_0)] += gammas[0][i]
        lex_norm[0] += np.sum(e_norm[I:])

        queue.put((lex_counts, lex_norm, al_counts, log_likelihood))


def load_params(file_name):
    t_params = dict()
    infile = open(file_name, "r")
    for line in infile:
        els = line.strip().split(" ")
        p_type = els[0]
        if p_type == "t":
            e = int(els[1])
            f = int(els[2])
            p = float(els[3])
            if f not in t_params:
                t_params[f] = dict()
            t_params[f][e] = p
        else:
            raise Exception("Should not happen.")
    infile.close()
    return t_params


def load_vecs(file_name):
    vec_ids = dict()
    infile = open(file_name, "r")
    for line in infile:
        els = line.strip().split()
        jmp, cid = els[0].split(".")
        if cid not in vec_ids:
            vec_ids[cid] = dict()

        vec_ids[cid][int(jmp)] = sorted(map(int, els[1:]))
    infile.close()
    return vec_ids


def load_weights(file_name):
    d_weights = []
    with open(file_name, "r") as infile:
        for line in infile:
            _, w_id, w = line.strip().split()
            d_weights.append(float(w))

    return np.array(d_weights)


def aggregate_counts(queue, counts_file):
    # total = [Counter(), Counter(), Counter(), Counter(), Counter(), Counter()]
    # total_ll = 0
    initial = queue.get()
    total = initial[:-1]
    total_ll = initial[-1]
    while True:
        counts = queue.get()
        if counts is None:
            break
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
    logger.info("Counts written.")


#############################################
# main
if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-corpus", required=True)
    arg_parser.add_argument("-params", required=True)
    arg_parser.add_argument("-vecs", required=True)
    arg_parser.add_argument("-weights", required=True)
    arg_parser.add_argument("-num_workers", required=False, type=int, default=2)
    arg_parser.add_argument("-p_0", required=False, type=float, default=0.2)
    SMALL_PROB_CONST = 0.00000001

    args = arg_parser.parse_args()

    counts_file_name = args.params + ".counts"

    update_queue = mp.Queue()
    num_workers = max(1, args.num_workers - 1)
    updater = mp.Process(target=aggregate_counts, args=(update_queue, counts_file_name))
    updater.start()
    process_queue = mp.Queue(maxsize=num_workers)

    vec_ids = load_vecs(args.vecs)
    d_weights = load_weights(args.weights)

    logger.info("Loading parameters.")
    t_params = load_params(args.params)

    pool = []
    for w in xrange(num_workers):
        p = mp.Process(target=train_iteration, args=(process_queue, update_queue))
        p.start()
        pool.append(p)

    logger.info("Starting worker processes..")
    corpus = SubcorpusReader(args.corpus)
    for buff in corpus:
        process_queue.put(buff)
    # Send termination signal
    for _ in xrange(num_workers):
        process_queue.put(None)
    logger.info("Entire corpus loaded.")
    for p in pool:
        p.join()
    # Send termination signal
    update_queue.put(None)
    logger.info("Waiting for update process to terminate.")
    updater.join()
