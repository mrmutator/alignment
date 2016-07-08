from __future__ import division
from collections import Counter
import numpy as np
import multiprocessing as mp
import argparse
from CorpusReader import SubcorpusReader, Corpus_Buffer
import logging
import hmt
from scipy.sparse import lil_matrix

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s  %(message)s')
logger = logging.getLogger()


def train_iteration(buffer, p_0, dist_cons, dist_weights, queue):
    # set all counts to zero
    lex_counts = Counter()  # (e,f)
    lex_norm = Counter()  # e
    al_counts = Counter()  # (static_cond, dynamic_cond)
    ll = 0
    norm_coeff = 1.0 - p_0
    SMALL_PROB_CONST = 0.00000001

    corpus, trans_params = buffer
    feature_dim = len(dist_weights)

    for (e_toks, f_toks, f_heads, feature_ids) in corpus:
        I = len(e_toks)
        I_double = 2 * I
        J = len(f_toks)


        translation_matrix = np.zeros((J, I_double))
        # start probs
        # i_p is 0 for start_probs
        feature_matrix = lil_matrix((I, feature_dim))
        f_0 = f_toks[0]
        for i in xrange(I):
            translation_matrix[0][i] = trans_params.get((e_toks[i], f_0), SMALL_PROB_CONST) # abuse for loop
            features_i =  dist_cons[feature_ids[0][0][1][i]]
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
        translation_matrix[0][I:] = trans_params.get((0, f_0), SMALL_PROB_CONST) # null word for first word
        for j in xrange(1, J):
            f_j = f_toks[j]
            translation_matrix[j][I:] = trans_params.get((0, f_j), SMALL_PROB_CONST)
            #temp_probs = np.zeros((I_double, I_double))
            for i_p in xrange(I):
                feature_matrix = lil_matrix((I, feature_dim))
                translation_matrix[j][i_p] = trans_params.get((e_toks[i_p], f_j), SMALL_PROB_CONST)
                for i in xrange(I):
                    features_i =  dist_cons[feature_ids[j][i_p][1][i]]
                    feature_matrix.rows[i] = features_i
                    feature_matrix.data[i] = [1.0] * len(features_i)
                feature_matrix = feature_matrix.tocsr()
                num = np.exp(feature_matrix.dot(dist_weights))
                d_probs[j-1, i_p, :I] = num
                d_probs[j-1, i_p +I, :I] = num

        d_probs = ((d_probs / np.sum(d_probs, axis=2)[:, :, np.newaxis]) * norm_coeff) + template

        gammas, xis, pair_ll = hmt.upward_downward(J, I_double, f_heads, translation_matrix, d_probs,
                                                   start_prob)

        # update counts

        # add start counts and counts for lex f_0
        for j, f_tok in enumerate(f_toks):
            if (0, f_tok) in trans_params:
                gammas_0_j = np.sum(gammas[j][I:])
                lex_counts[(0, f_tok)] += gammas_0_j
                lex_norm[0] += gammas_0_j
            for i, e_tok in enumerate(e_toks):
                if (e_tok, f_tok) in trans_params:
                    lex_counts[(e_tok, f_tok)] += gammas[j][i]
                    lex_norm[e_tok] += gammas[j][i]
                if j == 0:
                    static_cond = feature_ids[j][0][0]
                    dynamic_cond = feature_ids[j][0][1][i]
                    al_counts[(static_cond, dynamic_cond)] += gammas[0][i]
                    continue

                for i_p in range(I_double):
                    if i_p < I:
                        actual_i_p = i_p
                    else:
                        actual_i_p = i_p - I
                    static_cond = feature_ids[j][actual_i_p][0]
                    dynamic_cond = feature_ids[j][actual_i_p][1][i]
                    al_counts[(static_cond, dynamic_cond)] += xis[j][i_p][i]

        ll += pair_ll
    queue.put((lex_counts, lex_norm, al_counts, ll))


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
            t_params[(e, f)] = p
        else:
            raise Exception("Should not happen.")
    infile.close()
    return t_params


def load_cons(file_name):
    cond_ids = dict()
    infile = open(file_name, "r")
    for line in infile:
        els = line.strip().split()
        cid = int(els[0])
        feature_ids = map(int, els[1:])
        cond_ids[cid] = frozenset(feature_ids)
    infile.close()
    return cond_ids


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
    arg_parser.add_argument("-cons", required=True)
    arg_parser.add_argument("-weights", required=True)
    arg_parser.add_argument("-num_workers", required=False, type=int, default=2)
    arg_parser.add_argument("-p_0", required=False, type=float, default=0.2)
    arg_parser.add_argument("-buffer_size", required=False, type=int, default=20)

    args = arg_parser.parse_args()

    counts_file_name = args.params + ".counts"

    update_queue = mp.Queue()
    process_queue = mp.Queue(maxsize=args.num_workers - 1)

    cond_ids = load_cons(args.cons)
    d_weights = load_weights(args.weights)


    def worker_wrapper(process_queue):
        cons = dict(cond_ids)
        weights = np.array(d_weights)
        while True:
            buffer = process_queue.get()
            if buffer is None:
                return
            train_iteration(buffer, args.p_0, cons, weights, update_queue)


    corpus = SubcorpusReader(args.corpus)
    pool = []
    for w in xrange(args.num_workers - 1):
        p = mp.Process(target=worker_wrapper, args=(process_queue,))
        p.start()
        pool.append(p)

    logger.info("Loading parameters.")
    t_params = load_params(args.params)

    updater = mp.Process(target=aggregate_counts, args=(update_queue, counts_file_name))
    updater.start()

    corpus_buffer = Corpus_Buffer(corpus, buffer_size=args.buffer_size)
    logger.info("Starting worker processes..")

    for buff in corpus_buffer:
        # get all t-params of buffer
        required_ts = set()
        for (e_toks, f_toks, f_heads, feature_ids) in buff:
            I = len(e_toks)
            for e_tok in e_toks + [0]:
                for f_tok in f_toks:
                    required_ts.add((e_tok, f_tok))

        t_probs = {ef: t_params[ef] for ef in required_ts if ef in t_params}
        process_queue.put((buff, t_probs))
    # Send termination signal
    for _ in xrange(args.num_workers - 1):
        process_queue.put(None)
    logger.info("Entire corpus loaded.")
    for p in pool:
        p.join()
    # Send termination signal
    update_queue.put(None)
    logger.info("Waiting for update process to terminate.")
    updater.join()
