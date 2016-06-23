from __future__ import division
from collections import Counter
import numpy as np
import multiprocessing as mp
import argparse
from CorpusReader import SubcorpusReader, Corpus_Buffer
import logging
import hmt

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s  %(message)s')
logger = logging.getLogger()


def train_iteration(buffer, p_0, queue):
    # set all counts to zero
    lex_counts = Counter()  # (e,f)
    lex_norm = Counter()  # e
    al_counts = Counter()  # (static_cond, dynamic_cond)
    ll = 0
    norm_coeff = 1.0 - p_0
    corpus, trans_params, dist_weights, dist_cons = buffer
    feature_dim = len(dist_weights)
    for (e_toks, f_toks, f_heads, feature_ids) in corpus:
        I = len(e_toks)
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

        gammas, xis, pair_ll = hmt.upward_downward(f_toks, e_toks + [0] * I, f_heads, trans_params, d_probs,
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
    process_queue = mp.Queue(maxsize=int(np.ceil((args.num_workers - 1) / 4)))


    def worker_wrapper(process_queue):
        while True:
            buffer = process_queue.get()
            if buffer is None:
                return
            train_iteration(buffer, args.p_0, update_queue)


    corpus = SubcorpusReader(args.corpus)
    pool = []
    for w in xrange(args.num_workers - 1):
        p = mp.Process(target=worker_wrapper, args=(process_queue,))
        p.start()
        pool.append(p)

    logger.info("Loading parameters.")
    t_params = load_params(args.params)
    cond_ids = load_cons(args.cons)
    d_weights = load_weights(args.weights)

    updater = mp.Process(target=aggregate_counts, args=(update_queue, counts_file_name))
    updater.start()

    corpus_buffer = Corpus_Buffer(corpus, buffer_size=args.buffer_size)
    logger.info("Starting worker processes..")

    for buff in corpus_buffer:
        # get all t-params of buffer
        required_ts = set()
        required_cons = dict()
        for (e_toks, f_toks, f_heads, feature_ids) in buff:
            I = len(e_toks)
            for e_tok in e_toks + [0]:
                for f_tok in f_toks:
                    required_ts.add((e_tok, f_tok))

            # start cons
            static_cond = feature_ids[0][0][0]
            required_cons[static_cond] = cond_ids[static_cond]
            for cid in feature_ids[0][0][1]:
                required_cons[cid] = cond_ids[cid]
            for j in xrange(1, len(f_toks)):
                for i_p in xrange(I):
                    static_cond = feature_ids[j][i_p][0]
                    required_cons[static_cond] = cond_ids[static_cond]
                    for cid in feature_ids[j][i_p][1]:
                        required_cons[cid] = cond_ids[cid]

        t_probs = {ef: t_params[ef] for ef in required_ts if ef in t_params}
        process_queue.put((buff, t_probs, np.array(d_weights), required_cons))
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
