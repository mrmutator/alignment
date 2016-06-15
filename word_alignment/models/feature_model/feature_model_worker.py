from __future__ import division
from collections import Counter
import numpy as np
import multiprocessing as mp
import argparse
from CorpusReader import SubcorpusReader, Corpus_Buffer
import logging
import hmt
import features

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s  %(message)s')
logger = logging.getLogger()

def make_feature_vector(feature_set, length):
    vector = np.ones(length) * -1
    for fi in feature_set:
        vector[fi] = 1.0
    return vector

def train_iteration(buffer, p_0, queue):
    # set all counts to zero
    lex_counts = Counter()  # (e,f)
    lex_norm = Counter()  # e
    start_counts = Counter()  # (I, i)
    start_norm = Counter()  # I
    al_counts = Counter()  # (i_p, i)
    al_norm = Counter()  # (i_p)
    ll = 0
    norm_coeff = 1.0 - p_0


    corpus, trans_params, d_params, s_params, num_start_features, num_dist_features, start_cons, dist_cons = buffer
    for (e_toks, f_toks, f_heads, order, feature_ids) in corpus:
        I = len(e_toks)
        I_double = 2 * I

        # start probs
        start_weights = np.zeros((I, num_start_features))
        for i in xrange(I):
            start_weights[i] = s_params[i]
        start_feature_vector = make_feature_vector(start_cons[feature_ids[0]], num_start_features)
        numerator = np.exp(np.dot(start_weights, start_feature_vector))
        Z = np.sum(numerator)
        s_probs = (numerator / Z)

        s_probs = s_probs * norm_coeff
        start_prob = np.hstack((s_probs, np.ones(I) * (p_0 / I)))

        # dist probs
        d_probs = dict()
        for j in xrange(1, len(f_toks)):
            tmp_prob = np.zeros((I, I))
            for i_p in xrange(I):
                j_dist_vector = make_feature_vector(dist_cons[feature_ids[j][i_p]], num_dist_features)
                dist_weights = np.zeros((I, num_dist_features))
                for i in xrange(I):
                    jmp = i - i_p
                    dist_weights[i] = d_params[jmp]
                numerator = np.exp(np.dot(dist_weights, j_dist_vector))
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
        f_0 = f_toks[0]
        for i, e_tok in enumerate(e_toks):
            start_counts[(feature_ids[0], i)] += gammas[0][i]
            start_norm[feature_ids[0]] += gammas[0][i]
            if (e_tok, f_0) in trans_params:
                lex_counts[(e_tok, f_0)] += gammas[0][i]
                lex_norm[e_tok] += gammas[0][i]
        if (0, f_0) in trans_params:
            zero_sum = np.sum(gammas[0][I:])
            lex_counts[(0, f_0)] += zero_sum
            lex_norm[0] += zero_sum

        for j_p, f_tok in enumerate(f_toks[1:]):
            j = j_p + 1
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
                    al_counts[(feature_ids[j][actual_i_p], i-actual_i_p)] += xis[j][i_p][i]
                    al_norm[feature_ids[j][actual_i_p]] += gammas[j_p][i_p]

        ll += pair_ll

    queue.put((lex_counts, lex_norm, al_counts, al_norm, start_counts, start_norm, ll))


def load_params(t_params, d_params, s_params, file_name):
    num_dist_features = 0
    num_start_features = 0
    max_jump = 0
    temp_start_params = dict()
    temp_dist_params = dict()
    infile = open(file_name, "r")
    for line in infile:
        els = line.strip().split(" ")
        p_type = els[0]
        if p_type == "t":
            e = int(els[1])
            f = int(els[2])
            p = float(els[3])
            t_params[(e, f)] = p
        elif p_type == "dw":
            j = int(els[1])
            fi = int(els[2])
            w = float(els[3])
            temp_dist_params[(j, fi)] = w
            if fi > num_dist_features:
                num_dist_features = fi
            if j > max_jump:
                max_jump = j
        elif p_type == "sw":
            j = int(els[1])
            fi = int(els[2])
            w = float(els[3])
            temp_start_params[(j, fi)] = w
            if fi > num_start_features:
                num_start_features = fi
        else:
            raise Exception("Should not happen.")
    infile.close()

    # dist weights

    for j in xrange(-max_jump, max_jump+1):
        j_dist_weights = np.zeros(num_dist_features+1)
        for fi in xrange(num_dist_features+1):
            j_dist_weights[fi] = temp_dist_params[j, fi]

        d_params[j] = j_dist_weights
    # start weights
    for j in xrange(max_jump+1):
        j_start_weights = np.zeros(num_start_features+1)
        for fi in xrange(num_start_features+1):
            j_start_weights[fi] = temp_start_params[j, fi]

        s_params[j] = j_start_weights

    return num_start_features+1, num_dist_features+1





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
    arg_parser.add_argument("-start_cons", required=True)
    arg_parser.add_argument("-dist_cons", required=True)
    arg_parser.add_argument("-num_workers", required=False, type=int, default=1)
    arg_parser.add_argument("-p_0", required=False, type=float, default=0.2)
    arg_parser.add_argument("-buffer_size", required=False, type=int, default=20)

    args = arg_parser.parse_args()

    counts_file_name = args.params + ".counts"

    update_queue = mp.Queue()
    process_queue = mp.Queue(maxsize=int(np.ceil((args.num_workers - 1) / 4)))
    t_params = dict()
    d_params = dict()
    s_params = dict()

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
    num_start_features, num_dist_features = load_params(t_params, d_params, s_params, args.params)

    updater = mp.Process(target=aggregate_counts, args=(update_queue, counts_file_name))
    updater.start()

    corpus_buffer = Corpus_Buffer(corpus, buffer_size=args.buffer_size)
    logger.info("Starting worker processes..")

    dist_con_voc = features.FeatureConditions()
    dist_con_voc.load_voc(args.dist_cons)
    start_con_voc = features.FeatureConditions()
    start_con_voc.load_voc(args.start_cons)

    for buff in corpus_buffer:
        # get all t-params of buffer
        required_ts = set()
        max_I = 0
        required_start_cons = dict()
        required_dist_cons = dict()
        for (e_toks, f_toks, f_heads, order, feature_ids) in buff:
            for e_tok in e_toks + [0]:
                for f_tok in f_toks:
                    required_ts.add((e_tok, f_tok))
            I = len(e_toks)
            if I > max_I:
                max_I = I
            start_set = start_con_voc.get_feature_set(feature_ids[0])
            required_start_cons[feature_ids[0]] = start_set
            for j in xrange(1, len(f_toks)):
                for i_p in xrange(I):
                    dist_con_id = feature_ids[j][i_p]
                    required_dist_cons[dist_con_id] = dist_con_voc.get_feature_set(dist_con_id)



        # get a copy from shared dicts
        t_probs = {ef: t_params[ef] for ef in required_ts if ef in t_params}
        s_probs = {j_: np.copy(s_params[j_]) for j_ in xrange(max_I)}
        d_probs = {j_: np.copy(d_params[j_]) for j_ in xrange(-max_I+1, max_I)}
        process_queue.put((buff, t_probs, d_probs, s_probs, num_start_features, num_dist_features, required_start_cons, required_dist_cons))
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
