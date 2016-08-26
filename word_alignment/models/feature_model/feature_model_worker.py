from __future__ import division
from collections import Counter, defaultdict
import numpy as np
import multiprocessing as mp
import argparse
from CorpusReader import SubcorpusReader
import logging
import hmt

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s  %(message)s')
logger = logging.getLogger()


def train_iteration(process_queue, queue):
    max_jump = args.max_jump
    if max_jump == 0:
        max_jump = np.inf
    p_0 = args.p_0
    norm_coeff = 1.0 - p_0
    SMALL_PROB_CONST = 0.00000001
    lex_counts = Counter()  # (e,f)
    lex_norm = Counter()  # e
    al_counts = Counter()
    ll = 0
    processed_count = 0
    while True:
        if processed_count > 3000:
            queue.put((lex_counts, lex_norm, al_counts, ll))
            lex_counts, lex_norm, al_counts, ll = Counter(), Counter(), Counter(), 0
        buffer = process_queue.get()
        if buffer is None:
            queue.put((lex_counts, lex_norm, al_counts, ll))
            return

        processed_count += 1
        e_toks, f_toks, f_heads, feature_ids = buffer
        # set all counts to zero

        I = len(e_toks)
        I_double = 2 * I
        J = len(f_toks)

        translation_matrix = np.ones((J, I_double)) * SMALL_PROB_CONST
        marginals = np.zeros((J, I_double))

        # start probs
        # i_p is 0 for start_probs
        t_params_j = t_params.get(f_toks[0], None)
        if t_params_j is not None:
            for i, e_tok in enumerate(e_toks):
                translation_matrix[0][i] = t_params_j.get(e_tok, SMALL_PROB_CONST)
            translation_matrix[0][I:] = t_params_j.get(0, SMALL_PROB_CONST)

        numerator = start_params[feature_ids[0]][:I]
        s_probs = (numerator / np.sum(numerator)) * norm_coeff
        marginals[0] = np.hstack((s_probs, np.ones(I) * (p_0 / I)))

        # dist probs
        d_probs = np.zeros((J - 1, I_double, I_double))
        tmp = np.hstack((np.zeros((I, I)), np.identity(I) * p_0))
        template = np.vstack((tmp, tmp))
        for j in xrange(1, J):
            t_params_j = t_params.get(f_toks[j], None)
            if t_params_j is not None:
                translation_matrix[j][I:] = t_params_j.get(0, SMALL_PROB_CONST)
            for actual_i_p, running_ip, in enumerate(xrange(I-1, -1, -1)):
                if t_params_j is not None:
                    translation_matrix[j][actual_i_p] = t_params_j.get(e_toks[actual_i_p], SMALL_PROB_CONST)
                all_params = dist_params[feature_ids[j][actual_i_p]]
                all_I = int((len(all_params)+1) / 2)
                diff_I = all_I-I
                tmp = all_params[running_ip+diff_I:running_ip+I+diff_I]
                d_probs[j - 1, actual_i_p, :I] = tmp
                d_probs[j - 1, actual_i_p + I, :I] = tmp

        dist_probs = ((d_probs / np.sum(d_probs, axis=2)[:, :, np.newaxis]) * norm_coeff) + template

        gammas, xis, log_likelihood = hmt.upward_downward(J, I_double, f_heads, translation_matrix, dist_probs,
                                                   marginals)

        # update counts
        gammas[:, I] = np.sum(gammas[:, I:], axis=1)
        e_norm = np.sum(gammas[:,:I+1], axis=0)
        e_toks += [0]
        for i, e_tok in enumerate(e_toks):
            lex_norm[e_tok] += e_norm[i]
            for j, f_tok in enumerate(f_toks):
                if translation_matrix[j, i] > SMALL_PROB_CONST:
                    lex_counts[(e_tok, f_tok)] += gammas[j][i]

        con_j0 = feature_ids[0]
        for i in xrange(I):
            al_counts[(con_j0, i)] += gammas[0][i]  # start counts
        for j, ips in enumerate(feature_ids[1:]):
            xis[j+1][:I,:I] += xis[j+1][I:,:I]
            for i_p, con in enumerate(ips):
                for i in xrange(I):
                    jmp = i-i_p
                    if jmp < -max_jump:
                        jmp = -max_jump
                    elif jmp > max_jump:
                        jmp = max_jump
                    al_counts[(ips[i_p], jmp)] += xis[j+1][i_p,i]

        ll += log_likelihood


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



def load_convoc_params(fname):
    start_params = defaultdict(list)
    dist_params = defaultdict(list)
    with open(fname, "r") as infile:
        for line in infile:
            t, con, i, p = line.split()
            if t == "s":
                start_params[con].append(float(p))
            elif t == "j":
                dist_params[con].append(float(p))

    new_start_params = dict()
    new_dist_params = dict()
    for k, l in start_params.iteritems():
        new_start_params[k] = np.array(l)
    for k, l in dist_params.iteritems():
        new_dist_params[k] = np.array(l)

    return new_start_params, new_dist_params


def aggregate_counts(queue,  counts_file):
    total = [Counter(), Counter(), Counter()]
    total_ll = 0
    while True:
        counts = queue.get()
        if counts is None:
            break
        for i, c in enumerate(counts[:3]):
            total[i].update(c)
        total_ll += counts[-1]

    logger.info("Writing counts to file.")

    # store counts
    # types = ["lex_counts", "lex_norm", "al_counts", "al_norm", "start_counts", "start_norm"]
    types = ["0", "1", "2"]
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
    arg_parser.add_argument("-convoc_params", required=True)
    arg_parser.add_argument("-num_workers", required=False, type=int, default=2)
    arg_parser.add_argument("-p_0", required=False, type=float, default=0.2)
    arg_parser.add_argument("-max_jump", type=int, default=0)
    SMALL_PROB_CONST = 0.00000001

    args = arg_parser.parse_args()

    counts_file_name = args.params + ".counts"

    update_queue = mp.Queue()
    num_workers = args.num_workers
    updater = mp.Process(target=aggregate_counts, args=(update_queue, counts_file_name))
    updater.start()
    process_queue = mp.Queue(maxsize=num_workers*2)


    logger.info("Loading parameters.")
    t_params = load_params(args.params)
    start_params, dist_params = load_convoc_params(args.convoc_params)

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
