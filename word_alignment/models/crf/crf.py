from collections import Counter, defaultdict
import numpy as np
import multiprocessing as mp
import argparse
from CorpusReader import SubcorpusReader
import logging
import hmt
from features import load_vecs, load_weights, convoc_reader
from scipy.sparse import lil_matrix

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s  %(message)s')
logger = logging.getLogger()


def all_traces(x):
    jj = np.tile(np.arange(x.shape[1]),x.shape[0])
    ii = (np.arange(x.shape[1])+np.arange(x.shape[0])[::-1,None]).ravel()
    z = np.zeros(((x.shape[0]+x.shape[1]-1),x.shape[1]))
    z[ii,jj] = x.ravel()
    return z.sum(axis=1)


def datapoint_worker(process_queue, queue):
    while True:
        buffer = process_queue.get()
        if buffer is None:
            return

        (e_toks, f_toks, f_heads, gold_aligned, feature_ids), start_params, dist_params = buffer
        # set all counts to zero

        I = len(e_toks)
        I_ext = I + 1
        J = len(f_toks)

        gold_ll = 0

        marginals = np.zeros((J, I_ext))

        # start probs

        numerator = start_params[feature_ids[0]][:I_ext]
        marginals[0] = (numerator / np.sum(numerator))
        gold_ll += np.log(marginals[0, gold_aligned[0]])

        # dist probs
        d_probs = np.zeros((J - 1, I_ext, I_ext))
        for j in xrange(1, J):
            for actual_i_p, running_ip, in enumerate(xrange(I, -2, -1)):
                all_params = dist_params[feature_ids[j][actual_i_p]]
                null, all_params = all_params[0], all_params[1:]
                all_I = int(len(all_params) / 2)
                diff_I = all_I-I
                tmp = all_params[running_ip+diff_I:running_ip+I+diff_I]
                d_probs[j - 1, actual_i_p, :I+1] = tmp
                d_probs[j-1, actual_i_p, 0] = null
            gold_ll += d_probs[j-1, gold_aligned[f_heads[j]], gold_aligned[j]]

        dist_probs = (d_probs / np.sum(d_probs, axis=2)[:, :, np.newaxis])

        gammas, xis, log_likelihood = hmt.upward_downward(J, I_ext, f_heads, dist_probs,
                                                   marginals)

        for j in xrange(1, J):
            for i_p in xrange(I_ext):
                vecs = vec_ids[feature_ids][j][i_p]
                for i in xrange(I):





        return xis, log_likelihood, gold_ll



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


def compute_lr_worker(process_queue, update_queue):
    dist_vecs = dict(vec_ids)
    feature_dim = len(global_weights)
    while True:
        buff = process_queue.get()
        if buff is None:
            return
        t, con, max_I, weights = buff
        if t == "s":
            a,b = 0, max_I

        else:
            a,b = -max_I+1, max_I
            max_I = (2* max_I)-1

        vecs_con = dist_vecs[con]

        feature_matrix = lil_matrix((max_I+1, feature_dim))
        for i, jmp in enumerate(xrange(a,b)):
            features_i = vecs_con[jmp]
            feature_matrix.rows[i+1] = features_i
            feature_matrix.data[i+1] = [1.0] * len(features_i)

        features_i = vecs_con["TN"]
        feature_matrix.rows[i] = features_i
        feature_matrix.data[i] = [1.0] * len(features_i)

        feature_matrix = feature_matrix.tocsr()
        numerator = np.exp(feature_matrix.dot(weights))

        update_queue.put((t, con, numerator))


#############################################
# main
if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-corpus", required=True)
    arg_parser.add_argument("-weights", required=True)
    arg_parser.add_argument("-vecs", required=True)
    arg_parser.add_argument("-convoc_list", required=True)
    arg_parser.add_argument("-kappa", required=True, type=float)
    arg_parser.add_argument("-num_workers", required=False, type=int, default=3)

    args = arg_parser.parse_args()
    num_workers = max(1, args.num_workers)

    kappa = args.kappa

    global_weights = load_weights(args.weights)
    feature_dim = len(global_weights)
    vec_ids = load_vecs(args.vecs)

    feature_process_queue = mp.Queue(maxsize=num_workers*2)
    parameters_queue = mp.Queue()

    feature_pool = []
    for w in xrange(num_workers):
        p = mp.Process(target=compute_lr_worker, args=(feature_dim, feature_process_queue, parameters_queue))
        p.start()
        feature_pool.append(p)

    process_queue = mp.Queue(maxsize=num_workers*2)
    result_queue = mp.Queue()

    convoc_list = convoc_reader(args.convoc_list)
    corpus = SubcorpusReader(args.corpus)
    def objective_func(d_weights):
        # compute regularized expected complete log-likelihood
        start_params = dict()
        dist_params = dict()
        ll = 0
        grad_ll = np.zeros(feature_dim)

        # precompute unnormalized parameters
        for i, buff in enumerate(convoc_list):
            feature_process_queue.put(buff)
        for _ in xrange(i+1):
            t, con, numerator = parameters_queue.get()

            if t == "s":
                start_params[con] = numerator
            elif t == "j":
                dist_params[con] = numerator
        # compute ll

        corpus.reset()
        for i, buff in enumerate(corpus):
            process_queue.put(buff, start_params, dist_params)

        total_ll =0
        for _ in xrange(i+1):
            xis, Z, gold_ll = result_queue.get()
            total_ll += -Z + gold_ll







            ll += buffer_ll
            grad_ll += buffer_grad_ll

            # l2-norm:
            l2_norm = np.linalg.norm(d_weights)
            ll -= kappa * l2_norm

            grad_ll -= 2 * kappa * d_weights
        return -ll, -grad_ll


    original_weights = np.array(d_weights)
    initial_ll, _ = objective_func(original_weights)

    optimized_weights, best_ll, _ = fmin_l_bfgs_b(objective_func, np.array(d_weights), m=10, iprint=1)
    logger.info("Optimization done.")
    logger.info("Initial likelihood: " + str(-initial_ll))
    logger.info("Best likelihood: " + str(-best_ll))
    logger.info("LL diff: " + str(-initial_ll + best_ll))

    for p in pool[:-1]:  # last one in pool is trans normalization
        process_queue.put(None)
    for p in pool:
        p.join()

    logger.info("Writing weight file.")
    write_weight_file(args.weights + ".updated", optimized_weights)

    logger.info("Total log-likelihood before update: %s" % log_likelihood_before_update)
    with open("log_likelihood", "w") as outfile:
        outfile.write("Log-Likelihood: " + str(log_likelihood_before_update) + "\n")

