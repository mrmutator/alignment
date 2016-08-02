from collections import defaultdict, Counter
import argparse
import glob
import re
import logging
import numpy as np
import multiprocessing as mp
from feature_model_worker import load_weights, load_vecs
from scipy.optimize import fmin_l_bfgs_b
from scipy.sparse import lil_matrix

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s  %(message)s')
logger = logging.getLogger()


def update_count_file(file_name, total, static_dynamic_dict):
    with open(file_name, "r") as infile:
        ll = float(infile.readline().strip().split("\t")[1])
        total[3] += ll

        for line in infile:
            count_i, k, v = line.strip().split("\t")
            count_i = int(count_i)
            v = float(v)
            k = k.split(" ")
            if len(k) == 1:
                k = int(k[0])
            else:
                if count_i == 2:
                    k = (k[0], int(k[1]))
                    # make association of static and dynamic conds
                    static = k[0]
                    dynamic = k[1]
                    static_dynamic_dict[static].add(dynamic)
                else:
                    k = tuple(map(int, k))
            total[count_i][k] += v


def write_param_file(count_file_name, normalized_trans_prob):
    param_file_name = re.sub(r"counts\.(\d+)$", r"params.\1", count_file_name)
    with open(param_file_name, "w") as outfile:
        with open(count_file_name, "r") as infile:
            infile.readline()
            for line in infile:
                count_i, k, _ = line.strip().split("\t")
                count_i = int(count_i)
                if count_i == 0:
                    k_str = k.split(" ")
                    k_int = (int(k_str[0]), int(k_str[1]))
                    value = str(normalized_trans_prob[k_int])
                    outfile.write(" ".join(["t", k_str[0], k_str[1], value]) + "\n")


def normalize_trans(in_queue):
    SMALL_PROB_CONST = 0.00000001
    lex_counts, lex_norm = in_queue.get()
    trans_prob = dict()
    for (e, f), count in lex_counts.iteritems():
        v = count / lex_norm[e]
        if v > SMALL_PROB_CONST:
            trans_prob[(e, f)] = v

    def write_param_file(count_file_name):
        param_file_name = re.sub(r"counts\.(\d+)$", r"params.\1", count_file_name)
        with open(param_file_name, "w") as outfile:
            with open(count_file_name, "r") as infile:
                infile.readline()
                for line in infile:
                    count_i, k, _ = line.strip().split("\t")
                    count_i = int(count_i)
                    if count_i == 0:
                        k_str = k.split(" ")
                        k_int = (int(k_str[0]), int(k_str[1]))
                        if k_int in trans_prob:
                            value = str(trans_prob[k_int])
                            outfile.write(" ".join(["t", k_str[0], k_str[1], value]) + "\n")

    while True:
        fname = in_queue.get()
        if fname is not None:
            write_param_file(fname)
        else:
            break
    logger.info("Translation parameter files written.")


def compute_expectation_vectors(static_dynamic_dict, al_counts):
    for static_cond_id in static_dynamic_dict:
        dynamic_ids = list(static_dynamic_dict[static_cond_id])
        expectation_vector = np.array([al_counts[static_cond_id, dynamic_cid] for dynamic_cid in dynamic_ids])
        static_dynamic_dict[static_cond_id] = (expectation_vector, dynamic_ids)


def write_weight_file(out_file_name, weights):
    with open(out_file_name, "w") as outfile:
        for w_id, w in enumerate(weights):
            outfile.write("w " + str(w_id) + " " + str(w) + "\n")


def optimization_worker(feature_dim, process_queue, results_queue):
    while True:
        buffer = process_queue.get()
        if buffer is None:
            return
        d_weights, expectation_vector, dynamic_feat_sets = buffer

        f_matrix = lil_matrix((len(dynamic_feat_sets), feature_dim))
        for i, dynamic_feat_set in enumerate(dynamic_feat_sets):
            f_matrix.rows[i] = dynamic_feat_set
            f_matrix.data[i] = [1.0] * len(dynamic_feat_set)
        f_matrix = f_matrix.tocsr()
        numerator = np.exp(f_matrix.dot(d_weights))
        cond_params = (numerator / np.sum(numerator))
        ll = np.sum(np.multiply(expectation_vector, np.log(cond_params)))
        # ll += expectation_vector.dot(np.log(cond_params)) # slower

        c_t_sum = f_matrix.T.dot(cond_params)
        grad_c_t_w = np.asarray(f_matrix - c_t_sum)  # still a |d| x |f| matrix
        grad_c_t = expectation_vector.dot(grad_c_t_w)  # multiply each row by d_expectation_count

        results_queue.put((ll, grad_c_t))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-dir", required=True)
    arg_parser.add_argument("-weights", required=True)
    arg_parser.add_argument("-vecs", required=True)
    arg_parser.add_argument("-kappa", required=True, type=float)
    arg_parser.add_argument("-num_workers", required=False, type=int, default=3)

    args = arg_parser.parse_args()

    kappa = args.kappa

    exp_files = glob.glob(args.dir.rstrip("/") + "/*.counts.*")

    results_queue = mp.Queue()
    process_queue = mp.Queue(maxsize=args.num_workers)
    trans_queue = mp.Queue()

    d_weights = load_weights(args.weights)
    feature_dim = len(d_weights)

    pool = []
    for w in xrange(max(1, args.num_workers - 2)):
        p = mp.Process(target=optimization_worker, args=(feature_dim, process_queue, results_queue))
        p.start()
        pool.append(p)

    p = mp.Process(target=normalize_trans, args=(trans_queue,))
    p.start()
    pool.append(p)

    # types = ["lex_counts", "lex_norm", "al_counts", "ll"]
    total = [Counter(), Counter(), Counter(), 0.0]
    static_dynamic_dict = defaultdict(set)
    logger.info("Aggregating counts.")
    for f in exp_files:
        update_count_file(f, total, static_dynamic_dict)

    lex_counts, lex_norm, al_counts, log_likelihood_before_update = total

    logger.info("Starting translation parameter normalization")
    trans_queue.put((lex_counts, lex_norm))
    for f in exp_files:
        trans_queue.put(f)
    trans_queue.put(None)

    logger.info("Optimizing / M-step")



    dist_vecs = load_vecs(args.vecs)

    compute_expectation_vectors(static_dynamic_dict, al_counts)
    data_num = len(static_dynamic_dict)

    def objective_func(d_weights):
        # compute regularized expected complete log-likelihood
        ll = 0
        grad_ll = np.zeros(feature_dim)
        for static_cond_id, (expectation_vector, dynamic_ids) in static_dynamic_dict.iteritems():
            vecs = dist_vecs[static_cond_id]
            process_queue.put((np.array(d_weights), expectation_vector, map(vecs.__getitem__, dynamic_ids)))

        for _ in xrange(data_num):
            buffer_ll, buffer_grad_ll = results_queue.get()
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
