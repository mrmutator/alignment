from collections import defaultdict, Counter
import argparse
import glob
import re
import logging
import numpy as np
import multiprocessing as mp
from compute_params import load_weights
from scipy.optimize import fmin_l_bfgs_b
from scipy.sparse import lil_matrix, diags

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s  %(message)s')
logger = logging.getLogger()

def load_vecs(file_name):
    vec_ids = dict()
    infile = open(file_name, "r")
    for line in infile:
        els = line.split()
        t, con, jmp = els[0], els[1], els[2]
        if con not in vec_ids:
            vec_ids[con] = dict()

        vec_ids[con][int(jmp)] = sorted(map(int, els[3:]))
    infile.close()
    return vec_ids


def update_count_file(file_name, total):
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
                else:
                    k = tuple(map(int, k))
            total[count_i][k] += v




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


def compute_expectation_vectors(convoc_list_file, al_counts, num_workers):
    datas = [dict() for _ in xrange(num_workers)]
    data_count = [0 for _ in xrange(num_workers)]
    data = dict()
    with open(convoc_list_file, "r") as infile:
        for line in infile:
            els = line.split()
            t, con, jmp = els[0], els[1], int(els[2])
            feature_i = sorted(map(int, els[3:]))
            if con not in data:
                data[con] = (list(), list())
            data[con][0].append(feature_i)
            data[con][1].append(al_counts[con, jmp])
    for con in data.keys():
        i = np.argmin(data_count)
        datas[i][con] = (data[con][0], np.array(data[con][1]))
        data_count[i] += len(data[con][0])
        del data[con]
    return datas, data_count

def write_weight_file(out_file_name, weights):
    with open(out_file_name, "w") as outfile:
        for w_id, w in enumerate(weights):
            outfile.write("w " + str(w_id) + " " + str(w) + "\n")


def optimization_worker(feature_dim, process_queue, results_queue):
    data, data_length = process_queue.get()
    f_matrix = lil_matrix((data_length, feature_dim))
    sum_template = lil_matrix((len(data), data_length))
    ci = 0
    cj = 0
    expectation_vector = np.array([])
    for _, (feature_ids, exp_vec) in data.iteritems():
        expectation_vector = np.append(expectation_vector, exp_vec)
        for feature_i in feature_ids:
            f_matrix.rows[ci] = feature_i
            f_matrix.data[ci] = [1.0] * len(feature_i)
            sum_template[cj, ci] = 1.0
            ci += 1
        cj += 1
    f_matrix = f_matrix.tocsr()
    sum_template = sum_template.tocsr()
    del data

    while True:
        buffer = process_queue.get()
        if buffer is None:
            return
        d_weights = buffer
        numerator = np.exp(f_matrix.dot(d_weights))
        num_sum = sum_template.T.dot(sum_template.dot(numerator))
        cond_params = (numerator / num_sum)
        ll = np.sum(np.multiply(expectation_vector, np.log(cond_params)))
        # ll + expectation_vector.dot(np.log(cond_params)) # slower

        f_cond = diags(cond_params, 0) * f_matrix # a |f| x |d| matrix
        f_sums = sum_template.T.dot(sum_template.dot(f_cond)) # --> |d| * |f|
        grad_c_t_w = f_matrix - f_sums
        grad_c_t = grad_c_t_w.T.dot(expectation_vector)  # multiply each row by d_expectation_count

        results_queue.put((ll, grad_c_t))


def aggregator(process_queue, results_queue, feature_dim):
    num_data = process_queue.get()
    ll = 0
    grad_ll = np.zeros(feature_dim)
    c = 0
    while True:
        if c == num_data:
            results_queue.put((ll, grad_ll))
            ll = 0
            grad_ll = np.zeros(feature_dim)
            c = 0
        buffer = process_queue.get()
        if buffer is None:
            assert c == 0
            return
        tmp_ll, tmp_grad_ll = buffer
        ll += tmp_ll
        grad_ll += tmp_grad_ll
        c += 1



if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-dir", required=True)
    arg_parser.add_argument("-weights", required=True)
    arg_parser.add_argument("-convoc_list", required=True)
    arg_parser.add_argument("-kappa", required=True, type=float)
    arg_parser.add_argument("-lbfgs_maxiter", required=False, type=int, default=15000)
    arg_parser.add_argument("-num_workers", required=False, type=int, default=3)

    args = arg_parser.parse_args()

    worker_num = max(1, args.num_workers - 2)

    kappa = args.kappa

    exp_files = glob.glob(args.dir.rstrip("/") + "/*.counts.*")

    results_queue = mp.Queue()
    process_queues = [mp.Queue() for _ in xrange(worker_num)]
    final_queue = mp.Queue()

    trans_queue = mp.Queue()

    d_weights = load_weights(args.weights)
    feature_dim = len(d_weights)

    pool = []
    for w in xrange(worker_num):
        p = mp.Process(target=optimization_worker, args=(feature_dim, process_queues[w], results_queue))
        p.start()
        pool.append(p)

    p = mp.Process(target=aggregator, args=(results_queue, final_queue, feature_dim))
    p.start()
    pool.append(p)

    p = mp.Process(target=normalize_trans, args=(trans_queue,))
    p.start()
    pool.append(p)

    # types = ["lex_counts", "lex_norm", "al_counts", "ll"]
    total = [Counter(), Counter(), Counter(), 0.0]

    logger.info("Aggregating counts.")
    for f in exp_files:
        update_count_file(f, total)

    lex_counts, lex_norm, al_counts, log_likelihood_before_update = total

    logger.info("Starting translation parameter normalization")
    trans_queue.put((lex_counts, lex_norm))
    for f in exp_files:
        trans_queue.put(f)
    trans_queue.put(None)

    logger.info("Loading conditions")
    datas, data_count = compute_expectation_vectors(args.convoc_list, al_counts, worker_num)
    results_queue.put(worker_num)


    logger.info("Sending data to workers")
    for i, data in enumerate(datas):
        process_queues[i].put((data, data_count[i]))

    def objective_func(d_weights):
        # compute regularized expected complete log-likelihood
        for i in xrange(worker_num):
            process_queues[i].put(np.array(d_weights))

        ll, grad_ll = final_queue.get()

        # l2-norm:
        l2_norm = np.linalg.norm(d_weights)
        ll -= kappa * l2_norm

        grad_ll -= 2 * kappa * d_weights
        return -ll, -grad_ll


    original_weights = np.array(d_weights)
    initial_ll, _ = objective_func(original_weights)

    logger.info("Starting lbfgs optimization.")
    optimized_weights, best_ll, _ = fmin_l_bfgs_b(objective_func, np.array(d_weights), m=10, iprint=1, maxiter=args.lbfgs_maxiter)
    logger.info("Optimization done.")
    logger.info("Initial likelihood: " + str(-initial_ll))
    logger.info("Best likelihood: " + str(-best_ll))
    logger.info("LL diff: " + str(-initial_ll + best_ll))


    for q in process_queues:  # last one in pool is trans normalization and aggregator
        q.put(None)
    results_queue.put(None)
    for p in pool:
        p.join()

    logger.info("Writing weight file.")
    write_weight_file(args.weights + ".updated", optimized_weights)

    logger.info("Total log-likelihood before update: %s" % log_likelihood_before_update)
    with open("log_likelihood", "w") as outfile:
        outfile.write("Log-Likelihood: " + str(log_likelihood_before_update) + "\n")
