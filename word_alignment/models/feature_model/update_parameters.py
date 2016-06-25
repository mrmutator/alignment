from collections import defaultdict, Counter
import argparse
import glob
import re
import logging
import numpy as np
import multiprocessing as mp
import features
from feature_model_worker import load_weights
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
                k = tuple(map(int, k))
                if count_i == 2:
                    # make association of static and dynamic conds
                    static = k[0]
                    dynamic = k[1]
                    static_dynamic_dict[static].add(dynamic)
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
    lex_counts, lex_norm = in_queue.get()
    trans_prob = dict()
    for (e, f), count in lex_counts.iteritems():
        trans_prob[(e, f)] = count / lex_norm[e]

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

def optimization_worker(buffer, results_queue):
    d_weights = buffer[0]
    feature_dim = len(d_weights)
    ll = 0
    grad_ll = np.zeros(len(d_weights))
    for static_feat_set, expectation_vector, dynamic_feat_sets in buffer[1]:
        f_matrix = np.zeros((len(dynamic_feat_sets), feature_dim))
        for si in static_feat_set:
            f_matrix[:, si] = 1.0
        for i, dynamic_feat_set in enumerate(dynamic_feat_sets):
            for di in dynamic_feat_set:
                f_matrix[i, di] = 1.0
        numerator = np.exp(np.dot(f_matrix, d_weights))
        Z = np.sum(numerator)
        cond_params = (numerator / Z)
        ll += np.sum(np.multiply(expectation_vector, np.log(cond_params)))

        c_t_sum = np.sum(f_matrix * cond_params[:, np.newaxis], axis=0)
        grad_c_t_w = f_matrix - c_t_sum  # still a |d| x |f| matrix
        grad_c_t = np.sum(expectation_vector[:, np.newaxis] * grad_c_t_w,
                          axis=0)  # multiply each row by d_expectation_count
        grad_ll += grad_c_t

    results_queue.put((ll, grad_ll))




if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-dir", required=True)
    arg_parser.add_argument("-weights", required=True)
    arg_parser.add_argument("-cons", required=True)
    arg_parser.add_argument("-kappa", required=True, type=float)
    arg_parser.add_argument("-learning_rate", required=False, type=float, default=0.001)
    arg_parser.add_argument("-num_workers", required=False, type=int, default=3)
    arg_parser.add_argument("-buffer_size", required=False, type=int, default=20)
    arg_parser.add_argument("-convergence_threshold", required=False, type=float, default=-0.1)
    arg_parser.add_argument("-it_limit", required=False, type=int, default=500)
    arg_parser.add_argument("-grid_it_limit", required=False, type=int, default=200)
    arg_parser.add_argument("-grid_convergence_threshold", required=False, type=float, default=-10.0)
    arg_parser.add_argument("-grid_trials", required=False, type=int, default=10)

    args = arg_parser.parse_args()

    kappa = args.kappa
    learning_rate = args.learning_rate

    exp_files = glob.glob(args.dir.rstrip("/") + "/*.counts.*")

    results_queue = mp.Queue()
    process_queue = mp.Queue()
    trans_queue = mp.Queue()

    def worker_wrapper(process_queue):
        while True:
            buffer = process_queue.get()
            if buffer is None:
                return
            optimization_worker(buffer, results_queue)

    pool = []
    for w in xrange(max(1, args.num_workers - 2)):
        p = mp.Process(target=worker_wrapper, args=(process_queue,))
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

    d_weights = load_weights(args.weights)
    feature_dim = len(d_weights)

    dist_con_voc = features.FeatureConditions()
    dist_con_voc.load_voc(args.cons)



    compute_expectation_vectors(static_dynamic_dict, al_counts)


    def optimize(d_weights, learning_rate, LIMIT, convergence_threshold, initial_ll=-np.inf):
        it_limit_achieved = False
        # compute regularized expected complete log-likelihood
        ll = initial_ll
        ll_diff = -np.inf
        last_diff = None
        # compute ll
        it_count = -1
        while ll_diff < convergence_threshold:
            old_ll = ll
            ll = 0
            last_diff = ll_diff
            weights_before_update = np.array(d_weights)
            buffers = 0
            it_count += 1
            if it_count % 50 == 0:
                logger.info("Performing iteration " + str(it_count))
            grad_ll = np.zeros(len(d_weights))
            buffer = [np.array(d_weights), []]
            buffer_c = 0
            for static_cond_id, (expectation_vector, dynamic_ids) in static_dynamic_dict.iteritems():
                buffer[1].append((dist_con_voc.get_feature_set(static_cond_id), expectation_vector, map(dist_con_voc.get_feature_set, dynamic_ids)))
                buffer_c += 1
                if buffer_c == 10:
                    buffers += 1
                    process_queue.put(buffer)
                    buffer_c = 0
                    buffer = [np.array(d_weights), []]
            if buffer_c > 0:
                buffers += 1
                process_queue.put(buffer)

            for _ in xrange(buffers):
                buffer_ll, buffer_grad_ll = results_queue.get()
                ll += buffer_ll
                grad_ll += buffer_grad_ll

            # l2-norm:
            l2_norm = np.linalg.norm(d_weights)
            ll -= kappa * l2_norm

            grad_ll -= 2 * kappa * d_weights

            d_weights = d_weights + (learning_rate * grad_ll)
            ll_diff = old_ll - ll
            #print ll, ll_diff, learning_rate
            logger.info("Iteration " + str(it_count) + ": " + " ".join(map(str, [ll, ll_diff])))
            if it_count == LIMIT:
                it_limit_achieved = True
                logger.info("Optimization trial done: %s %s %s" % (it_count, old_ll, learning_rate))
                break

        return old_ll, it_limit_achieved, weights_before_update, last_diff

    original_weights = np.array(d_weights)
    best_weights = None
    best_ll = -np.inf
    best_lrt = None
    lr_t = args.learning_rate
    grid_search_c = 0
    grid_threshold = args.grid_trials
    while True:
        grid_search_c += 1
        logger.info("Trial %s - New optimization with learning rate: %s" %(grid_search_c, lr_t))
        tmp_ll, limit_achieved, w, tmp_diff = optimize(np.array(original_weights), lr_t, args.grid_it_limit, args.grid_convergence_threshold)
        if tmp_diff != -np.inf and tmp_diff <= 0 and tmp_ll > best_ll:
            ll_diff = best_ll - tmp_ll
            logger.info("Improved best likelihood by: " + str(ll_diff))
            logger.info("Best learning rate: " + str(lr_t))
            best_ll = tmp_ll
            best_weights = w
            best_lrt = lr_t
        if limit_achieved:
            lr_t *= 1.05
        else:
            lr_t *= 0.5
        if grid_search_c == grid_threshold:
            logger.info("Grid search max limit reached.")
            if best_ll == -np.inf:
                grid_threshold += 1
            else:
                break

    logger.info("Grid search done.")
    logger.info("Perform final optimization.")
    logger.info("Initial LL: " +str(best_ll))
    final_ll, limit_achieved, optimized_weights, last_diff = optimize(best_weights, best_lrt, args.it_limit, args.convergence_threshold, initial_ll=best_ll)
    logger.info("M-Step optimization done.")
    logger.info("Best optimized likelihood: " + str(final_ll))
    logger.info("Converged: " + str(not limit_achieved))

    for p in pool[:-1]: # last one in pool is trans normalization
        process_queue.put(None)
    for p in pool:
        p.join()

    # optimized_weights = optimize_weights_batch_gd(d_weights, static_dynamic_dict, dist_con_voc, learning_rate)


    logger.info("Writing weight file.")
    write_weight_file(args.weights + ".updated", optimized_weights)

    logger.info("Log-likelihood before update: %s" % log_likelihood_before_update)
    with open("log_likelihood", "w") as outfile:
        outfile.write("Log-Likelihood: " + str(log_likelihood_before_update) + "\n")
