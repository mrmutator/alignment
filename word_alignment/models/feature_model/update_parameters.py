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




# def optimize_weights_sgd(d_weights, static_dynamic_dict, dist_con_voc, learning_rate):
#     # compute regularized expected complete log-likelihood
#     N = len(static_dynamic_dict)
#     sgd_reg_coeff = 2*kappa / float(N)
#     ll_diff = -np.inf
#     # compute ll
#     ll = 0
#     for static_cond_id, (expectation_vector, dynamic_ids) in static_dynamic_dict.iteritems():
#         static_feat_set = dist_con_voc.get_feature_set(static_cond_id)
#         f_matrix = np.zeros((len(dynamic_ids), feature_dim))
#         for si in static_feat_set:
#             f_matrix[:, si] = 1.0
#         for i, dynamic_cond_id in enumerate(dynamic_ids):
#             dynamic_feat_set = dist_con_voc.get_feature_set(dynamic_cond_id)
#             for di in dynamic_feat_set:
#                 f_matrix[i, di] = 1.0
#         numerator = np.exp(np.dot(f_matrix, d_weights))
#         Z = np.sum(numerator)
#         cond_params = (numerator / Z)
#         ll += np.sum(np.multiply(expectation_vector, np.log(cond_params)))
#     l2_norm = np.linalg.norm(d_weights)
#     ll -= kappa * l2_norm
#
#     old_ll = ll
#     it_count = 0
#     while ll_diff < -0.001:
#         it_count += 1
#         for static_cond_id, (expectation_vector, dynamic_ids) in static_dynamic_dict.iteritems():
#             static_feat_set = dist_con_voc.get_feature_set(static_cond_id)
#             f_matrix = np.zeros((len(dynamic_ids), feature_dim))
#             for si in static_feat_set:
#                 f_matrix[:, si] = 1.0
#             for i, dynamic_cond_id in enumerate(dynamic_ids):
#                 dynamic_feat_set = dist_con_voc.get_feature_set(dynamic_cond_id)
#                 for di in dynamic_feat_set:
#                     f_matrix[i, di] = 1.0
#             numerator = np.exp(np.dot(f_matrix, d_weights))
#             Z = np.sum(numerator)
#             cond_params = (numerator / Z)
#
#             c_t_sum = np.sum(f_matrix * cond_params[:, np.newaxis], axis=0)
#             grad_c_t_w = f_matrix - c_t_sum  # still a |d| x |f| matrix
#             grad_c_t = np.sum(expectation_vector[:, np.newaxis] * grad_c_t_w,
#                               axis=0)  # multiply each row by d_expectation_count
#             grad_c_t -=  sgd_reg_coeff * d_weights
#             d_weights = d_weights + (learning_rate * grad_c_t)
#
#         # compute ll
#         ll = 0
#         for static_cond_id, (expectation_vector, dynamic_ids) in static_dynamic_dict.iteritems():
#             static_feat_set = dist_con_voc.get_feature_set(static_cond_id)
#             f_matrix = np.zeros((len(dynamic_ids), feature_dim))
#             for si in static_feat_set:
#                 f_matrix[:, si] = 1.0
#             for i, dynamic_cond_id in enumerate(dynamic_ids):
#                 dynamic_feat_set = dist_con_voc.get_feature_set(dynamic_cond_id)
#                 for di in dynamic_feat_set:
#                     f_matrix[i, di] = 1.0
#             numerator = np.exp(np.dot(f_matrix, d_weights))
#             Z = np.sum(numerator)
#             cond_params = (numerator / Z)
#             ll += np.sum(np.multiply(expectation_vector, np.log(cond_params)))
#         l2_norm = np.linalg.norm(d_weights)
#         ll -= kappa * l2_norm
#         ll_diff = old_ll - ll
#         print ll, ll_diff, learning_rate
#         old_ll = ll
#
#     logger.info("M-step - Num of iterations: " + str(it_count))
#     return d_weights

# def optimize_weights_batch_gd(d_weights, static_dynamic_dict, dist_con_voc, learning_rate, convergence_threshold):
#     # compute regularized expected complete log-likelihood
#     ll = 0
#     old_ll = -np.inf
#     ll_diff = -np.inf
#     # compute ll
#     it_count = 0
#     while ll_diff < convergence_threshold:
#         it_count += 1
#         grad_ll = np.zeros(len(d_weights))
#         for static_cond_id, (expectation_vector, dynamic_ids) in static_dynamic_dict.iteritems():
#             static_feat_set = dist_con_voc.get_feature_set(static_cond_id)
#             f_matrix = np.zeros((len(dynamic_ids), feature_dim))
#             for si in static_feat_set:
#                 f_matrix[:, si] = 1.0
#             for i, dynamic_cond_id in enumerate(dynamic_ids):
#                 dynamic_feat_set = dist_con_voc.get_feature_set(dynamic_cond_id)
#                 for di in dynamic_feat_set:
#                     f_matrix[i, di] = 1.0
#             numerator = np.exp(np.dot(f_matrix, d_weights))
#             Z = np.sum(numerator)
#             cond_params = (numerator / Z)
#             ll += np.sum(np.multiply(expectation_vector, np.log(cond_params)))
#
#             c_t_sum = np.sum(f_matrix * cond_params[:, np.newaxis], axis=0)
#             grad_c_t_w = f_matrix - c_t_sum  # still a |d| x |f| matrix
#             grad_c_t = np.sum(expectation_vector[:, np.newaxis] * grad_c_t_w,
#                               axis=0)  # multiply each row by d_expectation_count
#             grad_ll += grad_c_t
#
#         # l2-norm:
#         l2_norm = np.linalg.norm(d_weights)
#         ll -= kappa * l2_norm
#
#         grad_ll -= 2 * kappa * d_weights
#         weights_before_update = d_weights
#         d_weights = d_weights + (learning_rate * grad_ll)
#         ll_diff = old_ll - ll
#         print ll, ll_diff, learning_rate
#         old_ll = ll
#         ll = 0
#
#     logger.info("M-step - Num of iterations: " + str(it_count))
#     return weights_before_update

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
    arg_parser.add_argument("-convergence_threshold", required=False, type=float, default=-0.01)
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

    # compute regularized expected complete log-likelihood
    ll = 0
    old_ll = -np.inf
    ll_diff = -np.inf
    # compute ll
    it_count = 0
    while ll_diff < args.convergence_threshold:
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
        weights_before_update = d_weights
        d_weights = d_weights + (learning_rate * grad_ll)
        ll_diff = old_ll - ll
        print ll, ll_diff, learning_rate
        old_ll = ll
        ll = 0

    logger.info("M-Step optimization done.")
    logger.info("After " + str(it_count) + " iterations.")
    optimized_weights = weights_before_update

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
