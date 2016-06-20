from collections import defaultdict, Counter
import argparse
import glob
import re
import logging
import numpy as np
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

def normalize_trans(lex_counts, lex_norm):
    trans_prob = dict()
    for (e, f), count in lex_counts.iteritems():
        trans_prob[(e, f)] = count / lex_norm[e]
    return trans_prob


def optimize_weights(d_weights, static_dynamic_dict, dist_con_voc, learning_rate):
    # compute regularized expected complete log-likelihood
    ll = 0
    old_ll = -np.inf
    ll_diff = -np.inf
    # compute ll
    it_count = 0
    while ll_diff < -0.1:
        it_count += 1
        grad_ll = np.zeros(len(d_weights))
        for static_cond_id, (expectation_vector, dynamic_ids) in static_dynamic_dict.iteritems():
            static_feat_set = dist_con_voc.get_feature_set(static_cond_id)
            f_matrix = np.zeros((len(dynamic_ids), feature_dim))
            for si in static_feat_set:
                f_matrix[:, si] = 1.0
            for i, dynamic_cond_id in enumerate(dynamic_ids):
                dynamic_feat_set = dist_con_voc.get_feature_set(dynamic_cond_id)
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

        # l2-norm:
        l2_norm = np.linalg.norm(d_weights)
        ll -= kappa * l2_norm

        grad_ll -= 2 * kappa * d_weights
        weights_before_update = d_weights
        d_weights = d_weights + (learning_rate * grad_ll)
        ll_diff = old_ll - ll
        #print ll, ll_diff, learning_rate
        old_ll = ll
        ll = 0

    logger.info("M-step - Num of iterations: " + str(it_count))
    return weights_before_update

def compute_expectation_vectors(static_dynamic_dict, al_counts):
    for static_cond_id in static_dynamic_dict:
        dynamic_ids = list(static_dynamic_dict[static_cond_id])
        expectation_vector = np.array([al_counts[static_cond_id, dynamic_cid] for dynamic_cid in dynamic_ids])
        static_dynamic_dict[static_cond_id] = (expectation_vector, dynamic_ids)

def write_weight_file(out_file_name, weights):
    with open(out_file_name, "w") as outfile:
        for w_id, w in enumerate(weights):
            outfile.write("w " + str(w_id) + " " + str(w) + "\n")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-dir", required=True)
    arg_parser.add_argument("-weight_file", required=True)
    arg_parser.add_argument("-feature_cond", required=True)
    arg_parser.add_argument("-kappa", required=True, type=float)
    arg_parser.add_argument("-learning_rate", required=False, type=float, default=0.001)
    args = arg_parser.parse_args()

    kappa = args.kappa
    learning_rate = args.learning_rate

    exp_files = glob.glob(args.dir.rstrip("/") + "/*.counts.*")


    # types = ["lex_counts", "lex_norm", "al_counts", "al_norm", "start_counts", "start_norm", "ll", "max_I_dist", "max_I_start"]
    total = [Counter(), Counter(), Counter(), 0.0]
    static_dynamic_dict = defaultdict(set)
    logger.info("Aggregating counts.")
    for f in exp_files:
        update_count_file(f, total, static_dynamic_dict)

    lex_counts, lex_norm, al_counts, ll = total

    logger.info("Optimizing / M-step")

    d_weights = load_weights(args.weight_file)
    feature_dim = len(d_weights)

    dist_con_voc = features.FeatureConditions()
    dist_con_voc.load_voc(args.feature_cond)

    compute_expectation_vectors(static_dynamic_dict, al_counts)
    optimized_weights = optimize_weights(d_weights, static_dynamic_dict, dist_con_voc, learning_rate)

    normalized_trans_prob = normalize_trans(lex_counts, lex_norm)



    logger.info("Writing parameter files.")
    for f in exp_files:
        write_param_file(f, normalized_trans_prob)
    write_weight_file(args.weight_file + ".updated", optimized_weights)

    logger.info("Log-likelihood before update: %s" % ll)
    with open("log_likelihood", "w") as outfile:
        outfile.write("Log-Likelihood: " + str(ll) + "\n")
