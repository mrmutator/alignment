from collections import defaultdict, Counter
import argparse
import glob
import re
import logging
import numpy as np
import multiprocessing as mp
from compute_params import load_weights, load_vecs

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

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-dir", required=True)
    arg_parser.add_argument("-weights", required=True)
    arg_parser.add_argument("-vecs", required=True)

    args = arg_parser.parse_args()


    exp_files = glob.glob(args.dir.rstrip("/") + "/*.counts.*")

    trans_queue = mp.Queue()

    d_weights = load_weights(args.weights)
    feature_dim = len(d_weights)

    trans_norm_process = mp.Process(target=normalize_trans, args=(trans_queue,))
    trans_norm_process.start()

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


    dist_vecs = load_vecs(args.vecs)

    compute_expectation_vectors(static_dynamic_dict, al_counts)
    data_num = len(static_dynamic_dict)

    optimized_weights = np.zeros(feature_dim)
    for static_cond_id, (expectation_vector, dynamic_ids) in static_dynamic_dict.iteritems():
        vecs = dist_vecs[static_cond_id]
        feature_sets =  map(vecs.__getitem__, dynamic_ids)
        Z = np.sum(expectation_vector)
        for i, feature_set in enumerate(feature_sets):
            assert len(feature_set) == 1
            optimized_weights[feature_set[0]] = expectation_vector[i] / Z


    logger.info("Writing weight file.")
    write_weight_file(args.weights + ".updated", optimized_weights)
    trans_norm_process.join()
