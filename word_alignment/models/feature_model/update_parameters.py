from collections import defaultdict, Counter
import argparse
import glob
import multiprocessing as mp
import re
import logging
import numpy as np
import features
from feature_model_worker import load_params

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

def load_weights(file_name):
    d_weights = []
    with open(file_name, "r") as infile:
        for line in infile:
            w_id, w = line.strip().split()
            d_weights.append(float(w))

    return d_weights








def write_param_file(count_file_name, normalized_counts):
    param_file_name = re.sub(r"counts\.(\d+)$", r"params.\1", count_file_name)
    lengths_I = set()
    pos_jumps = defaultdict(set)
    with open(param_file_name, "w") as outfile:
        with open(count_file_name, "r") as infile:
            infile.readline()
            for line in infile:
                count_i, k, _ = line.strip().split("\t")
                count_i = int(count_i)
                if count_i == 0:
                    k_str = k.split(" ")
                    k_int = (int(k_str[0]), int(k_str[1]))
                    value = str(normalized_counts["trans_prob"][k_int])
                    outfile.write(" ".join(["t", k_str[0], k_str[1], value]) + "\n")
                elif count_i == 5:
                    k = int(k)
                    lengths_I.add(k)
                elif count_i == 2:
                    k_str = k.split(" ")
                    k_int = (int(k_str[0]), int(k_str[1]), int(k_str[2]))
                    jmp = k_int[2]-k_int[1]
                    pos_jumps[k_int[0]].add(jmp)

        for I in lengths_I:
            for i in xrange(I):
                value = normalized_counts["start_prob"][(I, i)]
                key_str = ["s"] + map(str, [I, i, value])
                outfile.write(" ".join(key_str) + "\n")

        for p in pos_jumps:
            max_I = max(pos_jumps[p])+1
            for jmp in xrange(-max_I + 1, max_I):
                value = normalized_counts["jmp_prob"][p, jmp]
                key_str = ["j"] + map(str, [p, jmp, value])
                outfile.write(" ".join(key_str) + "\n")




if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-dir", required=True)
    arg_parser.add_argument("-weight_file", required=True)
    arg_parser.add_argument("-feature_cond", required=True)
    arg_parser.add_argument("-kappa", required=True, type=float)
    args = arg_parser.parse_args()

    kappa = args.kappa

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

    def normalize_trans(queue):
        trans_prob = dict()
        for (e, f), count in lex_counts.iteritems():
            trans_prob[(e, f)] = count / lex_norm[e]
        queue.put(("trans_prob", trans_prob))


    def normalize_jumps(queue):
        # compute regularized expected complete log-likelihood
        ll = 0
        # compute ll
        for static_cond_id in static_dynamic_dict:
            dynamic_ids = list(static_dynamic_dict[static_cond_id])
            expectation_vector = np.array([al_counts[static_cond_id, dynamic_cid] for dynamic_cid in dynamic_ids])
            static_feat_set = dist_con_voc.get_feature_set(static_cond_id)
            #feature_vector = features.make_feature_vector(feat_set, num_dist_features)
            f_matrix = np.zeros((len(dynamic_ids), feature_dim ))
            for i, dynamic_cond_id in enumerate(dynamic_ids):
                dynamic_feat_set = dist_con_voc.get_feature_set(dynamic_cond_id)
                f_vector = features.make_feature_vector(static_feat_set, dynamic_feat_set, feature_dim)
                f_matrix[i] = f_vector
            numerator = np.exp(np.dot(f_matrix, d_weights))
            Z = np.sum(numerator)
            log_cond_params = np.log(numerator / Z)
            ll += np.sum(np.multiply(expectation_vector, log_cond_params))


        # l2-norm:
        tmp = 0
        l2_norm = np.linalg.norm(d_weights)
        ll -= kappa * l2_norm



    def normalize_start(queue):
        start_prob = dict()
        for (I, i), count in start_counts.iteritems():
            start_prob[(I, i)] = count / start_norm[I]
        queue.put(("start_prob", start_prob))

    results = mp.Queue()
    processes = [mp.Process(target=x, args=(results,)) for x in
                 [normalize_start, normalize_jumps, normalize_trans]]

    for p in processes:
        p.start()

    normalized_counts = dict()

    for p in processes:
        name, data = results.get()
        normalized_counts[name] = data
    for p in processes:
        a = p.join()

    logger.info("Writing parameter files.")
    for f in exp_files:
        write_param_file(f, normalized_counts)

    logger.info("Log-likelihood before update: %s" % ll)
    with open("log_likelihood", "w") as outfile:
        outfile.write("Log-Likelihood: " + str(ll) + "\n")
