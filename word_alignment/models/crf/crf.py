import numpy as np
import multiprocessing as mp
import argparse
from CorpusReader import SubcorpusReader
import logging
import hmt
from features import load_vecs, load_weights
from scipy.sparse import lil_matrix
from scipy.optimize import fmin_l_bfgs_b

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s  %(message)s')
logger = logging.getLogger()



def datapoint_worker(process_queue, result_queue):
    dist_vecs = dict(vec_ids)
    feature_dim = len(global_weights)
    while True:
        buffer = process_queue.get()
        if buffer is None:
            return

        (e_toks, f_toks, f_heads, gold_aligned, feature_ids), weights = buffer
        # set all counts to zero

        I = len(e_toks)
        I_ext = I + 1
        J = len(f_toks)

        gold_ll = 0

        feature_matrices = [] * J
        empirical_expectation = np.zeros(feature_dim)

        marginals = np.zeros((J, I_ext))

        start_vecs = dist_vecs[feature_ids[0]]
        feature_matrix = lil_matrix((I_ext, feature_dim))
        for i in xrange(I):
            features_i = start_vecs[i]
            feature_matrix.rows[i+1] = features_i
            feature_matrix.data[i+1] = [1.0] * len(features_i)

        features_i = start_vecs["TN"]
        feature_matrix.rows[0] = features_i
        feature_matrix.data[0] = [1.0] * len(features_i)

        start_feature_matrix = feature_matrix.tocsr()
        feature_matrices[0].append(start_feature_matrix)
        numerator = np.exp(start_feature_matrix.dot(weights))

        marginals[0] = (numerator / np.sum(numerator))
        gold_ll += np.log(marginals[0, gold_aligned[0]])
        empirical_expectation += start_feature_matrix[gold_aligned[0]]

        # dist probs
        d_probs = np.zeros((J - 1, I_ext, I_ext))
        for j in xrange(1, J):
            for ip in xrange(I_ext):
                ip_vecs = dist_vecs[feature_ids[j][ip]]
                feature_matrix = lil_matrix((I_ext, feature_dim))
                for i in xrange(I):
                    features_i = ip_vecs[i]
                    feature_matrix.rows[i + 1] = features_i
                    feature_matrix.data[i + 1] = [1.0] * len(features_i)
                features_i = ip_vecs["TN"]
                feature_matrix.rows[0] = features_i
                feature_matrix.data[0] = [1.0] * len(features_i)

                ip_feature_matrix = feature_matrix.tocsr()
                feature_matrices[j].append(ip_feature_matrix)
                d_probs[j - 1, ip, :I+1] =  np.exp(ip_feature_matrix.dot(weights))

            gold_ll += d_probs[j-1, gold_aligned[f_heads[j]], gold_aligned[j]]
            empirical_expectation += feature_matrices[j][gold_aligned[f_heads][j]][gold_aligned[j]]

        dist_probs = (d_probs / np.sum(d_probs, axis=2)[:, :, np.newaxis])

        gammas, xis, log_likelihood = hmt.upward_downward(J, I_ext, f_heads, dist_probs,
                                                   marginals)

        feature_expectations = np.zeros(feature_dim)
        feature_expectations += start_feature_matrix.T.dot(gammas[0])
        for j in xrange(1, J):
            for i_p in xrange(I_ext):
                ip_features = feature_matrices[j][i_p]
                feature_expectations += ip_features.T.dot(xis[j][i_p])


        gradient = empirical_expectation - feature_expectations

        result_queue.put((log_likelihood, gold_ll, gradient))



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

    process_queue = mp.Queue(maxsize=num_workers*2)
    result_queue = mp.Queue()

    pool = []
    for w in xrange(num_workers):
        p = mp.Process(target=datapoint_worker, args=(process_queue, result_queue))
        p.start()
        pool.append(p)

    corpus = SubcorpusReader(args.corpus)

    def objective_func(d_weights):
        # compute ll

        corpus.reset()
        for i, buff in enumerate(corpus):
            process_queue.put(buff, d_weights)

        total_ll =0
        total_gradient = np.zeros(feature_dim)
        for _ in xrange(i+1):
            Z, gold_ll, gradient = result_queue.get()
            total_ll += -Z + gold_ll
            total_gradient += gradient

        # l2-norm:
        l2_norm = np.linalg.norm(d_weights)
        total_ll -= kappa * l2_norm

        total_gradient -= 2 * kappa * d_weights
        return -total_ll, -total_gradient


    original_weights = np.array(global_weights)
    initial_ll, _ = objective_func(original_weights)

    optimized_weights, best_ll, _ = fmin_l_bfgs_b(objective_func, np.array(global_weights), m=10, iprint=1)
    logger.info("Optimization done.")
    logger.info("Initial likelihood: " + str(-initial_ll))
    logger.info("Best likelihood: " + str(-best_ll))
    logger.info("LL diff: " + str(-initial_ll + best_ll))

    for p in pool:
        process_queue.put(None)
    for p in pool:
        p.join()


