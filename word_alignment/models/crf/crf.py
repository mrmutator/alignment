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

        (I, f_heads, gold_aligned, feature_ids), weights = buffer
        # set all counts to zero

        I_ext = I + 1
        J = len(f_heads)

        gold_ll = 0

        feature_matrices = [[] for _ in xrange(J)]
        empirical_expectation = np.zeros(feature_dim)
        marginals = np.zeros((J, I_ext))

        feature_matrix = lil_matrix((I_ext, feature_dim))
        for i in xrange(I_ext):
            features_i = dist_vecs[feature_ids[0][i]]
            feature_matrix.rows[i] = features_i
            feature_matrix.data[i] = [1.0] * len(features_i)

        start_feature_matrix = feature_matrix.tocsr()
        feature_matrices[0].append(start_feature_matrix)
        marginals[0] = np.exp(start_feature_matrix.dot(weights))

        #marginals[0] = (numerator / np.sum(numerator))
        gold_ll += np.log(marginals[0, gold_aligned[0]])
        empirical_expectation += start_feature_matrix[gold_aligned[0]].toarray().flatten()

        # dist probs
        d_probs = np.zeros((J - 1, I_ext, I_ext))
        for j in xrange(1, J):
            for ip in xrange(I_ext):
                feature_matrix = lil_matrix((I_ext, feature_dim))
                for i in xrange(I_ext):
                    features_i = dist_vecs[feature_ids[j][ip][i]]
                    feature_matrix.rows[i] = features_i
                    feature_matrix.data[i] = [1.0] * len(features_i)

                ip_feature_matrix = feature_matrix.tocsr()
                feature_matrices[j].append(ip_feature_matrix)
                d_probs[j - 1, ip, :I+1] =  np.exp(ip_feature_matrix.dot(weights))

        gammas, xis, Z = hmt.upward_downward(J, I_ext, f_heads, d_probs,
                                                   marginals)

        #print Z
        feature_expectations = np.zeros(feature_dim)
        feature_expectations += start_feature_matrix.T.dot(gammas[0])
        for j in xrange(1, J):
            gold_ll += np.log(d_probs[j-1, gold_aligned[f_heads[j]], gold_aligned[j]])
            empirical_expectation += feature_matrices[j][gold_aligned[f_heads[j]]][gold_aligned[j]].toarray().flatten()
            for i_p in xrange(I_ext):
                ip_features = feature_matrices[j][i_p]
                feature_expectations += ip_features.T.dot(xis[j][i_p])

        gradient = empirical_expectation - feature_expectations

        result_queue.put((Z, gold_ll, gradient))

def predict(process_queue, result_queue):
    dist_vecs = dict(vec_ids)
    feature_dim = len(global_weights)
    while True:
        buffer = process_queue.get()
        if buffer is None:
            return

        (I, f_heads, gold_aligned, feature_ids), weights = buffer
        # set all counts to zero

        I_ext = I + 1
        J = len(f_heads)


        feature_matrix = lil_matrix((I_ext, feature_dim))
        for i in xrange(I_ext):
            features_i = dist_vecs[feature_ids[0][i]]
            feature_matrix.rows[i] = features_i
            feature_matrix.data[i] = [1.0] * len(features_i)

        start_feature_matrix = feature_matrix.tocsr()
        start_prob = np.exp(start_feature_matrix.dot(weights))


        # dist probs
        d_probs = np.zeros((J - 1, I_ext, I_ext))
        for j in xrange(1, J):
            for ip in xrange(I_ext):
                feature_matrix = lil_matrix((I_ext, feature_dim))
                for i in xrange(I_ext):
                    features_i = dist_vecs[feature_ids[j][ip][i]]
                    feature_matrix.rows[i] = features_i
                    feature_matrix.data[i] = [1.0] * len(features_i)

                ip_feature_matrix = feature_matrix.tocsr()
                d_probs[j - 1, ip, :I + 1] = np.exp(ip_feature_matrix.dot(weights))


        f = np.zeros((J, I_ext))
        f_in = np.zeros((J, I_ext))
        best = np.zeros((J, I_ext), dtype=int)

        for j in reversed(xrange(1, J)):
            values = (np.log(d_probs[j - 1])) + f_in[j]
            best_is = np.argmax(values, axis=1)
            best[j] = best_is
            f[j] = values[np.arange(I_ext), best_is]
            f_in[f_heads[j]] += f[j]

        f[0] = np.log(start_prob)  + f_in[0]

        last_best = np.argmax(f[0])
        alignment = [int(last_best)]
        for j in range(1, J):
            head = f_heads[j]
            if head == 0:
                alignment.append(best[j][last_best])
            else:
                alignment.append(best[j][alignment[head]])

        # alignment = [(al, order[j]) for j, al in enumerate(alignment) if al < I]
        # alignment = [(al, j) for j, al in enumerate(alignment) if al < I]
        correct = 0
        wrong = 0
        for j in xrange(J):
            pred = alignment[j]
            true = gold_aligned[j]
            if pred == true:
                correct += 1
            else:
                wrong += 1

        result_queue.put((correct, wrong))



#############################################
# main
if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-corpus", required=True)
    arg_parser.add_argument("-weights", required=True)
    arg_parser.add_argument("-vecs", required=True)
    arg_parser.add_argument("-sigma", required=True, type=float)
    arg_parser.add_argument("-num_workers", required=False, type=int, default=3)

    args = arg_parser.parse_args()
    num_workers = max(1, args.num_workers)

    sigma = args.sigma
    sigmasqrd = np.square(sigma)

    global_weights = load_weights(args.weights)
    feature_dim = len(global_weights)
    vec_ids = load_vecs(args.vecs)

    process_queue = mp.Queue(maxsize=num_workers*2)
    result_queue = mp.Queue()

    evaluate_queue = mp.Queue(maxsize=num_workers*2)
    evaluated_queue = mp.Queue()

    def evaluate(d_weights):
        correct, wrong = 0, 0
        c = 0
        for buff in corpus:
            c += 1
            evaluate_queue.put((buff, np.array(d_weights)))

        for _ in xrange(c):
            tmp_correct, tmp_wrong = evaluated_queue.get()
            correct += tmp_correct
            wrong += tmp_wrong

        total = correct + wrong
        return float(correct) / total


    pool = []
    for w in xrange(num_workers):
        p = mp.Process(target=datapoint_worker, args=(process_queue, result_queue))
        p.start()
        pool.append(p)
    eval_pool = []
    for w in xrange(num_workers):
        p = mp.Process(target=predict, args=(evaluate_queue, evaluated_queue))
        p.start()
        eval_pool.append(p)


    corpus = SubcorpusReader(args.corpus)
    corpus = [buff for buff in corpus]

    def objective_func(d_weights):
        # compute ll

        c = 0
        for buff in corpus:
            c += 1
            process_queue.put((buff, np.array(d_weights)))

        total_ll =0
        total_gradient = np.zeros(feature_dim)
        for _ in xrange(c):
            Z, gold_ll, gradient = result_queue.get()
            total_ll += -Z + gold_ll
            total_gradient += gradient

        # regularization
        sm_weights = np.sum(np.square(d_weights))
        total_ll -= float(sm_weights) / (2*sigmasqrd)

        total_gradient -= d_weights / sigmasqrd
        return -total_ll, -total_gradient


    original_weights = np.array(global_weights)
    initial_ll, _ = objective_func(original_weights)
    initial_acc = evaluate(original_weights)


    optimized_weights, best_ll, _ = fmin_l_bfgs_b(objective_func, np.array(global_weights), m=10, iprint=1)
    logger.info("Optimization done.")
    logger.info("Initial likelihood: " + str(-initial_ll))
    logger.info("Best likelihood: " + str(-best_ll))
    logger.info("LL diff: " + str(initial_ll - best_ll))
    final_acc = evaluate(optimized_weights)
    logger.info("Initial accuracy: " + str(initial_acc))
    logger.info("Final accuracy: " + str(final_acc))


    for p in pool:
        process_queue.put(None)
    for p in eval_pool:
        evaluate_queue.put(None)
    for p in pool:
        p.join()
    for p in eval_pool:
        p.join()


