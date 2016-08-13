import numpy as np
import multiprocessing as mp
import argparse
from CorpusReader import SubcorpusReader
import logging
from features import load_vecs, load_weights
from scipy.sparse import lil_matrix

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s  %(message)s')
logger = logging.getLogger()


def predict(process_queue, result_queue):
    dist_vecs = dict(vec_ids)
    feature_dim = len(global_weights)
    weights = np.array(global_weights)
    while True:
        buffer = process_queue.get()
        if buffer is None:
            return

        num, (I, f_heads, gold_aligned, feature_ids) = buffer
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
        alignment = [(al-1, j) for j, al in enumerate(alignment) if al > 0]


        result_queue.put((num, alignment))

def aggregate_alignments(queue):
    outfile = open(args.out_file, "w")
    buffer_dict = dict()
    next_c = 0
    while True:
        if next_c in buffer_dict:
            alignment = buffer_dict[next_c]
            alignment = [str(al[0]) + "-" + str(al[1]) for al in alignment]
            outfile.write(" ".join(alignment) + "\n")
            del buffer_dict[next_c]
            next_c += 1
            continue

        obj = queue.get()
        if obj is None:
            break
        num, alignment = obj
        if num == next_c:
            alignment = [str(al[0]) + "-" + str(al[1]) for al in alignment]
            outfile.write(" ".join(alignment) + "\n")
            next_c += 1
        else:
            buffer_dict[num] = alignment

    while len(buffer_dict) > 0:
        alignment = buffer_dict[next_c]
        alignment = [str(al[0]) + "-" + str(al[1]) for al in alignment]
        outfile.write(" ".join(alignment) + "\n")
        del buffer_dict[next_c]
        next_c += 1

    outfile.close()
#############################################
# main
if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-corpus", required=True)
    arg_parser.add_argument("-weights", required=True)
    arg_parser.add_argument("-vecs", required=True)
    arg_parser.add_argument("-out_file", required=True)
    arg_parser.add_argument("-num_workers", required=False, type=int, default=3)

    args = arg_parser.parse_args()
    num_workers = max(1, args.num_workers)


    global_weights = load_weights(args.weights)
    feature_dim = len(global_weights)
    vec_ids = load_vecs(args.vecs)


    evaluate_queue = mp.Queue(maxsize=num_workers*2)
    evaluated_queue = mp.Queue()

    eval_pool = []
    for w in xrange(num_workers):
        p = mp.Process(target=predict, args=(evaluate_queue, evaluated_queue))
        p.start()
        eval_pool.append(p)
    aggregator = mp.Process(target=aggregate_alignments, args=(evaluated_queue,))
    aggregator.start()


    corpus = SubcorpusReader(args.corpus)
    for i, buff in enumerate(corpus):
        evaluate_queue.put((i, buff))


    for p in eval_pool:
        evaluate_queue.put(None)
    for p in eval_pool:
        p.join()
    evaluated_queue.put(None)
    aggregator.join()


