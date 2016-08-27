from __future__ import division
import numpy as np
import multiprocessing as mp
import argparse
from CorpusReader import SubcorpusReader
import logging
from feature_model_worker import load_params, load_convoc_params

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s  %(message)s')
logger = logging.getLogger()

def get_viterbi_alignment(process_queue, queue):
    p_0 = args.p_0
    norm_coeff = 1.0 - p_0
    SMALL_PROB_CONST = 0.0000001


    while True:
        buffer = process_queue.get()
        if buffer is None:
            return

        pair_num, (e_toks, f_toks, f_heads, feature_ids) = buffer
        I = len(e_toks)
        J = len(f_toks)
        I_double = 2 * I

        translation_matrix = np.ones((J, I_double)) * SMALL_PROB_CONST

        # start probs
        # i_p is 0 for start_probs
        t_params_j = t_params.get(f_toks[0], None)
        if t_params_j is None:
            translation_matrix[0] = SMALL_PROB_CONST
        else:
            for i, e_tok in enumerate(e_toks):
                translation_matrix[0][i] = t_params_j.get(e_tok, SMALL_PROB_CONST)
            translation_matrix[0][I:] = t_params_j.get(0, SMALL_PROB_CONST)

        numerator = start_params[feature_ids[0]][:I_double]
        s_probs = (numerator / np.sum(numerator)) * norm_coeff
        start_prob = np.hstack((s_probs, np.ones(I) * (p_0 / I)))

        # dist probs
        d_probs = np.zeros((J - 1, I_double, I_double))
        tmp = np.hstack((np.zeros((I, I)), np.identity(I) * p_0))
        template = np.vstack((tmp, tmp))
        for j in xrange(1, J):
            t_params_j = t_params.get(f_toks[j], None)
            if t_params_j is not None:
                translation_matrix[j][I:] = t_params_j.get(0, SMALL_PROB_CONST)
            for actual_i_p, running_ip, in enumerate(xrange(I-1, -1, -1)):
                if t_params_j is not None:
                    translation_matrix[j][actual_i_p] = t_params_j.get(e_toks[actual_i_p], SMALL_PROB_CONST)
                all_params = dist_params[feature_ids[j][actual_i_p]]
                all_I = int((len(all_params)+1) / 2)
                diff_I = all_I-I
                tmp = all_params[running_ip+diff_I:running_ip+I+diff_I]
                d_probs[j - 1, actual_i_p, :I] = tmp
                d_probs[j - 1, actual_i_p + I, :I] = tmp

        dist_probs = ((d_probs / np.sum(d_probs, axis=2)[:, :, np.newaxis]) * norm_coeff) + template


        f = np.zeros((J, I_double))
        f_in = np.zeros((J, I_double))
        best = np.zeros((J, I_double), dtype=int)

        for j in reversed(xrange(1, J)):
            values = (np.log(translation_matrix[j]) + np.log(dist_probs[j-1])) + f_in[j]
            best_is = np.argmax(values, axis=1)
            best[j] = best_is
            f[j] = values[np.arange(I_double), best_is]
            f_in[f_heads[j]] += f[j]

        f[0] = np.log(start_prob) + np.log(translation_matrix[0]) + f_in[0]


        last_best = np.argmax(f[0])
        alignment = [int(last_best)]
        for j in range(1, J):
            head = f_heads[j]
            if head == 0:
                alignment.append(best[j][last_best])
            else:
                alignment.append(best[j][alignment[head]])

        # alignment = [(al, order[j]) for j, al in enumerate(alignment) if al < I]
        alignment = [(al, j) for j, al in enumerate(alignment) if al < I]

        queue.put((pair_num, alignment))

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
    arg_parser.add_argument("-params", required=True)
    arg_parser.add_argument("-convoc_params", required=True)
    arg_parser.add_argument("-out_file", required=True)
    arg_parser.add_argument("-num_workers", required=False, type=int, default=1)
    arg_parser.add_argument("-p_0", required=False, type=float, default=0.2)
    arg_parser.add_argument("-limit", required=False, type=int, default=0)

    args = arg_parser.parse_args()
    num_workers = max(1, args.num_workers-1)

    results_queue = mp.Queue()
    process_queue = mp.Queue(maxsize=num_workers)

    logger.info("Loading parameters.")
    t_params = load_params(args.params)
    start_params, dist_params = load_convoc_params(args.convoc_params)

    pool = []
    for w in xrange(num_workers):
        p = mp.Process(target=get_viterbi_alignment, args=(process_queue, results_queue))
        p.start()
        pool.append(p)

    aggregator = mp.Process(target=aggregate_alignments, args=(results_queue,))
    aggregator.start()

    logger.info("Loading corpus.")
    corpus = SubcorpusReader(args.corpus)
    for buff in enumerate(corpus):
        process_queue.put(buff)
        if buff[0]+1 == args.limit:
            break

    logger.info("Entire corpus loaded.")
    for _ in pool:
        process_queue.put(None)

    for p in pool:
        p.join()

    results_queue.put(None)
    aggregator.join()

    logger.info("Done.")
