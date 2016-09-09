import argparse
import multiprocessing as mp
import logging
import numpy as np
from scipy.sparse import lil_matrix
import glob
from collections import defaultdict
from features import max_dict

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s  %(message)s')
logger = logging.getLogger()


def convoc_reader(fname):
    with open(fname, "r") as infile:
        els = infile.readline().split()
        t, con = els[0], els[1]
        feature_i = map(int, els[3:])
        prev = (t, con)
        data = [feature_i]
        for line in infile:
            els = line.split()
            t, con = els[0], els[1]
            feature_i = map(int, els[3:])
            if prev != (t, con):
                yield prev[0], prev[1], data
                prev = (t, con)
                data = [feature_i]
            else:
                data.append(feature_i)
        yield prev[0], prev[1], data



def load_weights(file_name):
    d_weights = []
    with open(file_name, "r") as infile:
        for line in infile:
            _, w_id, w = line.strip().split()
            d_weights.append(float(w))

    return np.array(d_weights)



def compute_lr_worker(process_queue, update_queue):
    dist_weights = np.array(d_weights)
    feature_dim = len(dist_weights)
    while True:
        buff = process_queue.get()
        if buff is None:
            return
        t, con, data = buff

        feature_matrix = lil_matrix((len(data), feature_dim), dtype=bool)
        for i, features_i in enumerate(data):
            feature_matrix.rows[i] = features_i
            feature_matrix.data[i] = [True] * len(features_i)
        feature_matrix = feature_matrix.tocsr()
        numerator = np.exp(feature_matrix.dot(dist_weights))
        update_queue.put((t, con, numerator))

def write_params(fh, t, con, start_i, values):
    for v in values:
        fh.write(" ".join([t, con, str(start_i), str(v)]) + "\n")
        start_i += 1


def aggregate_params(update_queue, convoc_files):
    start_index = defaultdict(list)
    dist_index = defaultdict(list)
    outfiles = []
    max_dists = defaultdict(max_dict)

    for i, f in enumerate(convoc_files):
        with open(f, "r") as infile:
            for line in infile:
                t, con, max_I = line.split()
                if t == "s":
                    start_index[con].append((i, int(max_I)))
                elif t == "j":
                    dist_index[con].append((i, int(max_I)))
                    max_dists[con].add(int(max_I))
        outfiles.append(open(f + ".params", "w"))

    while True:
        buff = update_queue.get()
        if buff is None:
            break
        t, con, numerator = buff

        if t == "s":
            for fid, max_I in start_index[con]:
                write_params(outfiles[fid],t, con, 0, numerator[:max_I])
        elif t == "j":
            all_I = (len(numerator) + 1) / 2
            actual_max = max_dists[con].get()
            if all_I < actual_max:
                diff_I = actual_max - all_I
                # jumps have been clipped
                temp_numerator = np.zeros((actual_max*2)-1)
                temp_numerator[diff_I:-diff_I] = numerator
                temp_numerator[:diff_I+1] = numerator[0] / (diff_I + 1)
                temp_numerator[-(diff_I+1):] = numerator[-1] / (diff_I+1)
                numerator = temp_numerator
                all_I = (len(numerator) + 1) / 2

            for fid, max_I in dist_index[con]:
                if all_I != max_I:
                    diff_I = all_I-max_I
                    write_params(outfiles[fid], t, con, diff_I, numerator[diff_I:-diff_I])
                else:
                    write_params(outfiles[fid], t, con, 0, numerator)


    for outfile in outfiles:
        outfile.close()
    logger.info("Parameters written.")
    return

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-convoc_list", required=True)
    arg_parser.add_argument("-weights", required=True)
    arg_parser.add_argument("-num_workers", default=8, type=int, required=False)
    args = arg_parser.parse_args()

    convoc_files = glob.glob("./*convoc.*")

    num_workers = max(1, args.num_workers - 1)

    aggregate_queue = mp.Queue()

    aggregator = mp.Process(target=aggregate_params, args=(aggregate_queue, convoc_files))
    aggregator.start()
    process_queue = mp.Queue(maxsize=num_workers*2)

    logger.info("Loading weights and vecs.")
    d_weights = load_weights(args.weights)

    logger.info("Starting workers.")
    pool = []
    for w in xrange(num_workers):
        p = mp.Process(target=compute_lr_worker, args=(process_queue, aggregate_queue))
        p.start()
        pool.append(p)

    convoc_list = convoc_reader(args.convoc_list)

    for buff in convoc_list:
        process_queue.put(buff)

    logger.info("Entire corpus loaded.")
    for p in pool:
        process_queue.put(None)
    logger.info("Waiting for processes to terminate.")
    for p in pool:
        p.join()
    aggregate_queue.put(None)
    aggregator.join()



