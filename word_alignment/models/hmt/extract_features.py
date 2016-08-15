import argparse
from CorpusReader import CorpusReader
import numpy as np
from collections import defaultdict
from features import max_dict, FeatureStore, ExtractedFeatures
import multiprocessing as mp


def extract_features(feature_pool, process_queue, write_queue, result_queue):
    all_jmp_condition_ids = defaultdict(max_dict)
    all_start_condition_ids = defaultdict(max_dict)

    while True:
        buff = process_queue.get()
        if buff is None:
            result_queue.put((all_start_condition_ids, all_jmp_condition_ids))
            return

        num, (e_toks, f_toks, f_heads, pos, rel, hmm_transitions, order) = buff
        out = "\n".join([" ".join(map(str, e_toks)), " ".join(map(str, f_toks)), " ".join(map(str, f_heads)), ""])

        J = len(f_toks)
        I = len(e_toks)

        l = [0] + [order[j] - order[f_heads[j]] for j in xrange(1, J)]
        dir = map(np.sign, l)


        # sentence_level
        features_sentence_level = ExtractedFeatures(feature_pool)
        features_sentence_level.add_feature(("I", I))


        # j=0
        j=0
        features_0 = ExtractedFeatures(feature_pool, features_sentence_level)
        features_0.add_feature(("cpos", pos[j]))
        features_0.add_feature(("crel", rel[j]))
        features_0.add_exclusive_feature(("j", j))

        conditions = features_0.get_feature_ids()
        if not conditions:
            conditions = frozenset([0])
        condition_id = "-".join(map(str, sorted(conditions)))
        all_start_condition_ids[condition_id].add(I)

        # do all the writing here
        out += condition_id + "\n"


        # rest
        for j in xrange(1, J):
            features_j = ExtractedFeatures(feature_pool, features_sentence_level)
            features_j.add_feature(("j", j))
            # add features
            h = f_heads[j]
            features_j.add_feature(("pl", l[h]))
            features_j.add_feature(("ppos", pos[h]))
            features_j.add_feature(("prel", rel[h]))
            features_j.add_feature(("pdir", dir[h]))
            features_j.add_feature(("cpos", pos[j]))
            features_j.add_feature(("crel", rel[j]))
            features_j.add_feature(("cdir", dir[j]))
            features_j.add_feature(("cl", l[j]))
            features_j.add_feature(("phmm", hmm_transitions[h]))
            features_j.add_feature(("chmm", hmm_transitions[j]))


            for i_p in xrange(I):
                features_i_p = ExtractedFeatures(feature_pool, features_j)
                # static features complete
                # check if there is a feature that matches
                conditions = features_i_p.get_feature_ids()
                if not conditions:
                    conditions = frozenset([0])
                condition_id = "-".join(map(str, sorted(conditions)))
                all_jmp_condition_ids[condition_id].add(I)

                # do all the wirting here
                out += condition_id + "\n"


        write_queue.put((num, out))




def write_corpus(queue):
    outfile = open(args.corpus + ".extracted", "w")
    buffer_dict = dict()
    next_c = 0
    while True:
        if next_c in buffer_dict:
            tmp = buffer_dict[next_c]
            outfile.write(tmp + "\n")
            del buffer_dict[next_c]
            next_c += 1
            continue

        obj = queue.get()
        if obj is None:
            break
        num, tmp_string = obj
        if num == next_c:
            outfile.write(tmp_string + "\n")
            next_c += 1
        else:
            buffer_dict[num] = tmp_string

    while len(buffer_dict) > 0:
        tmp_string = buffer_dict[next_c]
        outfile.write(tmp_string + "\n")
        del buffer_dict[next_c]
        next_c += 1

    outfile.close()

def aggregate(result_queue):
    all_jmp_condition_ids = defaultdict(max_dict)
    all_start_condition_ids = defaultdict(max_dict)
    while True:
        buff = result_queue.get()
        if buff is None:
            break
        start_condition_ids, jmp_condition_ids = buff
        for cid in start_condition_ids:
            max_I = start_condition_ids[cid].get()
            all_start_condition_ids[cid].add(max_I)
        for cid in jmp_condition_ids:
            max_I = jmp_condition_ids[cid].get()
            all_jmp_condition_ids[cid].add(max_I)


    with open(args.corpus + ".convoc", "w") as outfile:
        for cid in all_start_condition_ids:
            max_I = all_start_condition_ids[cid].get()
            outfile.write(" ".join(["s", cid, str(max_I)]) + "\n")

        for cid in all_jmp_condition_ids:
            max_I = all_jmp_condition_ids[cid].get()
            outfile.write(" ".join(["j", cid, str(max_I)]) + "\n")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-corpus", required=True)
    arg_parser.add_argument("-feature_file", required=True)
    arg_parser.add_argument("-num_workers", required=False, type=int, default=4)
    args = arg_parser.parse_args()
    corpus = CorpusReader(args.corpus)

    feature_pool = FeatureStore(args.feature_file)


    with open(args.corpus + ".cons", "w") as outfile:
        outfile.write(feature_pool.get_voc())

    num_workers = max(1, args.num_workers - 1)

    write_queue = mp.Queue()
    result_queue = mp.Queue()
    process_queue = mp.Queue(maxsize=num_workers*2)


    pool = []
    for w in xrange(num_workers):
        p = mp.Process(target=extract_features, args=(feature_pool, process_queue, write_queue, result_queue))
        p.start()
        pool.append(p)

    writer = mp.Process(target=write_corpus, args=(write_queue,))
    writer.start()

    aggregator = mp.Process(target=aggregate, args=(result_queue,))
    aggregator.start()

    for buff in enumerate(corpus):
        process_queue.put(buff)

    for _ in pool:
        process_queue.put(None)

    for p in pool:
        p.join()

    write_queue.put(None)
    result_queue.put(None)
    writer.join()
    aggregator.join()
