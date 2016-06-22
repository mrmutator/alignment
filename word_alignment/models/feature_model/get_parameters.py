from collections import defaultdict
import random
import numpy as np
import argparse
from CorpusReader import CorpusReader
import features
import imp
import multiprocessing as mp
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s  %(message)s')
logger = logging.getLogger()


class ArchiveReader(object):

    def __init__(self, file_name):
        self.f = open(file_name, "r")
        self.archive = dict()

    def get(self, sent_id, f, i_p):
        if (sent_id, f, i_p) in self.archive:
            fid = self.archive[sent_id, f, i_p]
            del self.archive[sent_id, f, i_p]
            return fid
        else:
            while True:
                s, j, ipp, fid = map(int, self.f.readline().strip().split())
                if s == sent_id and j == f and ipp == i_p:
                    return fid
                else:
                    self.archive[s, j, ipp] = fid


def reorder(data, order):
    """
    Order is a list with same length as data that specifies for each position of data, which rank it has in the new
    order.
    :param data:
    :param order:
    :return:
    """
    new_data = [None] * len(data)
    for i, j in enumerate(order):
        new_data[j] = data[i]
    return new_data


def hmm_reorder(f_toks, pos, rel, dir, order):
    # HMM reorder
    J = len(f_toks)
    new_f_toks = reorder(f_toks, order)
    new_pos = reorder(pos, order)
    new_rel = reorder(rel, order)
    new_dir = reorder(dir, order)
    new_f_heads = [0] + range(J - 1)
    new_order = range(J)
    return new_f_toks, new_f_heads, new_pos, new_rel, new_dir, new_order


def random_weight():
    return random.uniform(-1, 1)


class Parameters(object):
    def __init__(self, corpus, hmm=False, num_workers=1):
        self.num_workers = num_workers
        self.corpus = corpus
        self.cooc = set()
        self.hmm = hmm
        self.dist_cons = features.FeatureConditions()
        self.c = 0

        self.add_corpus(corpus)
        self.t_params = dict()
        self.dist_params = dict()

    def add_corpus(self, corpus):

        def aggregate_features(in_queue, out_queue):
            dist_features = features.Features()
            cons = features.FeatureConditions()
            add_features = dist_features.add
            outfile = open("all_feature_set_ids.txt", "w", buffering=-1)
            while True:
                token = in_queue.get()
                if token is None:
                    break
                for sent_id, j, i_p, fset  in token:
                    fid_set = set()
                    for fname in fset:
                        fid = add_features(fname)
                        fid_set.add(fid)
                    if sent_id is not None:
                        set_id = cons.get_id(frozenset(fid_set))
                        outfile.write(" ".join(map(str, [sent_id, j, i_p, set_id])) + "\n")

            outfile.close()
            out_queue.put((dist_features, cons))


        def extract_features(in_queue, out_queue):
            f_sets = set()
            while True:
                token = in_queue.get()
                if token is None:
                    break
                sent_id, e_toks, f_toks, f_heads, pos, rel, dir, order = token

                I = len(e_toks)
                I_ = 1
                for j, f in enumerate(f_toks):
                    for i_p in xrange(I_):
                        static_set = set()
                        for feat_name in extract_static_features(e_toks, f_toks, f_heads, pos, rel, order, j, i_p):
                            static_set.add(feat_name)
                        f_sets.add((sent_id, j, i_p, frozenset(static_set)))

                        dynamic_set = set()
                        for i in xrange(I):
                            for feat_name in extract_dynamic_features(e_toks, f_toks, f_heads, order, j, i_p, i):
                                dynamic_set.add(feat_name)
                            f_sets.add((None, None, None, frozenset(dynamic_set)))
                    I_ = I

                if len(f_sets) > 90:
                    out_queue.put(f_sets)
                    f_sets = set()
            if len(f_sets) > 0:
                out_queue.put(f_sets)

        logger.info("Collecting features from data.")
        feature_extraction_queue = mp.Queue(maxsize=1000)
        aggregate_queue = mp.Queue()
        result_queue = mp.Queue()
        aggregation_process = mp.Process(target=aggregate_features, args=(aggregate_queue, result_queue))
        aggregation_process.start()
        pool = []
        for _ in xrange(max(self.num_workers-2, 1)):
            extract_feature_process = mp.Process(target=extract_features, args=(feature_extraction_queue, aggregate_queue))
            extract_feature_process.start()
            pool.append(extract_feature_process)

        self.c = 0
        c = 0
        for e_toks, f_toks, f_heads, pos, rel, dir, order in corpus:
            self.c += 1
            c += 1
            if self.hmm:
                f_toks, f_heads, pos, rel, dir, order = hmm_reorder(f_toks, pos, rel, dir, order)

            feature_extraction_queue.put((c, e_toks, f_toks, f_heads, pos, rel, dir, order))

            # lexical parameters
            for f in f_toks:
                for e in e_toks + [0]:
                    self.cooc.add((e, f))
        logger.info("Entire corpus loaded.")
        for _ in pool:
            feature_extraction_queue.put(None)
        for p in pool:
            p.join()
        aggregate_queue.put(None)
        aggregation_process.join()

        self.dist_features, self.dist_cons = result_queue.get()
        logger.info("Feature collection completed.")


    def initialize_trans_t_file(self, t_file):
        trans_dict = defaultdict(dict)
        with open(t_file, "r") as infile:
            for line in infile:
                e, f, p = line.strip().split()
                e = int(e)
                f = int(f)
                if (e, f) in self.cooc:
                    trans_dict[e][f] = float(p)
        for e in trans_dict:
            Z = np.sum(trans_dict[e].values())
            for f in trans_dict[e]:
                self.t_params[(e, f)] = trans_dict[e][f] / float(Z)
        del self.cooc

    def initialize_dist_weights(self):
        for wid in xrange(self.dist_features.feature_num):
            self.dist_params[wid] = random_weight()

    def write_weight_file(self, out_file_name):
        with open(out_file_name, "w") as outfile:
            for w_id, w in sorted(self.dist_params.iteritems(), key=lambda t: t[0]):
                outfile.write("w " + str(w_id) + " " + str(w) + "\n")

    def write_params(self, sub_t, out_file_name):
        with open(out_file_name, "w") as outfile:
            for key in sub_t:
                value = self.t_params[key]
                key_str = ["t"] + map(str, [key[0], key[1], value])
                outfile.write(" ".join(key_str) + "\n")

    def write_cons(self, sub_cons, outfile_name):
        with open(outfile_name, "w") as outfile:
            for cond_id in sub_cons:
                feature_set = self.dist_cons.get_feature_set(cond_id)
                outfile.write("cid " + str(cond_id) + " " + " ".join(map(str, feature_set)) + "\n")

    def split_data_get_parameters(self, corpus, file_prefix, num_sentences):
        logger.info("Splitting corpus.")
        archive_reader = ArchiveReader("all_feature_set_ids.txt")
        subset_id = 1
        outfile_corpus = open(file_prefix + ".corpus." + str(subset_id), "w")
        sub_t = set()
        sub_cons = set()
        subset_c = 0
        total = 0
        for e_toks, f_toks, f_heads, pos, rel, dir, order in corpus:
            subset_c += 1
            total += 1
            if self.hmm:
                f_toks, f_heads, pos, rel, dir, order = hmm_reorder(f_toks, pos, rel, dir, order)

            # produce subcorpus file
            outfile_corpus.write(" ".join(map(str, e_toks)) + "\n")
            outfile_corpus.write(" ".join(map(str, f_toks)) + "\n")
            outfile_corpus.write(" ".join(map(str, f_heads)) + "\n")
            outfile_corpus.write(" ".join(map(str, order)) + "\n")

            I = len(e_toks)
            I_ = 1
            # feature extraction

            for j, f in enumerate(f_toks):
                con_ids = []
                for i_p in xrange(I_):
                    con_id = archive_reader.get(total, j, i_p)
                    con_ids.append(con_id)
                    sub_cons.add(con_id)
                outfile_corpus.write(" ".join(map(str, con_ids)) + "\n")

                # lexical features
                for e in e_toks + [0]:
                    if (e, f) in self.t_params:
                        sub_t.add((e, f))

                I_ = I

            outfile_corpus.write("\n")

            if subset_c == num_sentences:
                outfile_corpus.close()
                self.write_params(sub_t, file_prefix + ".params." + str(subset_id))
                self.write_cons(sub_cons, file_prefix + ".cons." + str(subset_id))
                if total < self.c:
                    subset_id += 1
                    outfile_corpus = open(file_prefix + ".corpus." + str(subset_id), "w")
                    sub_t = set()
                    sub_cons = set()
                    subset_c = 0
        if subset_c > 0:
            outfile_corpus.close()
            self.write_params(sub_t, file_prefix + ".params." + str(subset_id))
            self.write_cons(sub_cons, file_prefix + ".cons." + str(subset_id))


def prepare_data(corpus, t_file, num_sentences, file_prefix="", hmm=False, num_workers=1):
    parameters = Parameters(corpus, hmm=hmm, num_workers=num_workers)
    parameters.initialize_trans_t_file(t_file)
    parameters.initialize_dist_weights()

    parameters.split_data_get_parameters(corpus, file_prefix, num_sentences)
    logger.info("Writing parameters.")
    with open(file_prefix + ".weight_voc", "w") as outfile:
        outfile.write(parameters.dist_features.get_voc())

    with open(file_prefix + ".dist_cons", "w") as outfile:
        outfile.write(parameters.dist_cons.get_voc())

    parameters.write_weight_file(file_prefix + ".weights")
    logger.info("Done")

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-features", required=True)
    arg_parser.add_argument("-corpus", required=True)
    arg_parser.add_argument("-t_file", required=True)
    arg_parser.add_argument("-output_prefix", required=True)
    arg_parser.add_argument("-limit", required=False, type=int, default=0)
    arg_parser.add_argument("-group_size", required=False, type=int, default=-1)
    arg_parser.add_argument('-hmm', dest='hmm', action='store_true', default=False)
    arg_parser.add_argument('-num_workers', required=False, type=int, default=8)
    args = arg_parser.parse_args()

    defined_features = imp.load_source("features", args.features)
    extract_static_features = defined_features.extract_static_dist_features
    extract_dynamic_features = defined_features.extract_dynamic_dist_features
    corpus = CorpusReader(args.corpus, limit=args.limit)

    prepare_data(corpus=corpus, t_file=args.t_file, num_sentences=args.group_size, file_prefix=args.output_prefix,
                 hmm=args.hmm, num_workers=args.num_workers)
