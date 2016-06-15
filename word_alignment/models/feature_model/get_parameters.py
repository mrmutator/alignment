from collections import defaultdict
import random
import numpy as np
import argparse
from CorpusReader import CorpusReader
import features


def reorder(data, order):
    """
    Order is a list with same length as data that specifies for each position of data, which rank it has in the new order.
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


def random_prob():
    return random.random() * -1 + 1  # random number between 0 and 1, excluding 0, including 1


def random_weight():
    return random.uniform(-1, 1)


class Parameters(object):
    def __init__(self, corpus, hmm=False):
        self.corpus = corpus
        self.cooc = set()
        self.global_max_I = 0
        self.hmm = hmm

        self.start_features = features.Features(extract_func=features.extract_start_features)
        self.dist_features = features.Features(extract_func=features.extract_dist_features)
        self.c = 0

        self.add_corpus(corpus)
        self.t_params = dict()

    def add_corpus(self, corpus):
        self.c = 0
        for e_toks, f_toks, f_heads, pos, rel, dir, order in corpus:
            self.c += 1
            if self.hmm:
                f_toks, f_heads, pos, rel, dir, order = hmm_reorder(f_toks, pos, rel, dir, order)

            I = len(e_toks)
            if I > self.global_max_I:
                self.global_max_I = I

            for j, f in enumerate(f_toks):
                if j == 0:
                    feat_set = self.start_features.extract(e_toks, f_toks, f_heads, pos, rel, order)
                    for feat_name in feat_set:
                        self.start_features.add(feat_name)
                else:
                    for i_p in xrange(I):
                        for feat_name in self.dist_features.extract(e_toks, f_toks, f_heads, pos, rel, order, j, i_p):
                            self.dist_features.add(feat_name)

                # lexical parameters
                for e in e_toks + [0]:
                    self.cooc.add((e, f))


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

    def write_params(self, max_I, sub_t, out_file_name):
        outfile = open(out_file_name, "w")

        # lexical parameters
        for key in sub_t:
            value = self.t_params[key]
            key_str = ["t"] + map(str, [key[0], key[1], value])
            outfile.write(" ".join(key_str) + "\n")

        # distorion parameters
        for jmp in range(-max_I + 1, max_I):
            for weight_id in xrange(self.dist_features.feature_num):
                value = random_weight()
                key_str = ["dw"] + map(str, [jmp, weight_id, value])
                outfile.write(" ".join(key_str) + "\n")

        # start parameters
        for i in range(max_I):
            for weight_id in xrange(self.start_features.feature_num):
                value = random_weight()
                key_str = ["sw"] + map(str, [i, weight_id, value])
                outfile.write(" ".join(key_str) + "\n")

        outfile.close()

    def split_data_get_parameters(self, corpus, file_prefix, num_sentences):
        subset_id = 1
        outfile_corpus = open(file_prefix + ".corpus." + str(subset_id), "w")
        sub_t = set()
        sub_I = set()
        subset_c = 0
        total = 0
        for e_toks, f_toks, f_heads, pos, rel, dir, order in corpus:
            subset_c += 1
            total += 1
            if self.hmm:
                f_toks, f_heads, pos, rel, dir, order = hmm_reorder(f_toks, pos, rel, dir, order)

            # feature extraction

            I = len(e_toks)
            sub_I.add(I)

            # produce subcorpus file
            outfile_corpus.write(" ".join([str(w) for w in e_toks]) + "\n")
            outfile_corpus.write(" ".join([str(w) for w in f_toks]) + "\n")
            outfile_corpus.write(" ".join([str(h) for h in f_heads]) + "\n")
            outfile_corpus.write(" ".join([str(p) for p in pos]) + "\n")
            outfile_corpus.write(" ".join([str(r) for r in rel]) + "\n")
            outfile_corpus.write(" ".join([str(o) for o in order]) + "\n")


            for j, f in enumerate(f_toks):
                if j == 0:
                    # start features
                    feat_set = self.start_features.extract(e_toks, f_toks, f_heads, pos, rel, order)
                    outfile_corpus.write(str(j) + "\t" + " ".join(map(str, map(self.start_features.get_feat_id, feat_set))) + "\n")
                else:
                    # static dist features
                    for i_p in xrange(I):
                        feat_set = self.dist_features.extract(e_toks, f_toks, f_heads, pos, rel, order, j, i_p)
                        outfile_corpus.write(str(j) + "," + str(i_p) + "\t" + " ".join(map(str, map(self.dist_features.get_feat_id, feat_set))) + "\n")


                # lexical features
                for e in e_toks + [0]:
                    if (e, f) in self.t_params:
                        sub_t.add((e, f))
            outfile_corpus.write("\n")

            if subset_c == num_sentences:
                outfile_corpus.close()
                self.write_params(max(sub_I), sub_t, file_prefix + ".params." + str(subset_id))
                if total < self.c:
                    subset_id += 1
                    outfile_corpus = open(file_prefix + ".corpus." + str(subset_id), "w")
                    sub_t = set()
                    sub_I = set()
                    subset_c = 0
        if subset_c > 0:
            outfile_corpus.close()
            self.write_params(max(sub_I), sub_t, file_prefix + ".params." + str(subset_id))


def prepare_data(corpus, t_file, num_sentences, file_prefix="", hmm=False):
    parameters = Parameters(corpus, hmm=hmm)
    parameters.initialize_trans_t_file(t_file)

    parameters.split_data_get_parameters(corpus, file_prefix, num_sentences)
    with open(file_prefix + ".dist_feat_voc", "w") as outfile:
        outfile.write(parameters.dist_features.get_voc())


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-corpus", required=True)
    arg_parser.add_argument("-output_prefix", required=True)
    arg_parser.add_argument("-t_file", required=True)
    arg_parser.add_argument("-limit", required=False, type=int, default=0)
    arg_parser.add_argument("-group_size", required=False, type=int, default=-1)
    arg_parser.add_argument('-hmm', dest='hmm', action='store_true', default=False)
    args = arg_parser.parse_args()

    corpus = CorpusReader(args.corpus, limit=args.limit)

    prepare_data(corpus=corpus, t_file=args.t_file, num_sentences=args.group_size, file_prefix=args.output_prefix,
                 hmm=args.hmm)
