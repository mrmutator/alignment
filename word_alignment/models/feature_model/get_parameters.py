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
        self.dist_features = features.Features(extract_static_func=features.extract_static_dist_features, extract_dynamic_func=features.extract_dynamic_dist_features)
        self.dist_cons = features.FeatureConditions()
        self.c = 0

        self.add_corpus(corpus)
        self.t_params = dict()
        self.dist_params = dict()

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
                for i_p in xrange(I):
                    for feat_name in self.dist_features.extract_static(e_toks, f_toks, f_heads, pos, rel, order, j, i_p):
                        self.dist_features.add(feat_name)
                    for i in xrange(I):
                        for feat_name in self.dist_features.extract_dynamic(e_toks, f_toks, f_heads, pos, rel, order, j, i_p, i):
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

    def initialize_dist_weights(self):
        for wid in xrange(self.dist_features.feature_num):
            self.dist_params[wid] = random_weight()

    def write_weight_file(self, out_file_name):
        with open(out_file_name, "w") as outfile:
            # dist params
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
                outfile.write("cid " + str(cond_id) + " " +  " ".join(map(str, feature_set)) + "\n")

    def split_data_get_parameters(self, corpus, file_prefix, num_sentences):
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

            I = len(e_toks)

            # produce subcorpus file
            outfile_corpus.write(" ".join([str(w) for w in e_toks]) + "\n")
            outfile_corpus.write(" ".join([str(w) for w in f_toks]) + "\n")
            outfile_corpus.write(" ".join([str(h) for h in f_heads]) + "\n")
            outfile_corpus.write(" ".join([str(o) for o in order]) + "\n")

            # feature extraction
            for j, f in enumerate(f_toks):
                if j == 0:
                    I_ = 1
                else:
                    I_ = I
                for i_p in xrange(I_):
                    static_features = self.dist_features.extract_static(e_toks, f_toks, f_heads, pos, rel, order, j, i_p)
                    static_set = frozenset(map(self.dist_features.get_feat_id, static_features))
                    static_cond_id = self.dist_cons.get_id(static_set)
                    sub_cons.add(static_cond_id)
                    dynamic_i_set = []
                    for i in xrange(I):
                        dynamic_features = self.dist_features.extract_dynamic(e_toks, f_toks, f_heads, pos, rel, order, j, i_p, i)
                        dynamic_set =  frozenset(map(self.dist_features.get_feat_id, dynamic_features))
                        dynamic_cond_id = self.dist_cons.get_id(dynamic_set)
                        sub_cons.add(dynamic_cond_id)
                        dynamic_i_set.append(dynamic_cond_id)
                    outfile_corpus.write(str(static_cond_id) + " " + " ".join(map(str, dynamic_i_set))+ "\n")

                # lexical features
                for e in e_toks + [0]:
                    if (e, f) in self.t_params:
                        sub_t.add((e, f))
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


def prepare_data(corpus, t_file, num_sentences, file_prefix="", hmm=False):
    parameters = Parameters(corpus, hmm=hmm)
    parameters.initialize_trans_t_file(t_file)
    parameters.initialize_dist_weights()

    parameters.split_data_get_parameters(corpus, file_prefix, num_sentences)
    with open(file_prefix + ".dist_feat_voc", "w") as outfile:
        outfile.write(parameters.dist_features.get_voc())

    with open(file_prefix + ".dist_cons", "w") as outfile:
        outfile.write(parameters.dist_cons.get_voc())

    parameters.write_weight_file(file_prefix + ".weights")

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
