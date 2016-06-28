from collections import defaultdict
import random
import numpy as np
import argparse
from CorpusReader import CorpusReader

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
    def __init__(self, corpus, hmm=False):
        self.corpus = corpus
        self.cooc = set()
        self.hmm = hmm
        self.c = 0

        self.add_corpus(corpus)
        self.t_params = dict()

    def add_corpus(self, corpus):
        self.c = 0
        for e_toks, f_toks, f_heads, pos, rel, dir, order in corpus:
            self.c += 1
            if self.hmm:
                f_toks, f_heads, pos, rel, dir, order = hmm_reorder(f_toks, pos, rel, dir, order)


            for j, f in enumerate(f_toks):
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


    def write_params(self, sub_t, out_file_name):
        with open(out_file_name, "w") as outfile:
            for key in sub_t:
                value = self.t_params[key]
                key_str = ["t"] + map(str, [key[0], key[1], value])
                outfile.write(" ".join(key_str) + "\n")

    def split_data_get_parameters(self, corpus, file_prefix, num_sentences, num_sentences_feature_extraction):
        subset_id = 1
        feature_id = 1
        outfile_corpus = open(file_prefix + "." + str(feature_id) + ".sub_feat", "w")
        order_file = open(file_prefix + ".order", "w")
        sub_t = set()
        subset_c = 0
        total = 0
        feature_c = 0
        for e_toks, f_toks, f_heads, pos, rel, dir, order in corpus:
            subset_c += 1
            feature_c += 1
            total += 1
            if self.hmm:
                f_toks, f_heads, pos, rel, dir, order = hmm_reorder(f_toks, pos, rel, dir, order)


            # produce subcorpus file
            outfile_corpus.write(" ".join(map(str, e_toks)) + "\n")
            outfile_corpus.write(" ".join(map(str, f_toks)) + "\n")
            outfile_corpus.write(" ".join(map(str, f_heads)) + "\n")
            outfile_corpus.write(" ".join(map(str, pos)) + "\n")
            outfile_corpus.write(" ".join(map(str, rel)) + "\n")
            outfile_corpus.write("0\n") # dummy dir
            outfile_corpus.write(" ".join(map(str, order)) + "\n")
            outfile_corpus.write("\n")
            order_file.write(" ".join(map(str, order)) + "\n")

            for j, f in enumerate(f_toks):
                for e in e_toks + [0]:
                    if (e, f) in self.t_params:
                        sub_t.add((e, f))

            if subset_c == num_sentences:
                self.write_params(sub_t, file_prefix + ".params." + str(subset_id))
                subset_id += 1
                sub_t = set()
                subset_c = 0

            if feature_c == num_sentences_feature_extraction:
                outfile_corpus.close()
                feature_c = 0
                feature_id += 1
                if total < self.c:
                    outfile_corpus = open(file_prefix + "." + str(feature_id) + ".sub_feat", "w")

        if subset_c > 0:
            self.write_params(sub_t, file_prefix + ".params." + str(subset_id))
        if feature_c > 0:
            outfile_corpus.close()
        order_file.close()


def prepare_data(corpus, t_file, num_sentences, num_feature_sentences, file_prefix="", hmm=False):
    parameters = Parameters(corpus, hmm=hmm)
    parameters.initialize_trans_t_file(t_file)

    parameters.split_data_get_parameters(corpus, file_prefix, num_sentences, num_feature_sentences)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-corpus", required=True)
    arg_parser.add_argument("-t_file", required=True)
    arg_parser.add_argument("-output_prefix", required=True)
    arg_parser.add_argument("-limit", required=False, type=int, default=0)
    arg_parser.add_argument("-group_size", required=False, type=int, default=-1)
    arg_parser.add_argument('-hmm', dest='hmm', action='store_true', default=False)
    arg_parser.add_argument('-group_size_feature_extraction', required=False, default=-1, type=int)
    args = arg_parser.parse_args()

    corpus = CorpusReader(args.corpus, limit=args.limit)

    prepare_data(corpus=corpus, t_file=args.t_file, num_sentences=args.group_size,
                 num_feature_sentences=args.group_size_feature_extraction, file_prefix=args.output_prefix,
                 hmm=args.hmm)
