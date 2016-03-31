from collections import defaultdict
import random
import numpy as np
import argparse
from CorpusReader import CorpusReader


def random_prob():
    return random.random() * -1 + 1  # random number between 0 and 1, excluding 0, including 1


class Parameters(object):
    def __init__(self, corpus, p_0=0.2):
        self.corpus = corpus
        self.cooc = set()
        self.lengths = set()
        self.p_0 = p_0

        self.c = 0
        self.add_corpus(corpus)

        self.t_params = dict()
        self.j_params = dict()
        self.s_params = dict()

    def add_corpus(self, corpus):
        self.c = 0
        for e_toks, f_toks, _, _ in corpus:
            self.c += 1
            I = len(e_toks)
            self.lengths.add(I)
            for f in f_toks:
                self.cooc.add((0, f))
                for e in e_toks:
                    self.cooc.add((e, f))

    def initialize_start_randomly(self):
        for I in self.lengths:
            Z = 0
            start = np.zeros(I)
            for i in xrange(I):
                p = random_prob()
                start[i] = p
                Z += p
            Z += self.p_0  # p_0 for null word at start
            start = start / Z
            for i in xrange(I):
                self.s_params[(I, i)] = start[i]

    def initialize_start_uniformly(self):
        for I in self.lengths:
            Z = 0
            start = (1.0 - self.p_0) / I
            for i in xrange(I):
                self.s_params[(I, i)] = start

    def initialize_dist_randomly(self):
        for jmp in xrange(-max(self.lengths) + 1, max(self.lengths)):
            self.j_params[jmp] = random_prob()

    def initialize_dist_uniformly(self):
        for jmp in xrange(-max(self.lengths) + 1, max(self.lengths)):
            self.j_params[jmp] = 0.5  # number doesn't matter

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

    def write_params(self, sub_lengths, sub_t, out_file_name):
        outfile = open(out_file_name, "w")
        for key in sub_t:
            value = self.t_params[key]
            key_str = ["t"] + map(str, [key[0], key[1], value])
            outfile.write(" ".join(key_str) + "\n")

        for I in sub_lengths:
            for i in xrange(I):
                value = self.s_params[(I, i)]
                key_str = ["s"] + map(str, [I, i, value])
                outfile.write(" ".join(key_str) + "\n")
        max_I = max(sub_lengths)
        for jmp in xrange(-max_I + 1, max_I):
            value = self.j_params[jmp]
            key_str = ["j"] + map(str, [jmp, value])
            outfile.write(" ".join(key_str) + "\n")

        outfile.close()

    def split_data_get_parameters(self, corpus, file_prefix, num_sentences):
        subset_id = 1
        outfile_corpus = open(file_prefix + ".corpus." + str(subset_id), "w")
        sub_t = set()
        sub_lengths = set()
        subset_c = 0
        total = 0
        for e_toks, f_toks, f_heads, order in corpus:
            subset_c += 1
            total += 1
            outfile_corpus.write(" ".join([str(w) for w in e_toks]) + "\n")
            outfile_corpus.write(" ".join([str(w) for w in f_toks]) + "\n")
            outfile_corpus.write(" ".join([str(h) for h in f_heads]) + "\n")
            outfile_corpus.write(" ".join([str(o) for o in order]) + "\n\n")
            # add parameters
            I = len(e_toks)
            sub_lengths.add(I)
            for e in e_toks + [0]:
                for f in f_toks:
                    if (e, f) in self.t_params:
                        sub_t.add((e, f))

            if subset_c == num_sentences:
                outfile_corpus.close()
                self.write_params(sub_lengths, sub_t, file_prefix + ".params." + str(subset_id))
                if total < self.c:
                    subset_id += 1
                    outfile_corpus = open(file_prefix + ".corpus." + str(subset_id), "w")
                    params_subset = set()
                    subset_c = 0
        if subset_c > 0:
            outfile_corpus.close()
            self.write_params(sub_lengths, sub_t, file_prefix + ".params." + str(subset_id))


def prepare_data(corpus, t_file, num_sentences, p_0=0.2, file_prefix="", random=False):
    parameters = Parameters(corpus, p_0=p_0)

    if random:
        parameters.initialize_dist_randomly()
        parameters.initialize_start_randomly()
    else:
        parameters.initialize_dist_uniformly()
        parameters.initialize_start_uniformly()

    parameters.initialize_trans_t_file(t_file)

    parameters.split_data_get_parameters(corpus, file_prefix, num_sentences)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-corpus", required=True)
    arg_parser.add_argument("-output_prefix", required=True)
    arg_parser.add_argument("-t_file", required=True)
    arg_parser.add_argument("-limit", required=False, type=int, default=0)
    arg_parser.add_argument("-group_size", required=False, type=int, default=-1)
    arg_parser.add_argument("-p_0", required=False, default=0.2, type=float)

    args = arg_parser.parse_args()

    corpus = CorpusReader(args.corpus, limit=args.limit)

    prepare_data(corpus=corpus, t_file=args.t_file, num_sentences=args.group_size, p_0=args.p_0,
                 file_prefix=args.output_prefix, random=False)
