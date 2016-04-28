from collections import defaultdict
import random
import numpy as np
import argparse
from CorpusReader import CorpusReader
from scipy.stats import norm

def reorder(f_toks, pos, rel, dir, order):
    J = len(f_toks)
    new_f_toks = [None] * J
    new_pos = [None] * J
    new_rel = [None] * J
    new_dir = [None] * J
    for j in xrange(J):
        i = order[j]
        new_f_toks[i] = f_toks[j]
        new_pos[i] = pos[j]
        new_rel[i] = rel[j]
        new_dir[i] = dir[j]
    new_f_heads = [0] + range(J-1)
    new_order = range(J)
    return new_f_toks, new_f_heads, new_pos, new_rel, new_dir, new_order

class CondVoc(object):

    def __init__(self):
        self.i2v = dict()
        self.v2i = dict()
        self.i = 0

    def get_id(self, w):
        if w not in self.v2i:
            i = self.i
            self.i2v[i] = w
            self.v2i[w] = i
            self.i += 1
        return self.v2i[w]

    def get_voc(self):
        string = ""
        for i in sorted(self.i2v):
            string += str(i) + "\t" + str(self.i2v[i]) + "\n"
        return string


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
        self.cond_voc = CondVoc()

    def add_corpus(self, corpus):
        self.c = 0
        for e_toks, f_toks, _, _, _, _, _ in corpus:
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

    def initialize_jumps_smartly(self, scale=1):
        jmps = range(-max(self.lengths) + 1, max(self.lengths))
        ps = norm.pdf(jmps, loc=0, scale=scale)
        ps = ps / np.sum(ps)
        for k in xrange(len(ps)):
            self.j_params[jmps[k]] = ps[k]

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

    def write_params(self, sub_lengths_pos, sub_lengths, sub_t, out_file_name):
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

        for pos in sub_lengths_pos:
            max_I = max(sub_lengths_pos[pos])
            for jmp in xrange(-max_I + 1, max_I):
                value = self.j_params[jmp]
                key_str = ["j"] + map(str, [pos, jmp, value])
                outfile.write(" ".join(key_str) + "\n")

        outfile.close()

    def split_data_get_parameters(self, corpus, file_prefix, num_sentences, head_con="", tok_con="", hmm=False):
        subset_id = 1
        outfile_corpus = open(file_prefix + ".corpus." + str(subset_id), "w")
        sub_t = set()
        sub_lengths_pos = defaultdict(set)
        sub_lengths = set()
        subset_c = 0
        total = 0
        for e_toks, f_toks, f_heads, pos, rel, dir, order in corpus:
            subset_c += 1
            total += 1
            if hmm:
                f_toks, f_heads, pos, rel, dir, order = reorder(f_toks, pos, rel, dir, order)
            outfile_corpus.write(" ".join([str(w) for w in e_toks]) + "\n")
            outfile_corpus.write(" ".join([str(w) for w in f_toks]) + "\n")
            outfile_corpus.write(" ".join([str(h) for h in f_heads]) + "\n")

            # add parameters
            I = len(e_toks)
            sub_lengths.add(I)
            conditions = [-1]
            for j in range(1, len(f_toks)):
                par = f_heads[j]
                orig_tok_pos = order[j]
                orig_head_pos = order[par]
                parent_distance = abs(orig_head_pos-orig_tok_pos)
                con_tok = [None, None, None, None, None, None, None]
                if "p" in head_con:
                    con_tok[0] =  pos[par]
                if "r" in head_con:
                    con_tok[1] = rel[par]
                if "d" in head_con:
                    con_tok[2] = dir[par]
                if "p" in tok_con:
                    con_tok[3] = pos[j]
                if "r" in tok_con:
                    con_tok[4] = rel[j]
                if "d" in tok_con:
                    con_tok[5] = dir[j]
                if "l" in tok_con:
                    con_tok[6] = parent_distance

                cond_id = self.cond_voc.get_id(tuple(con_tok))
                conditions.append(cond_id)
                sub_lengths_pos[cond_id].add(I)
            outfile_corpus.write(" ".join([str(c) for c in conditions]) + "\n")
            outfile_corpus.write(" ".join([str(o) for o in order]) + "\n\n")
            for e in e_toks + [0]:
                for f in f_toks:
                    if (e, f) in self.t_params:
                        sub_t.add((e, f))

            if subset_c == num_sentences:
                outfile_corpus.close()
                self.write_params(sub_lengths_pos, sub_lengths, sub_t, file_prefix + ".params." + str(subset_id))
                if total < self.c:
                    subset_id += 1
                    outfile_corpus = open(file_prefix + ".corpus." + str(subset_id), "w")
                    sub_lengths_pos = defaultdict(set)
                    sub_lengths = set()
                    sub_t = set()
                    subset_c = 0
        if subset_c > 0:
            outfile_corpus.close()
            self.write_params(sub_lengths_pos, sub_lengths, sub_t, file_prefix + ".params." + str(subset_id))


def prepare_data(corpus, t_file, num_sentences, p_0=0.2, file_prefix="", init='u', cond_head="", cond_tok="", hmm=False):
    parameters = Parameters(corpus, p_0=p_0)

    if init=="r":
        parameters.initialize_dist_randomly()
        parameters.initialize_start_randomly()
    elif init=="u":
        parameters.initialize_dist_uniformly()
        parameters.initialize_start_uniformly()

    elif init.startswith("s"):
        scale = float(init[1:])
        parameters.initialize_start_uniformly()
        parameters.initialize_jumps_smartly(scale=scale)

    parameters.initialize_trans_t_file(t_file)

    parameters.split_data_get_parameters(corpus, file_prefix, num_sentences, head_con=cond_head, tok_con=cond_tok, hmm=hmm)
    with open(file_prefix+ ".condvoc", "w") as outfile:
        outfile.write(parameters.cond_voc.get_voc())


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-corpus", required=True)
    arg_parser.add_argument("-output_prefix", required=True)
    arg_parser.add_argument("-t_file", required=True)
    arg_parser.add_argument("-limit", required=False, type=int, default=0)
    arg_parser.add_argument("-group_size", required=False, type=int, default=-1)
    arg_parser.add_argument("-p_0", required=False, default=0.2, type=float)
    arg_parser.add_argument("-cond_tok", required=False, default="", type=str)
    arg_parser.add_argument("-cond_head", required=False, default="", type=str)
    arg_parser.add_argument('-hmm', dest='hmm', action='store_true', default=False)
    arg_parser.add_argument('-init', required=False, default='u')

    args = arg_parser.parse_args()

    assert args.init.strip() in ["r", "u"] or args.init.strip().startswith("s")


    corpus = CorpusReader(args.corpus, limit=args.limit)

    prepare_data(corpus=corpus, t_file=args.t_file, num_sentences=args.group_size, p_0=args.p_0,
                 file_prefix=args.output_prefix, init=args.init, cond_head=args.cond_head, cond_tok=args.cond_tok, hmm=args.hmm)
