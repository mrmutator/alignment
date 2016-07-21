from collections import defaultdict
import random
import numpy as np
import argparse
from CorpusReader import CorpusReader


def make_smoothed_probs(I, alpha):
    # alpha = 1.0 is uniform
    mid = I / 2
    probs = np.ones(I)
    pn = 1.0 / ((I - 1) + alpha)
    probs = probs * pn
    probs[mid] = alpha * pn
    return probs

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
    def __init__(self, corpus, init_t=1.0, init_c=1.0):
        self.corpus = corpus
        self.cooc = set()
        self.lengths = set()
        self.init_t = init_t
        self.init_c = init_c

        self.c = 0
        self.add_corpus(corpus)

        self.t_params = dict()
        self.s_params = dict()
        self.j_params = dict()
        self.c_params = dict()
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

    def initialize_start_uniformly(self):
        for I in self.lengths:
            start = 1.0 / I
            for i in xrange(I):
                self.s_params[(I, i)] = start

    def initialize_j(self):
        max_I = max(self.lengths)
        num_jumps = 1 + ((max_I - 1) * 2)
        probs = make_smoothed_probs(num_jumps, self.init_t)

        for jmp_i, jmp in enumerate(xrange(-max_I + 1, max_I)):
            value = probs[jmp_i]
            self.j_params[jmp] = value

    def initialize_c(self):
        max_I = max(self.lengths)
        num_jumps = 1 + ((max_I - 1) * 2)
        probs = make_smoothed_probs(num_jumps, self.init_c)

        for jmp_i, jmp in enumerate(xrange(-max_I + 1, max_I)):
            value = probs[jmp_i]
            self.c_params[jmp] = value

    def initialize_trans_t_file(self, t_file):
        with open(t_file, "r") as infile:
            for line in infile:
                e, f, p = line.strip().split()
                e = int(e)
                f = int(f)
                if (e, f) in self.cooc:
                    self.t_params[(e, f)] = float(p)
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
            if pos in self.hmm_cons:
                probs = self.c_params
            else:
                probs = self.j_params
            for jmp in xrange(-max_I + 1, max_I):
                value = probs[jmp]
                key_str = ["j"] + map(str, [pos, jmp, value])
                outfile.write(" ".join(key_str) + "\n")

        outfile.close()

    def split_data_get_parameters(self, corpus, file_prefix, num_sentences, tj_head_con="", tj_tok_con="",
                                  cj_head_con="", cj_tok_con=""):
        self.hmm_cons = set()
        subset_id = 1
        outfile_corpus = open(file_prefix + ".corpus." + str(subset_id), "w")
        sub_t = set()
        sub_lengths_pos = defaultdict(set)
        sub_lengths = set()
        subset_c = 0
        total = 0
        for e_toks, f_toks, f_heads, pos, rel, hmm_transitions, order in corpus:
            dir = [np.sign(order[j] - order[f_heads[j]]) for j in xrange(len(f_toks))]
            subset_c += 1
            total += 1

            outfile_corpus.write(" ".join([str(w) for w in e_toks]) + "\n")
            outfile_corpus.write(" ".join([str(w) for w in f_toks]) + "\n")
            outfile_corpus.write(" ".join([str(h) for h in f_heads]) + "\n")
            # add parameters
            I = len(e_toks)
            sub_lengths.add(I)
            conditions = [-1]
            tree_level = [0]
            for j in range(1, len(f_toks)):
                head_con = tj_head_con
                tok_con = tj_tok_con
                hmm_j = False
                if hmm_transitions[j]:
                    hmm_j = True
                    head_con = cj_head_con
                    tok_con = cj_tok_con
                par = f_heads[j]
                tree_level.append(tree_level[par] + 1)
                orig_tok_pos = order[j]
                orig_head_pos = order[par]
                parent_distance = abs(orig_head_pos - orig_tok_pos)
                con_tok = [None, None, None, None, None, None, None, None, hmm_j]
                if "p" in head_con:
                    con_tok[0] = pos[par]
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
                if "t" in tok_con:
                    con_tok[7] = tree_level[j]

                cond_id = self.cond_voc.get_id(tuple(con_tok))
                conditions.append(cond_id)
                sub_lengths_pos[cond_id].add(I)
                if hmm_j:
                    self.hmm_cons.add(cond_id)
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


def prepare_data(corpus, t_file, num_sentences, file_prefix="", init_c=1.0, init_t=1.0, tj_cond_head="",
                 tj_cond_tok="",
                 cj_cond_head="", cj_con_tok=""):
    parameters = Parameters(corpus, init_c=init_c, init_t=init_t)

    parameters.initialize_start_uniformly()
    parameters.initialize_c()
    parameters.initialize_j()
    parameters.initialize_trans_t_file(t_file)

    parameters.split_data_get_parameters(corpus, file_prefix, num_sentences, tj_head_con=tj_cond_head,
                                         tj_tok_con=tj_cond_tok, cj_head_con=cj_cond_head, cj_tok_con=cj_con_tok)
    with open(file_prefix + ".condvoc", "w") as outfile:
        outfile.write(parameters.cond_voc.get_voc())


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-corpus", required=True)
    arg_parser.add_argument("-output_prefix", required=True)
    arg_parser.add_argument("-t_file", required=True)
    arg_parser.add_argument("-limit", required=False, type=int, default=0)
    arg_parser.add_argument("-group_size", required=False, type=int, default=-1)
    arg_parser.add_argument("-tj_cond_tok", required=False, default="", type=str)
    arg_parser.add_argument("-tj_cond_head", required=False, default="", type=str)
    arg_parser.add_argument('-init_t', required=False, default=1.0, type=float)
    arg_parser.add_argument('-init_c', required=False, default=1.0, type=float)
    arg_parser.add_argument("-cj_cond_tok", required=False, default="", type=str)
    arg_parser.add_argument("-cj_cond_head", required=False, default="", type=str)

    args = arg_parser.parse_args()

    corpus = CorpusReader(args.corpus, limit=args.limit)

    prepare_data(corpus=corpus, t_file=args.t_file, num_sentences=args.group_size,
                 file_prefix=args.output_prefix, init_c=args.init_c, init_t=args.init_t, tj_cond_head=args.tj_cond_head,
                 tj_cond_tok=args.tj_cond_tok,
                 cj_con_tok=args.cj_cond_tok, cj_cond_head=args.cj_cond_head)
