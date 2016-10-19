import random
import numpy as np
import argparse
from CorpusReader import CorpusReader

def random_prob():
    return random.random() * -1 + 1  # random number between 0 and 1, excluding 0, including 1


class Parameters(object):
    def __init__(self, corpus):
        self.corpus = corpus
        self.cooc = set()

        self.c = 0
        self.add_corpus(corpus)

        self.t_params = dict()


    def add_corpus(self, corpus):
        self.c = 0
        for e_toks, f_toks, _, _, _, _, _ in corpus:
            self.c += 1
            for f in f_toks:
                self.cooc.add((0, f))
                for e in e_toks:
                    self.cooc.add((e, f))

    def initialize_trans_t_file(self, t_file):
        with open(t_file, "r") as infile:
            for line in infile:
                e, f, p = line.strip().split()
                e = int(e)
                f = int(f)
                if (e, f) in self.cooc:
                    self.t_params[(e, f)] = float(p)
        del self.cooc

    def write_params(self, sub_t, out_file_name):
        outfile = open(out_file_name, "w")
        for key in sub_t:
            value = self.t_params[key]
            key_str = ["t"] + map(str, [key[0], key[1], value])
            outfile.write(" ".join(key_str) + "\n")

        outfile.close()

    def split_data_get_parameters(self, corpus, file_prefix, num_sentences, parent_pattern="", tok_pattern="", start_pattern=""):
        order_file = open(file_prefix + ".order", "w")
        subset_id = 1
        outfile_corpus = open(file_prefix + "."  + str(subset_id) + ".sub_feat", "w")
        sub_t = set()
        subset_c = 0
        total = 0
        all_start_conditions = set()
        all_j_conditions = set()

        for e_toks, f_toks, f_heads, pos, rel, hmm_trans, order in corpus:
            subset_c += 1
            total += 1
            order_file.write(" ".join(map(str, order)) + "\n")
            outfile_corpus.write(" ".join([str(w) for w in e_toks]) + "\n")
            outfile_corpus.write(" ".join([str(w) for w in f_toks]) + "\n")
            outfile_corpus.write(" ".join([str(h) for h in f_heads]) + "\n")
            outfile_corpus.write(" ".join([str(h) for h in pos]) + "\n")
            outfile_corpus.write(" ".join([str(h) for h in rel]) + "\n")
            outfile_corpus.write(" ".join([str(h) for h in hmm_trans]) + "\n")
            outfile_corpus.write(" ".join([str(h) for h in order]) + "\n\n")

            for e in e_toks + [0]:
                for f in f_toks:
                    if (e, f) in self.t_params:
                        sub_t.add((e, f))

            I = len(e_toks)
            J = len(f_toks)

            l = [0] + [order[j] - order[f_heads[j]] for j in xrange(1, J)]
            dir = map(np.sign, l)

            start_cons = set()
            # j = 0
            if "p" in start_pattern:
                start_cons.add(("cpos", pos[0]))
            if "r" in start_pattern:
                start_cons.add(("crel", rel[0]))
            if "I" in start_pattern:
                start_cons.add(("I", I))
            if "s" in start_pattern:
                start_cons.add(("j", 0))

            all_start_conditions.add(frozenset(start_cons))

            for j in xrange(1, J):
                parent = f_heads[j]
                j_cons = set()
                if "p" in tok_pattern:
                    j_cons.add(("cpos", pos[j]))
                if "r" in tok_pattern:
                    j_cons.add(("crel", rel[j]))
                if "d" in tok_pattern:
                    j_cons.add(("cdir", dir[j]))
                if "l" in tok_pattern:
                    j_cons.add(("cl", l[j]))
                if "I" in tok_pattern:
                    j_cons.add(("I", I))
                if "h" in tok_pattern:
                    j_cons.add(("chmm", hmm_trans[j]))

                if "p" in parent_pattern:
                    j_cons.add(("ppos", pos[parent]))
                if "r" in parent_pattern:
                    j_cons.add(("prel", rel[parent]))
                if "d" in parent_pattern:
                    j_cons.add(("pdir", dir[parent]))
                if "l" in parent_pattern:
                    j_cons.add(("pl", l[parent]))
                if "h" in parent_pattern:
                    j_cons.add(("phmm", hmm_trans[parent]))

                all_j_conditions.add(frozenset(j_cons))


            if subset_c == num_sentences:
                outfile_corpus.close()
                self.write_params(sub_t, file_prefix + ".params." + str(subset_id))
                if total < self.c:
                    subset_id += 1
                    outfile_corpus = open(file_prefix + "." + str(subset_id) + ".sub_feat", "w")
                    sub_t = set()
                    subset_c = 0

        if subset_c > 0:
            outfile_corpus.close()
            self.write_params(sub_t, file_prefix + ".params." + str(subset_id))

        order_file.close()

        with open(file_prefix + ".features", "w") as outfile:
            for sc in all_start_conditions:
                if sc:
                    outfile.write("j 0\t" + "\t".join([f + " " + str(v) for f, v in sc]) + "\n")

            for jc in all_j_conditions:
                if jc:
                    outfile.write("\t".join([f + " " + str(v) for f, v in jc]) + "\n")


def prepare_data(corpus, t_file, num_sentences, file_prefix="", tj_cond_head="", tj_cond_tok="", start_cond_tok=""):
    parameters = Parameters(corpus)

    parameters.initialize_trans_t_file(t_file)

    parameters.split_data_get_parameters(corpus, file_prefix, num_sentences, parent_pattern=tj_cond_head,
                                         tok_pattern=tj_cond_tok, start_pattern=start_cond_tok)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-corpus", required=True)
    arg_parser.add_argument("-output_prefix", required=True)
    arg_parser.add_argument("-t_file", required=True)
    arg_parser.add_argument("-group_size", required=False, type=int, default=-1)
    arg_parser.add_argument("-tj_cond_tok", required=False, default="", type=str)
    arg_parser.add_argument("-tj_cond_head", required=False, default="", type=str)
    arg_parser.add_argument("-start_cond_tok", required=False, default="", type=str)

    args = arg_parser.parse_args()

    corpus = CorpusReader(args.corpus)

    prepare_data(corpus=corpus, t_file=args.t_file, num_sentences=args.group_size,
                 file_prefix=args.output_prefix, tj_cond_head=args.tj_cond_head,
                 tj_cond_tok=args.tj_cond_tok, start_cond_tok=args.start_cond_tok)
