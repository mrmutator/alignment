import random
import argparse
from CorpusReader import CorpusReader

def random_weight():
    return random.uniform(-1, 1)


class Parameters(object):
    def __init__(self, corpus):
        self.corpus = corpus
        self.cooc = set()
        self.c = 0

        self.add_corpus(corpus)
        self.t_params = dict()

    def add_corpus(self, corpus):
        self.c = 0
        for e_toks, f_toks, f_heads, pos, rel, dir, order in corpus:
            self.c += 1

            for j, f in enumerate(f_toks):
                for e in e_toks + [0]:
                    self.cooc.add((e, f))

    def initialize_trans_t_file(self, t_file):
        with open(t_file, "r") as infile:
            for line in infile:
                e, f, p = line.strip().split()
                e = int(e)
                f = int(f)
                if (e, f) in self.cooc:
                    self.t_params[(e,f)] = float(p)
        del self.cooc

    def write_params(self, sub_t, out_file_name):
        with open(out_file_name, "w") as outfile:
            for key in sub_t:
                value = self.t_params[key]
                key_str = ["t"] + map(str, [key[0], key[1], value])
                outfile.write(" ".join(key_str) + "\n")

    def split_data_get_parameters(self, corpus, file_prefix, num_sentences):
        subset_id = 1
        outfile_corpus = open(file_prefix + "." + str(subset_id) + ".sub_feat", "w")
        order_file = open(file_prefix + ".order", "w")
        sub_t = set()
        subset_c = 0
        total = 0
        for e_toks, f_toks, f_heads, pos, rel, hmm_transitions, order in corpus:
            subset_c += 1
            total += 1

            # produce subcorpus file
            outfile_corpus.write(" ".join(map(str, e_toks)) + "\n")
            outfile_corpus.write(" ".join(map(str, f_toks)) + "\n")
            outfile_corpus.write(" ".join(map(str, f_heads)) + "\n")
            outfile_corpus.write(" ".join(map(str, pos)) + "\n")
            outfile_corpus.write(" ".join(map(str, rel)) + "\n")
            outfile_corpus.write(" ".join(map(str, hmm_transitions)) + "\n")
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
                outfile_corpus.close()
                if total < self.c:
                    outfile_corpus = open(file_prefix + "." + str(subset_id) + ".sub_feat", "w")

        if subset_c > 0:
            self.write_params(sub_t, file_prefix + ".params." + str(subset_id))
            outfile_corpus.close()
        order_file.close()


def prepare_data(corpus, t_file, num_sentences, file_prefix=""):
    parameters = Parameters(corpus)
    parameters.initialize_trans_t_file(t_file)

    parameters.split_data_get_parameters(corpus, file_prefix, num_sentences)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-corpus", required=True)
    arg_parser.add_argument("-t_file", required=True)
    arg_parser.add_argument("-output_prefix", required=True)
    arg_parser.add_argument("-group_size", required=False, type=int, default=-1)
    args = arg_parser.parse_args()

    corpus = CorpusReader(args.corpus)

    prepare_data(corpus=corpus, t_file=args.t_file, num_sentences=args.group_size, file_prefix=args.output_prefix)
