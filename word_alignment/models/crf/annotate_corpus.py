import argparse
from CorpusReader import CorpusReader
import numpy as np
import codecs

def load_ibm1(t_file):
    t_dict = dict()
    with open(t_file, "r") as infile:
        for line in infile:
            e, f, p = line.strip().split()
            e = int(e)
            f = int(f)
            if e not in t_dict:
                t_dict[e] = dict()
            t_dict[e][f] = float(p)

    return t_dict

def load_vcb(vcb_file):
    vcb = dict()
    with codecs.open(vcb_file, "r", "utf-8") as infile:
        for line in infile:
            i, w, _ = line.split()
            vcb[int(i)] = w
    vcb[0] = "#NULL#"
    return vcb


class LazyFile(object):
    def __init__(self, file_name):
        self.file_name = file_name
        self.f = None

    def write(self, buffer):
        self.f = codecs.open(self.file_name, "w", "utf-8")
        self.f.write(buffer)
        self.write = self.__write

    def __write(self, buffer):
        self.f.write(buffer)

    def close(self):
        if self.f:
            self.f.close()

def split_data_get_parameters(corpus, file_prefix, ibm1_table, e_voc, f_voc):
    outfile_corpus = LazyFile(file_prefix)
    order_file = LazyFile(file_prefix + ".order")
    subset_c = 0
    for e_toks, f_toks, f_heads, pos, rel, hmm_transitions, order in corpus:

        e_str = map(e_voc.__getitem__, e_toks)
        f_str = map(f_voc.__getitem__, f_toks)
        subset_c += 1

        # compute ibm1-features
        best_ibm1 = []
        for e_tok in [0] + e_toks:
            ibm1_e = ibm1_table.get(e_tok, {})
            scores = []
            for j, f_tok in enumerate(f_toks):
                scores.append(ibm1_e.get(f_tok, 0.000000000001))
            best_j = np.argmax(scores)
            best_ibm1.append(best_j)

        # produce subcorpus file
        outfile_corpus.write(" ".join(map(str, e_toks)) + "\n")
        outfile_corpus.write(" ".join(map(str, f_toks)) + "\n")
        outfile_corpus.write(" ".join(map(str, f_heads)) + "\n")
        outfile_corpus.write(" ".join(map(str, pos)) + "\n")
        outfile_corpus.write(" ".join(map(str, rel)) + "\n")
        outfile_corpus.write(" ".join(map(str, hmm_transitions)) + "\n")
        outfile_corpus.write(" ".join(map(str, order)) + "\n")
        outfile_corpus.write(" ".join(map(str, [0]*len(f_toks))) + "\n")
        outfile_corpus.write(" ".join(map(str, best_ibm1)) + "\n")
        outfile_corpus.write(" ".join(e_str) + "\n")
        outfile_corpus.write(" ".join(f_str) + "\n")
        outfile_corpus.write("\n")
        order_file.write(" ".join(map(str, order)) + "\n")

    outfile_corpus.close()
    order_file.close()




if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-corpus", required=True)
    arg_parser.add_argument("-ibm1_table", required=True)
    arg_parser.add_argument("-e_voc", required=True)
    arg_parser.add_argument("-f_voc", required=True)
    arg_parser.add_argument('-include_p', action='store_true', required=False)
    args = arg_parser.parse_args()

    corpus = CorpusReader(args.corpus).iter_sent()
    ibm1_table = load_ibm1(args.ibm1_table)
    e_vocab = load_vcb(args.e_voc)
    f_vocab = load_vcb(args.f_voc)

    split_data_get_parameters(corpus, args.corpus + ".annotated", ibm1_table, e_vocab, f_vocab)