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

class GoldFile(object):

    def __init__(self, file_name, order="ef", sure_only=True):
        self.f = open(file_name, "r")
        self.next = self.__iter_sent()
        assert  order=="ef" or order=="fe"
        if order == "ef":
            self.order = self.order_ef
        else:
            self.order = self.order_fe
        self.sure_only = sure_only

    def order_ef(self, x, y):
        return int(x), int(y)

    def order_fe(self, x, y):
        return int(y), int(x)


    def reset(self):
        self.f.seek(0)

    def __iter_sent(self):
        self.reset()
        buffer = []
        prev = 1
        for line in self.f:
            snt, x, y, t = line.split()
            e, f = self.order(x,y)
            if int(snt) == prev:
                if t == "S" or not self.sure_only:
                    buffer.append((e,f, t))
            else:
                yield buffer
                buffer = []
                if t == "S" or not self.sure_only:
                    buffer.append((e,f, t))
                prev = int(snt)

        yield buffer

    def __iter__(self):
        return self.__iter_sent()


def split_data_get_parameters(corpus, gold_file, file_prefix, train_sentences, ibm1_table, e_voc, f_voc):
    outfile_corpus = LazyFile(file_prefix + ".training")
    order_file = LazyFile(file_prefix + ".training.order")
    subset_c = 0
    for gold_als in gold_file:
        e_toks, f_toks, f_heads, pos, rel, hmm_transitions, order = corpus.next()

        e_str = map(e_voc.__getitem__, e_toks)
        f_str = map(f_voc.__getitem__, f_toks)
        subset_c += 1

        als = {f:set() for f in xrange(1, len(f_toks)+1)}
        for (e,f, _) in gold_als:
            als[f].add(e)

        alignment = []
        for f in xrange(1, len(f_toks)+1):
            if not als[f]:
                alignment.append(0)
            else:
                alignment.append(min(als[f]))

        # reorder alignment
        reordered_alignment = []
        for o in order:
            reordered_alignment.append(alignment[o])


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
        outfile_corpus.write(" ".join(map(str, reordered_alignment)) + "\n")
        outfile_corpus.write(" ".join(map(str, best_ibm1)) + "\n")
        outfile_corpus.write(" ".join(e_str) + "\n")
        outfile_corpus.write(" ".join(f_str) + "\n")
        outfile_corpus.write("\n")
        order_file.write(" ".join(map(str, order)) + "\n")


        if subset_c == train_sentences:
            outfile_corpus.close()
            order_file.close()
            outfile_corpus = LazyFile(file_prefix + ".test")
            order_file = LazyFile(file_prefix + ".test.order")

    outfile_corpus.close()
    order_file.close()

def write_gold_files(gold_file, training_size, output_prefix):
    gold_file.sure_only = False
    gold_file.reset()
    out_file = LazyFile(output_prefix + ".training")
    c = 0
    ct = 0
    for gold_als in gold_file:
        c += 1
        ct += 1
        for al in gold_als:
            out_file.write(" ".join([str(ct)] + map(str, list(al))) + "\n")

        if c == training_size:
            ct = 0
            out_file.close()
            out_file = LazyFile(output_prefix + ".test")
    out_file.close()




if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-corpus", required=True)
    arg_parser.add_argument("-gold", required=True)
    arg_parser.add_argument("-ibm1_table", required=True)
    arg_parser.add_argument("-e_voc", required=True)
    arg_parser.add_argument("-f_voc", required=True)
    arg_parser.add_argument("-training_size", required=True, type=int)
    arg_parser.add_argument("-gold_order", required=True, type=str, default="ef")
    arg_parser.add_argument('-include_p', action='store_true', required=False)
    args = arg_parser.parse_args()

    corpus = CorpusReader(args.corpus)
    gold_file = GoldFile(args.gold, order=args.gold_order, sure_only= not args.include_p)

    ibm1_table = load_ibm1(args.ibm1_table)
    e_vocab = load_vcb(args.e_voc)
    f_vocab = load_vcb(args.f_voc)

    split_data_get_parameters(corpus, gold_file, args.corpus + ".labelled", args.training_size, ibm1_table, e_vocab, f_vocab)

    write_gold_files(gold_file, args.training_size, args.gold)