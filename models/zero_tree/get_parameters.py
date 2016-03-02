from utils.Corpus_Reader import Corpus_Reader
from collections import defaultdict
import codecs
import random
import cPickle as pickle
import argparse

def random_prob():
    return random.random()*-1 + 1 # random number between 0 and 1, excluding 0, including 1

class Vocab(object):

    def __init__(self):
        self.i = 1
        self.w2index = dict()
        self.index2w = dict()

    def get_index(self, w):
        j = self.w2index.get(w, self.i)
        if j == self.i:
            self.w2index[w] = j
            self.index2w[j] = w
            self.i += 1
        return j

    def get_w(self, i):
        return self.index2w.get(i, 0)

    def write_vocab(self, file_name):
        with codecs.open(file_name, "w", "utf-8") as outfile:
            for k,v in self.index2w.iteritems():
                outfile.write(str(k) + "\t" + v + "\n")

class GizaVocab(Vocab):
    def __init__(self, file_name):
        self.w2index = dict()
        self.index2w = dict()
        self.read_file(file_name)

    def read_file(self, file_name):
        with codecs.open(file_name, "r", "utf-8") as infile:
            for line in infile:
                i, w, _= line.strip().split()
                self.w2index[w] = int(i)
                self.index2w[int(i)] = w

    def get_index(self, w):
        return self.w2index[w]



class Parameters(object):

    def __init__(self, corpus, e_vocab=Vocab(), f_vocab=Vocab(), alpha=0):
        self.corpus = corpus
        self.trans_probs = defaultdict(set)
        self.al_prob = defaultdict(set)
        self.e_vocab = e_vocab
        self.f_vocab = f_vocab
        self.alpha = alpha

        self.add_corpus(corpus)

    def write_corpus_format(self, out_file_name):
        outfile_e = open(out_file_name + ".e", "w")
        outfile_f = open(out_file_name + ".f", "w")

        for e_toks, f_toks in corpus:
            outfile_e.write(" ".join([str(self.e_vocab.get_index(w)) for w in e_toks]) + "\n")
            outfile_f.write(" ".join([str(self.f_vocab.get_index(w)) for w in f_toks]) + "\n")

        outfile_e.close()
        outfile_f.close()

    def add_corpus(self, corpus):
        for e_toks, f_pairs in corpus:
            I = len(e_toks)
            J = len(f_pairs)
            for i, e_tok in enumerate(e_toks):
                for j, (f_tok, f_head) in enumerate(f_pairs):
                    self.trans_probs[self.e_vocab.get_index(e_tok)].add(self.f_vocab.get_index(f_tok))
                    self.al_prob[(I, J, f_head, j)].add(i+1)
            for j, (f_tok, f_head) in enumerate(f_pairs):
                self.trans_probs[0].add(self.f_vocab.get_index(f_tok))
                self.al_prob[(I, J, f_head, j)].add(0)



    def initialize_al_randomly(self):
        al_probs = defaultdict(dict)
        for cond in self.al_prob:
            Z = 0
            for v in self.al_prob[cond]:
                p = random_prob()
                al_probs[cond][v] = p
                Z += p
            for v in al_probs[cond]:
                al_probs[cond][v] = al_probs[cond][v] / float(Z)
        self.al_prob = al_probs

    def initialize_trans_randomly(self):
        trans_probs = dict()
        for e in self.trans_probs:
            trans_probs[e] = dict()
            fs = self.trans_probs[e]
            rand_probs = [random_prob() for _ in xrange(len(fs))]
            Z = sum(rand_probs)
            for i, f in enumerate(fs):
                trans_probs[e][f] = rand_probs[i] / Z

        self.trans_probs = trans_probs

    def initialize_trans_t_file(self, t_file):
        trans_dict = dict()
        with open(t_file, "r") as infile:
            for line in infile:
                e, f, p = line.strip().split()
                trans_dict[(int(e), int(f))] = float(p)

        trans_probs = dict()
        for e in self.trans_probs:
            trans_probs[e] = dict()
            Z = 0
            for f in self.trans_probs[e]:

                t_e_f = trans_dict.get((e, f), 0.0000001)
                trans_probs[e][f] = t_e_f
                Z += t_e_f
            for f in self.trans_probs[e]:
                trans_probs[e][f] = trans_probs[e][f] / float(Z)

        self.trans_probs = trans_probs


    def split_data(self, corpus, num_sentences, file_prefix):
        part_num = 1
        outfile_e = open(file_prefix +"."+str(part_num) + ".e", "w")
        outfile_f = open(file_prefix +"."+str(part_num) + ".f", "w")
        trans_param = dict()
        al_param = dict()
        c = 0
        for e_toks, f_pairs in corpus:
            c += 1
            outfile_e.write(" ".join([str(self.e_vocab.get_index(w)) for w in e_toks]) + "\n")
            outfile_f.write(" ".join([str(self.f_vocab.get_index(w)) + "_" + str(h) for w, h in f_pairs]) + "\n")

            I = len(e_toks)
            J = len(f_pairs)
            for i, e_tok in enumerate(e_toks):
                for j, (f_tok, f_head) in enumerate(f_pairs):
                    e = self.e_vocab.get_index(e_tok)
                    f = self.f_vocab.get_index(f_tok)
                    trans_param[(e, f)] = self.trans_probs[e][f]
                    al_param[((I, J, f_head, j), i+1)] = self.al_prob[(I, J, f_head, j)][i+1]
            for j, (f_tok, f_head) in enumerate(f_pairs):
                trans_param[(0,self.f_vocab.get_index(f_tok))] = self.trans_probs[0][self.f_vocab.get_index(f_tok)]
                al_param[((I, J, f_head, j), 0)] = self.al_prob[(I, J, f_head, j)][0]
            if c == num_sentences:
                c = 0
                outfile_f.close()
                outfile_e.close()
                pickle.dump((trans_param, al_param), open(file_prefix +"."+str(part_num) + ".prms.u", "wb"))
                al_param = dict()
                trans_param = dict()
                part_num += 1
                outfile_e = open(file_prefix +"."+str(part_num) + ".e", "w")
                outfile_f = open(file_prefix +"."+str(part_num) + ".f", "w")

        if c > 0:
            outfile_f.close()
            outfile_e.close()
            pickle.dump((trans_param, al_param), open(file_prefix +"."+str(part_num) + ".prms.u", "wb"))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-e", required=True)
    arg_parser.add_argument("-f", required=True)
    arg_parser.add_argument("-limit", required=False, type=int, default=0)
    arg_parser.add_argument("-output_vocab_file", required=False, default="")
    arg_parser.add_argument("-group_size", required=False, type=int, default=-1)
    arg_parser.add_argument("-output_prefix", required=True)
    arg_parser.add_argument("-t_file", required=False, default="")
    arg_parser.add_argument("-e_voc", required=False, default="")
    arg_parser.add_argument("-f_voc", required=False, default="")
    arg_parser.add_argument("-alpha", required=False, default=0.0, type=float)

    args = arg_parser.parse_args()

    corpus = Corpus_Reader(args.e, args.f, limit=args.limit, strings=True, source_dep=True)

    e_vocab= Vocab()
    f_vocab = Vocab()
    if args.e_voc:
        e_vocab = GizaVocab(args.e_voc)
    if args.f_voc:
        f_vocab = GizaVocab(args.f_voc)

    parameters = Parameters(corpus, e_vocab=e_vocab, f_vocab=f_vocab, alpha=args.alpha)


    parameters.initialize_al_randomly()
    if not args.t_file:
        parameters.initialize_trans_randomly()
    else:
        parameters.initialize_trans_t_file(args.t_file)

    parameters.split_data(corpus, num_sentences=args.group_size, file_prefix=args.output_prefix)

    if args.output_vocab_file:
        parameters.e_vocab.write_vocab(args.output_vocab_file + ".voc.e")
        parameters.f_vocab.write_vocab(args.output_vocab_file + ".voc.f")