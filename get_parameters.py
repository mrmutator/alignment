from Corpus_Reader import Corpus_Reader
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

    def __init__(self, corpus, e_vocab=Vocab(), f_vocab=Vocab()):
        self.corpus = corpus
        self.trans_probs = defaultdict(set)
        self.al_probs = defaultdict(set)
        self.e_vocab = e_vocab
        self.f_vocab = f_vocab

        self.add_corpus(corpus)

    def write_corpus_format(self, out_file_name):
        outfile_e = open(out_file_name + ".e", "w")
        outfile_f = open(out_file_name + ".f", "w")

        for e_toks, f_toks in corpus:
            outfile_e.write(" ".join([str(self.e_vocab.get_index(w)) for w in e_toks]) + "\n")
            outfile_f.write(" ".join([str(self.f_vocab.get_index(w)) for w in f_toks]) + "\n")

        outfile_e.close()
        outfile_f.close()

    def add_corpus(self, corpus, out_file_name=None):
        for e_toks, f_toks in corpus:
            I = len(e_toks)
            for jmp in range(-I+1, I):
                self.al_probs[I].add(jmp)
            for i, e_tok in enumerate(e_toks):
                self.al_probs[(None, I)].add(i)
                for f_tok in f_toks:
                    self.trans_probs[self.e_vocab.get_index(e_tok)].add(self.f_vocab.get_index(f_tok))


    def initialize_al_uniformly(self):
        al_probs = dict()
        for I in self.al_probs:
            al_probs[I] = dict()
            jmps = self.al_probs[I]
            Z = len(jmps)
            for jmp in jmps:
                al_probs[I][jmp] = 1.0 / Z

        self.al_probs = al_probs

    def initialize_trans_uniformly(self):
        trans_probs = dict()
        for e in self.trans_probs:
            trans_probs[e] = dict()
            fs = self.trans_probs[e]
            Z = len(fs)
            for f in fs:
                trans_probs[e][f] = 1.0 / Z

        self.trans_probs = trans_probs

    def initialize_al_randomly(self):
        al_probs = dict()
        for I in self.al_probs:
            al_probs[I] = dict()
            jmps = self.al_probs[I]
            rand_probs = [random_prob() for _ in xrange(len(jmps))]
            Z = sum(rand_probs)
            for i, jmp in enumerate(jmps):
                al_probs[I][jmp] = rand_probs[i] / Z

        self.al_probs = al_probs

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
        for e_toks, f_toks in corpus:
            c += 1
            outfile_e.write(" ".join([str(self.e_vocab.get_index(w)) for w in e_toks]) + "\n")
            outfile_f.write(" ".join([str(self.f_vocab.get_index(w)) for w in f_toks]) + "\n")

            I = len(e_toks)
            for jmp in range(-I+1, I):
                al_param[(I, jmp)] = self.al_probs[I][jmp]
            for i, e_tok in enumerate(e_toks):
                al_param[((None, I), i)] = self.al_probs[(None, I)][i]
                for f_tok in f_toks:
                    e = self.e_vocab.get_index(e_tok)
                    f = self.f_vocab.get_index(f_tok)
                    trans_param[(e, f)] = self.trans_probs[e][f]
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
    init = arg_parser.add_mutually_exclusive_group(required=True)
    init.add_argument('-uniform', dest='uniform', action='store_true', default=False)
    init.add_argument('-random', dest='random', action='store_true', default=False)
    arg_parser.add_argument("-t_file", required=False, default="")
    arg_parser.add_argument("-e_voc", required=False, default="")
    arg_parser.add_argument("-f_voc", required=False, default="")

    args = arg_parser.parse_args()

    corpus = Corpus_Reader(args.e, args.f, limit=args.limit, strings=True)

    e_vocab= Vocab()
    f_vocab = Vocab()
    if args.e_voc:
        e_vocab = GizaVocab(args.e_voc)
    if args.f_voc:
        f_vocab = GizaVocab(args.f_voc)

    parameters = Parameters(corpus, e_vocab=e_vocab, f_vocab=f_vocab)

    if args.random:
        parameters.initialize_al_randomly()
        if not args.t_file:
            parameters.initialize_trans_randomly()
        else:
            parameters.initialize_trans_t_file(args.t_file)
    elif args.uniform:
        parameters.initialize_al_uniformly()
        if not args.t_file:
            parameters.initialize_trans_uniformly()
        else:
            parameters.initialize_trans_t_file(args.t_file)

    parameters.split_data(corpus, num_sentences=args.group_size, file_prefix=args.output_prefix)

    if args.output_vocab_file:
        parameters.e_vocab.write_vocab(args.output_vocab_file + ".voc.e")
        parameters.f_vocab.write_vocab(args.output_vocab_file + ".voc.f")