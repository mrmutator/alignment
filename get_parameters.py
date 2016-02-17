from Corpus_Reader import Corpus_Reader
from collections import defaultdict
import codecs
import random
import cPickle as pickle

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


class Parameters(object):

    def __init__(self, corpus):
        self.corpus = corpus
        self.trans_probs = defaultdict(set)
        self.al_probs = defaultdict(set)
        self.e_vocab = Vocab()
        self.f_vocab = Vocab()

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


    def initialize_uniformly(self):
        al_probs = dict()
        for I in self.al_probs:
            al_probs[I] = dict()
            jmps = self.al_probs[I]
            Z = len(jmps)
            for jmp in jmps:
                al_probs[I][jmp] = 1.0 / Z

        self.al_probs = al_probs

        trans_probs = dict()
        for e in self.trans_probs:
            trans_probs[e] = dict()
            fs = self.trans_probs[e]
            Z = len(fs)
            for f in fs:
                trans_probs[e][f] = 1.0 / Z

        self.trans_probs = trans_probs

    def initialize_randomly(self):
        al_probs = dict()
        for I in self.al_probs:
            al_probs[I] = dict()
            jmps = self.al_probs[I]
            rand_probs = [random_prob() for _ in xrange(len(jmps))]
            Z = sum(rand_probs)
            for i, jmp in enumerate(jmps):
                al_probs[I][jmp] = rand_probs[i] / Z

        self.al_probs = al_probs

        trans_probs = dict()
        for e in self.trans_probs:
            trans_probs[e] = dict()
            fs = self.trans_probs[e]
            rand_probs = [random_prob() for _ in xrange(len(fs))]
            Z = sum(rand_probs)
            for i, f in enumerate(fs):
                trans_probs[e][f] = rand_probs[i] / Z

        self.trans_probs = trans_probs

    def split_data(self, corpus, num_sentences, file_prefix):
        part_num = 1
        outfile_e = open(file_prefix +".e."+str(part_num), "w")
        outfile_f = open(file_prefix +".f."+str(part_num), "w")
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
                pickle.dump((trans_param, al_param), open(file_prefix +".prms."+str(part_num), "wb"))
                al_param = dict()
                trans_param = dict()
                part_num += 1
                outfile_e = open(file_prefix +".e."+str(part_num), "w")
                outfile_f = open(file_prefix +".f."+str(part_num), "w")


if __name__ == "__main__":
    # create corpus instance
    corpus = Corpus_Reader("data/file.en", "data/file.de", limit=10000, strings=True)
    parameters = Parameters(corpus)
    parameters.initialize_uniformly()
    parameters.split_data(corpus, num_sentences=1000, file_prefix="test/tp")