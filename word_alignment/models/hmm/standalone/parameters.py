from collections import defaultdict
import codecs
import random
import numpy as np

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

    def __init__(self, corpus, e_vocab=Vocab(), f_vocab=Vocab(), alpha=0.0, p_0=0.2):
        self.corpus = corpus
        self.trans_probs = defaultdict(set)
        self.lengths = set()
        self.e_vocab = e_vocab
        self.f_vocab = f_vocab
        self.alpha = alpha
        self.p_0 = p_0

        self.c = 0
        self.add_corpus(corpus)

        self.start_param = dict()
        self.dist_param = dict()
        self.trans_param = dict()


    def add_corpus(self, corpus):
        self.c = 0
        for e_toks, f_toks in corpus:
            self.c += 1
            I = len(e_toks)
            self.lengths.add(I)
            for f_tok in f_toks:
                f = self.f_vocab.get_index(f_tok)
                self.trans_probs[0].add(f)
                for i, e_tok in enumerate(e_toks):
                    self.trans_probs[self.e_vocab.get_index(e_tok)].add(f)


    def initialize_start_randomly(self):
        start = dict()
        for I in self.lengths:
            Z = 0
            start[I] = np.zeros(I)
            for i in xrange(I):
                p = random_prob()
                start[I][i] = p
                Z += p
            Z += self.p_0 # p_0 for null word at start
            start[I] = start[I] / Z
        self.start_param = start


    def initialize_dist_randomly(self):
        jumps = dict()
        for jmp in xrange(-max(self.lengths)+1, max(self.lengths)):
            jumps[jmp] = random_prob()

        for I in self.lengths:
            tmp_prob = np.zeros((I, I))
            for i_p in xrange(I):
                norm = np.sum([ jumps[i_pp - i_p] for i_pp in xrange(I)]) + self.p_0
                tmp_prob[i_p, :] = np.array([((jumps[i-i_p] / norm) * (1-self.alpha)) + (self.alpha * (1.0/I))  for i in xrange(I)])
            self.dist_param[I] = tmp_prob

    def initialize_trans_randomly(self):
        trans_probs = dict()
        for e in self.trans_probs:
            trans_probs[e] = dict()
            fs = self.trans_probs[e]
            rand_probs = [random_prob() for _ in xrange(len(fs))]
            Z = sum(rand_probs)
            for i, f in enumerate(fs):
                trans_probs[(e,f)] = rand_probs[i] / Z

        self.trans_param = trans_probs

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
                trans_probs[(e,f)] = trans_probs[e][f] / float(Z)

        self.trans_param = trans_probs


    def split_data_get_parameters(self, corpus, file_prefix):
        outfile_e = open(file_prefix  + ".e", "w")
        outfile_f = open(file_prefix  + ".f", "w")
        for e_toks, f_toks in corpus:
            outfile_e.write(" ".join([str(self.e_vocab.get_index(w)) for w in e_toks]) + "\n")
            outfile_f.write(" ".join([str(self.f_vocab.get_index(w)) for w in f_toks]) + "\n")

        outfile_f.close()
        outfile_e.close()

        return self.trans_param, self.dist_param, self.start_param




def prepare_data(corpus, e_voc=None, f_voc = None, alpha=0.0, p_0=0.2, t_file=None, num_workers=1, file_prefix="", output_vocab_file=None):
    e_vocab= Vocab()
    f_vocab = Vocab()
    if e_voc:
        e_vocab = GizaVocab(e_voc)
    if f_voc:
        f_vocab = GizaVocab(f_voc)

    parameters = Parameters(corpus, e_vocab=e_vocab, f_vocab=f_vocab, alpha=alpha, p_0=p_0)


    parameters.initialize_dist_randomly()
    parameters.initialize_start_randomly()
    if not t_file:
        parameters.initialize_trans_randomly()
    else:
        parameters.initialize_trans_t_file(t_file)

    trans_param, dist_param, start_param = parameters.split_data_get_parameters(corpus, file_prefix=file_prefix)

    if output_vocab_file:
        parameters.e_vocab.write_vocab(output_vocab_file + ".voc.e")
        parameters.f_vocab.write_vocab(output_vocab_file + ".voc.f")
    del parameters
    return trans_param, dist_param, start_param